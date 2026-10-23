"""Candidate-set scorer: context never crosses query boundaries.

Weighted BCE and training-only scaling match the independent MLP objective.
Inference evaluates each complete query separately, with a canonical candidate
order, so unrelated queries and candidate permutations cannot affect scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import fit_standardizer, validate_training


def validate_groups(group_sizes, rows):
    sizes = np.asarray(group_sizes)
    if sizes.ndim != 1 or (sizes.size and sizes.dtype.kind not in 'iu'):
        raise ValueError('group_sizes must be a one-dimensional integer array')
    if (sizes <= 0).any() or sum(int(n) for n in sizes) != rows:
        raise ValueError('Positive group_sizes must partition every candidate row')
    return sizes.astype(np.int64)


class CandidateSetNetwork(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim=32, hidden_dim=32, context_mode='mean_max'):
        super().__init__()
        self.context_mode = str(context_mode)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embedding_dim), torch.nn.ReLU())
        if self.context_mode == 'attn_max':
            self.attn = torch.nn.Linear(embedding_dim, 1)
        context_dim = 3 * embedding_dim + (1 if self.context_mode == 'leave_one_out' else 0)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(context_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1))

    def forward(self, x, mask):
        # x: queries x padded candidates x features. No batch normalization.
        encoded = self.encoder(x)
        valid = mask.unsqueeze(-1)
        if self.context_mode == 'leave_one_out':
            B, N, D = encoded.shape
            n_valid = valid.sum(dim=1, keepdim=True)
            has_competitor = (n_valid > 1).float()

            # Mean of other candidates excluding self: (sum - self) / max(1, n - 1)
            total_sum = (encoded * valid).sum(dim=1, keepdim=True)
            denom = torch.clamp(n_valid - 1, min=1.0)
            mean_others = (total_sum - encoded) / denom
            diff_mean = torch.where(has_competitor > 0, encoded - mean_others, torch.zeros_like(encoded))

            # Maximum of other candidates excluding self: top-2 with duplicates preserved
            e_masked = encoded.masked_fill(~valid, float('-inf'))
            k = min(2, N)
            if k == 1:
                diff_max = torch.zeros_like(encoded)
            else:
                vals, _ = torch.topk(e_masked, k=k, dim=1)
                top1 = vals[:, 0:1, :]
                top2 = torch.where(has_competitor > 0, vals[:, 1:2, :], top1)
                is_top1 = (encoded == top1) & valid
                max_others = torch.where(is_top1, top2, top1)
                diff_max = torch.where(has_competitor > 0, encoded - max_others, torch.zeros_like(encoded))

            flag = has_competitor.expand(B, N, 1)
            context = torch.cat((encoded, diff_mean, diff_max, flag), dim=-1)
        elif self.context_mode == 'attn_max':
            # Softmax attention pooling over candidates, strictly excluding padding
            attn_logits = self.attn(encoded).masked_fill(~valid, float('-inf'))
            attn_weights = torch.softmax(attn_logits, dim=1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            pooled = (encoded * attn_weights).sum(dim=1, keepdim=True)
            maximum = encoded.masked_fill(~valid, float('-inf')).amax(dim=1, keepdim=True)
            context = torch.cat((encoded, pooled.expand_as(encoded), maximum.expand_as(encoded)), dim=-1)
        else:
            mean = (encoded * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True)
            maximum = encoded.masked_fill(~valid, float('-inf')).amax(dim=1, keepdim=True)
            context = torch.cat((encoded, mean.expand_as(encoded), maximum.expand_as(encoded)), dim=-1)
        return self.head(context).squeeze(-1)


class SetDecisionModel:
    """One group is one query's candidates, not a vessel or a test batch."""
    def __init__(self, embedding_dim=32, hidden_dim=32, lr=0.001,
                 weight_decay=1e-3, epochs=25, batch_size=64, seed=20261016,
                 device='cpu', context_mode='mean_max', loss_mode='pair_bce'):
        if min(embedding_dim, hidden_dim, epochs, batch_size) < 1:
            raise ValueError('Network dimensions, epochs and batch_size must be positive')
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('Invalid optimizer parameters')
        if loss_mode not in ('pair_bce', 'row_bce', 'row_ce'):
            raise ValueError(f"Unknown loss_mode: {loss_mode}")
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.device = str(device)
        self.context_mode = str(context_mode)
        self.loss_mode = str(loss_mode)
        self.feature_names = []
        self.loss_history = []

    def _network(self, width):
        return CandidateSetNetwork(width, self.embedding_dim, self.hidden_dim, self.context_mode)

    @staticmethod
    def _matrix(x, width):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all():
            raise ValueError('Feature matrix must have the saved width and finite values')
        return x

    def fit(self, x, y, w, feature_names, *, group_sizes):
        x, y, w = validate_training(x, y, w)
        sizes = validate_groups(group_sizes, len(x))
        names = list(feature_names)
        if not names or len(names) != x.shape[1] or len(set(names)) != len(names):
            raise ValueError('Unique feature names must match the feature matrix')
        offsets = np.r_[0, np.cumsum(sizes)]
        if any(y[a:b].sum() != 1 for a, b in zip(offsets[:-1], offsets[1:])):
            raise ValueError('Each training query must contain exactly one positive candidate')
        self.feature_names = names
        self.mean, self.scale = fit_standardizer(x)
        scaled = (x - self.mean) / self.scale
        weights = w / w.mean()
        rng = np.random.default_rng(self.seed)
        device = torch.device(self.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        self.loss_history = []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            self.model = self._network(x.shape[1]).to(device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.model.train()
            for _ in range(self.epochs):
                order = rng.permutation(len(sizes))
                epoch_loss, epoch_pairs = 0., 0
                for start in range(0, len(order), self.batch_size):
                    groups = order[start:start + self.batch_size]
                    shape = (len(groups), int(sizes[groups].max()))
                    batch_x = np.zeros((*shape, x.shape[1]), dtype=np.float32)
                    batch_y = np.zeros(shape, dtype=np.float32)
                    batch_w = np.zeros(shape, dtype=np.float32)
                    mask = np.zeros(shape, dtype=bool)
                    # Pad only within this training batch. The mask excludes
                    # padding from both candidate context and the BCE average.
                    for batch_row, group in enumerate(groups):
                        first, stop = offsets[group:group + 2]
                        candidate_count = stop - first
                        batch_x[batch_row, :candidate_count] = scaled[first:stop]
                        batch_y[batch_row, :candidate_count] = y[first:stop]
                        batch_w[batch_row, :candidate_count] = weights[first:stop]
                        mask[batch_row, :candidate_count] = True
                    mask_t = torch.from_numpy(mask).to(device)
                    optimizer.zero_grad()
                    logits = self.model(torch.from_numpy(batch_x).to(device), mask_t)
                    batch_y_t = torch.from_numpy(batch_y).to(device)
                    batch_w_t = torch.from_numpy(batch_w).to(device)
                    if self.loss_mode == 'row_bce':
                        losses = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, batch_y_t, reduction='none')
                        row_counts = mask_t.sum(dim=1).clamp(min=1.0)
                        row_mean_bce = (losses * mask_t).sum(dim=1) / row_counts
                        row_weights = batch_w_t[:, 0]
                        loss = (row_mean_bce * row_weights).sum() / row_weights.sum().clamp(min=1e-8)
                        step_stat = float((row_mean_bce * row_weights).sum().detach().cpu())
                        count = len(groups)
                    elif self.loss_mode == 'row_ce':
                        masked_logits = logits.masked_fill(~mask_t, float('-inf'))
                        log_probs = torch.log_softmax(masked_logits, dim=-1)
                        target_idx = batch_y_t.argmax(dim=-1, keepdim=True)
                        row_ce = -log_probs.gather(dim=-1, index=target_idx).squeeze(-1)
                        row_weights = batch_w_t[:, 0]
                        loss = (row_ce * row_weights).sum() / row_weights.sum().clamp(min=1e-8)
                        step_stat = float((row_ce * row_weights).sum().detach().cpu())
                        count = len(groups)
                    else:
                        losses = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, batch_y_t, reduction='none')
                        numerator = (losses * batch_w_t * mask_t).sum()
                        count = int(mask.sum())
                        loss = numerator / count
                        step_stat = float(numerator.detach().cpu())

                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    loss.backward()
                    optimizer.step()
                    epoch_loss += step_stat
                    epoch_pairs += count
                self.loss_history.append(epoch_loss / epoch_pairs)
        self.model.eval()
        return self

    def predict(self, x, num_threads=None, *, group_sizes=None):
        if group_sizes is None:
            raise ValueError('Set scorer requires query boundaries; use predict_group or predict_groups')
        return self.predict_groups(x, group_sizes, num_threads=num_threads)

    def predict_group(self, x, num_threads=None):
        return self.predict_groups(x, [len(x)] if len(x) else [], num_threads=num_threads)

    def predict_groups(self, x, group_sizes, num_threads=None):
        x = self._matrix(x, len(self.feature_names))
        sizes = validate_groups(group_sizes, len(x))
        if num_threads is not None and torch.device(self.device).type == 'cpu':
            if not 1 <= num_threads <= 6:
                raise ValueError('num_threads must be between 1 and 6')
            torch.set_num_threads(num_threads)
        scores = np.empty(len(x), dtype=np.float32)
        offset = 0
        self.model.eval()
        with torch.inference_mode():
            for n in sizes:
                group = x[offset:offset + n]
                # Canonical reduction order also avoids floating-point mean changes
                # when callers permute candidates. Duplicate vectors get equal scores.
                order = np.lexsort(group.T[::-1])
                features = torch.from_numpy(((group[order] - self.mean) / self.scale)[None]).to(self.device)
                mask = torch.ones((1, int(n)), dtype=torch.bool, device=self.device)
                values = torch.sigmoid(self.model(features, mask)[0]).cpu().numpy()
                scores[offset + order] = values
                offset += n
        return scores

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = [directory / name for name in ('set_model.pt', 'set_scaler.npz', 'set_config.json')]
        if any(path.exists() for path in paths):
            raise FileExistsError('Set model files already exist; choose a new directory')
        config = {key: getattr(self, key) for key in (
            'embedding_dim', 'hidden_dim', 'lr', 'weight_decay', 'epochs', 'batch_size', 'seed', 'context_mode', 'loss_mode')}
        config.update(format_version=1, feature_names=self.feature_names, loss=self.loss_mode,
                      pooling=['mean', 'max'] if self.context_mode == 'mean_max' else ['attn', 'max'] if self.context_mode == 'attn_max' else ['leave_one_out_mean_diff', 'leave_one_out_max_diff', 'has_competitor'],
                      inference='canonical_one_query_at_a_time',
                      loss_history=self.loss_history)
        # Python's file handle also supports non-ASCII Windows paths.
        with paths[0].open('xb') as handle:
            torch.save(self.model.state_dict(), handle)
        np.savez(paths[1], mean=self.mean, scale=self.scale)
        paths[2].write_text(json.dumps(config, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory, device='cpu'):
        directory = Path(directory)
        config = json.loads((directory / 'set_config.json').read_text(encoding='utf-8'))
        if config.get('format_version') != 1:
            raise ValueError('Unsupported set model format')
        context_mode = config.get('context_mode', 'mean_max')
        loss_mode = config.get('loss_mode', 'pair_bce')
        result = cls(device=device, context_mode=context_mode, loss_mode=loss_mode, **{key: config[key] for key in (
            'embedding_dim', 'hidden_dim', 'lr', 'weight_decay', 'epochs', 'batch_size', 'seed')})
        result.feature_names = config['feature_names']
        with np.load(directory / 'set_scaler.npz', allow_pickle=False) as scaler:
            result.mean, result.scale = scaler['mean'], scaler['scale']
        width = len(result.feature_names)
        if (result.mean.shape != (width,) or result.scale.shape != (width,)
                or not np.isfinite(result.mean).all() or not np.isfinite(result.scale).all()
                or (result.scale <= 0).any()):
            raise ValueError('Invalid saved scaler')
        result.model = result._network(width)
        result.model.load_state_dict(torch.load(directory / 'set_model.pt', map_location='cpu', weights_only=True))
        result.model.to(device).eval()
        result.loss_history = config.get('loss_history', [])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'predict'])
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--context-mode', choices=['mean_max', 'leave_one_out', 'attn_max'], default='mean_max')
    parser.add_argument('--loss-mode', choices=['pair_bce', 'row_bce', 'row_ce'], default='pair_bce')
    args = parser.parse_args()
    torch.set_num_threads(6)
    with np.load(args.pairs, allow_pickle=False) as pairs:
        x, names, sizes = pairs['x'], pairs['feature_names'].tolist(), pairs['group_sizes']
        if args.action == 'train':
            if args.model.exists():
                raise FileExistsError(args.model)
            model = SetDecisionModel(epochs=args.epochs, device=args.device,
                                     context_mode=args.context_mode, loss_mode=args.loss_mode).fit(
                x, pairs['y'], pairs['w'], names, group_sizes=sizes)
            model.save(args.model)
        else:
            if args.output is None:
                parser.error('--output is required for predict')
            model = SetDecisionModel.load(args.model, device=args.device)
            if names != model.feature_names:
                raise ValueError('Feature order mismatch')
            scores = model.predict_groups(x, sizes)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('xb') as handle:
                np.save(handle, scores, allow_pickle=False)


if __name__ == '__main__':
    main()
