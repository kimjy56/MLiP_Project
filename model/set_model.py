"""Candidate-set scorer: context never crosses query boundaries.

CandidateSetNetwork, SetDecisionModel, and EnsembleSetScorer for standalone inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch


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
        encoded = self.encoder(x)
        valid = mask.unsqueeze(-1)
        if self.context_mode == 'leave_one_out':
            B, N, D = encoded.shape
            n_valid = valid.sum(dim=1, keepdim=True)
            has_competitor = (n_valid > 1).float()
            total_sum = (encoded * valid).sum(dim=1, keepdim=True)
            denom = torch.clamp(n_valid - 1, min=1.0)
            mean_others = (total_sum - encoded) / denom
            diff_mean = torch.where(has_competitor > 0, encoded - mean_others, torch.zeros_like(encoded))

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


class _EnsembleSetScorer:
    def __init__(self, models: List[SetDecisionModel]):
        self.models = models
        self.feature_names = models[0].feature_names

    def predict_groups(self, x, group_sizes, num_threads=None):
        total = np.zeros(len(x), dtype=np.float32)
        for m in self.models:
            total += m.predict_groups(x, group_sizes, num_threads=num_threads)
        return total / len(self.models)


class SetDecisionModel:
    def __init__(self, embedding_dim=32, hidden_dim=32, lr=0.001,
                 weight_decay=1e-3, epochs=25, batch_size=64, seed=20261016,
                 device='cpu', context_mode='mean_max', loss_mode='pair_bce'):
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.device = str(device)
        self.context_mode = str(context_mode)
        self.loss_mode = str(loss_mode)
        self.feature_names = []
        self.loss_history = []
        self.mean = None
        self.scale = None
        self.model = None

    def _network(self, width):
        return CandidateSetNetwork(width, self.embedding_dim, self.hidden_dim, self.context_mode)

    @staticmethod
    def _matrix(x, width):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all():
            raise ValueError('Feature matrix must have the saved width and finite values')
        return x

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
                order = np.lexsort(group.T[::-1])
                rev = np.empty_like(order)
                rev[order] = np.arange(n)
                ordered = (group[order] - self.mean) / self.scale
                ordered_t = torch.from_numpy(ordered).unsqueeze(0).to(self.device)
                mask_t = torch.ones((1, n), dtype=torch.bool, device=self.device)
                group_scores = self.model(ordered_t, mask_t).squeeze(0).cpu().numpy()
                scores[offset:offset + n] = group_scores[rev]
                offset += n
        return scores

    @classmethod
    def load(cls, directory, device='cpu'):
        directory = Path(directory)
        folds_dir = directory / 'folds'
        if folds_dir.is_dir():
            fold_subdirs = sorted([d for d in folds_dir.iterdir() if d.is_dir()])
            if fold_subdirs:
                models = [cls.load(d, device=device) for d in fold_subdirs]
                return _EnsembleSetScorer(models)

        config = json.loads((directory / 'set_config.json').read_text(encoding='utf-8'))
        if config.get('format_version') != 1:
            raise ValueError('Unsupported set model format')
        context_mode = config.get('context_mode', 'mean_max')
        loss_mode = config.get('loss_mode', 'row_bce')
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
        with (directory / 'set_model.pt').open('rb') as handle:
            result.model.load_state_dict(torch.load(handle, map_location=device))
        result.model.to(device).eval()
        return result
