# 리팩토링된 전체 코드

아래는 수정된 세 운영 코드 파일의 전체 내용입니다. 각 상대 경로에 그대로 저장할 수 있습니다. 기존 패키지와 프로젝트 구조를 사용하며 새 의존성은 없습니다.

## decision_models/common.py

```python
"""Shared cached-pair CLI. No retrieval, augmentation or validation fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def file_sha256(path):
    """Hash in bounded chunks without loading a potentially large pair cache."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fit_standardizer(features):
    """Fit on training pairs only; preserve float64 reduction/float32 storage.

    Constant or nearly constant columns use scale 1, avoiding division by zero
    without changing the existing models' normalization convention.
    """
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1
    return mean, scale


def run_cli(model_class):
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'predict'])
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with np.load(args.pairs, allow_pickle=False) as data:
        names = data['feature_names'].tolist()
        x = data['x']
        if x.ndim != 2 or x.shape[1] != len(names) or not np.isfinite(x).all():
            raise ValueError('Invalid feature matrix')
        tick = time.perf_counter()
        if args.action == 'train':
            if args.model.exists():
                raise FileExistsError(args.model)
            model = model_class().fit(x, data['y'], data['w'], names)
            args.model.mkdir(parents=True)
            model.save(args.model)
            report = {'pair_file_sha256': file_sha256(args.pairs),
                      'training_pairs': len(x), 'features': names,
                      'fit_and_save_seconds': time.perf_counter() - tick,
                      'model': model_class.__name__}
            (args.model / 'training.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        else:
            if args.output is None:
                parser.error('--output is required for predict')
            if args.output.exists():
                raise FileExistsError(args.output)
            model = model_class.load(args.model)
            if names != model.feature_names:
                raise ValueError('Feature order mismatch')
            scores = model.predict(x)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('wb') as handle:
                np.save(handle, scores, allow_pickle=False)


def validate_training(x, y, w):
    x = np.asarray(x, dtype=np.float32)
    y, w = np.asarray(y, dtype=np.float32), np.asarray(w, dtype=np.float32)
    if x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape:
        raise ValueError('Invalid training shapes')
    if not all(np.isfinite(a).all() for a in (x, y, w)) or (w <= 0).any():
        raise ValueError('Training values must be finite; weights positive')
    # Checking membership directly avoids sorting every training label.
    is_positive = y == 1
    is_negative = y == 0
    if not (is_positive | is_negative).all() or not is_positive.any() or not is_negative.any():
        raise ValueError('Both binary classes are required')
    return x, y, w
```

## decision_models/mlp_model.py

```python
"""Independent CPU/GPU MLP scorer with training-only scaling and rowwise inference."""
import json
from pathlib import Path
import numpy as np
import torch
from .common import fit_standardizer, run_cli, validate_training


class MLPDecisionModel:
    def __init__(self, hidden_dims=(64, 32), lr=0.001, weight_decay=1e-4,
                 dropout=0.0, activation='relu', epochs=20, batch_size=2048, device=None):
        self.hidden_dims = list(hidden_dims)
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.activation = activation.lower()
        self.epochs = epochs
        self.batch_size = batch_size
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.feature_names = []

    def build_network(self, input_dim):
        layers = []
        previous_width = input_dim
        for hidden_width in self.hidden_dims:
            layers.append(torch.nn.Linear(previous_width, hidden_width))
            if self.activation == 'gelu':
                layers.append(torch.nn.GELU())
            elif self.activation == 'silu':
                layers.append(torch.nn.SiLU())
            else:
                layers.append(torch.nn.ReLU())
            if self.dropout > 0:
                layers.append(torch.nn.Dropout(self.dropout))
            previous_width = hidden_width
        layers.append(torch.nn.Linear(previous_width, 1))
        return torch.nn.Sequential(*layers)

    def network(self, width):
        return self.build_network(width)

    def fit(self, x, y, w, feature_names):
        x, y, w = validate_training(x, y, w)
        device = self.device
        if device == 'cpu':
            import os
            torch.set_num_threads(min(6, os.cpu_count() or 4))
        self.feature_names = list(feature_names)
        self.mean, self.scale = fit_standardizer(x)
        generator = torch.Generator(device=device if device == 'cuda' else 'cpu').manual_seed(20261016)
        with torch.random.fork_rng(devices=[0] if device == 'cuda' else []):
            torch.manual_seed(20261016)
            if device == 'cuda':
                torch.cuda.manual_seed_all(20261016)
            self.model = self.build_network(x.shape[1]).to(device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            features = torch.from_numpy((x - self.mean) / self.scale).to(device)
            labels = torch.from_numpy(y).to(device)
            weights = torch.from_numpy(w / w.mean()).to(device)
            self.model.train()
            n_samples = len(x)
            for _ in range(self.epochs):
                order = torch.randperm(n_samples, generator=generator, device=device)
                for batch_indices in order.split(self.batch_size):
                    optimizer.zero_grad()
                    logits = self.model(features[batch_indices]).flatten()
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, labels[batch_indices], reduction='none')
                    # Normalize weights globally, then average within each batch.
                    (loss * weights[batch_indices]).mean().backward()
                    optimizer.step()
        self.model.eval()
        return self

    def _predict_batch(self, features):
        """Called inside inference_mode; retain the caller's batch boundaries."""
        scaled = torch.from_numpy((features - self.mean) / self.scale).to(self.device)
        logits = self.model(scaled).flatten()
        return torch.sigmoid(logits).cpu().numpy()

    def predict(self, x, num_threads=None):
        x = np.asarray(x, dtype=np.float32)
        if len(x) == 0:
            return np.empty(0, dtype=np.float32)
        device = self.device
        if device == 'cpu' and num_threads:
            torch.set_num_threads(num_threads)
        with torch.inference_mode():
            if len(x) > 16384 and device == 'cuda':
                # Keep the original CUDA chunk size: changing matrix shapes can
                # change floating-point rounding even with identical weights.
                scores = [self._predict_batch(x[start:start + 16384])
                          for start in range(0, len(x), 16384)]
                return np.concatenate(scores)
            return self._predict_batch(x)

    def save(self, directory):
        directory = Path(directory)
        torch.save(self.model.state_dict(), directory / 'mlp.pt')
        np.savez(directory / 'scaler.npz', mean=self.mean, scale=self.scale)
        (directory / 'features.json').write_text(json.dumps(self.feature_names), encoding='utf-8')
        config = {
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout': self.dropout,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
        (directory / 'mlp_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory, device=None):
        directory = Path(directory)
        cfg_path = directory / 'mlp_config.json'
        kwargs = {}
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            kwargs['hidden_dims'] = cfg.get('hidden_dims', [64, 32])
            kwargs['activation'] = cfg.get('activation', 'relu')
            kwargs['dropout'] = cfg.get('dropout', 0.0)
            kwargs['lr'] = cfg.get('lr', 0.001)
            kwargs['weight_decay'] = cfg.get('weight_decay', 1e-4)
            kwargs['epochs'] = cfg.get('epochs', 20)
            kwargs['batch_size'] = cfg.get('batch_size', 2048)
        result = cls(device=device, **kwargs)
        result.feature_names = json.loads((directory / 'features.json').read_text(encoding='utf-8'))
        with np.load(directory / 'scaler.npz', allow_pickle=False) as scaler:
            result.mean, result.scale = scaler['mean'], scaler['scale']
        result.model = result.build_network(len(result.feature_names))
        state = torch.load(directory / 'mlp.pt', map_location='cpu', weights_only=True)
        result.model.load_state_dict(state)
        result.model.to(result.device)
        result.model.eval()
        return result


if __name__ == '__main__':
    run_cli(MLPDecisionModel)
```

## decision_models/set_model.py

```python
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
    def __init__(self, input_dim, embedding_dim=32, hidden_dim=32):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embedding_dim), torch.nn.ReLU())
        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * embedding_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1))

    def forward(self, x, mask):
        # x: queries x padded candidates x features. No batch normalization.
        encoded = self.encoder(x)
        valid = mask.unsqueeze(-1)
        mean = (encoded * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True)
        maximum = encoded.masked_fill(~valid, float('-inf')).amax(dim=1, keepdim=True)
        context = torch.cat((encoded, mean.expand_as(encoded), maximum.expand_as(encoded)), dim=-1)
        return self.head(context).squeeze(-1)


class SetDecisionModel:
    """One group is one query's candidates, not a vessel or a test batch."""
    def __init__(self, embedding_dim=32, hidden_dim=32, lr=0.001,
                 weight_decay=1e-3, epochs=25, batch_size=64, seed=20261016,
                 device='cpu'):
        if min(embedding_dim, hidden_dim, epochs, batch_size) < 1:
            raise ValueError('Network dimensions, epochs and batch_size must be positive')
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('Invalid optimizer parameters')
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed
        self.device = str(device)
        self.feature_names = []
        self.loss_history = []

    def _network(self, width):
        return CandidateSetNetwork(width, self.embedding_dim, self.hidden_dim)

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
                    losses = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, torch.from_numpy(batch_y).to(device), reduction='none')
                    numerator = (losses * torch.from_numpy(batch_w).to(device) * mask_t).sum()
                    count = int(mask.sum())
                    loss = numerator / count
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(numerator.detach().cpu())
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
            'embedding_dim', 'hidden_dim', 'lr', 'weight_decay', 'epochs', 'batch_size', 'seed')}
        config.update(format_version=1, feature_names=self.feature_names, loss='weighted_pair_bce',
                      pooling=['mean', 'max'], inference='canonical_one_query_at_a_time',
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
        result = cls(device=device, **{key: config[key] for key in (
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
    args = parser.parse_args()
    torch.set_num_threads(6)
    with np.load(args.pairs, allow_pickle=False) as pairs:
        x, names, sizes = pairs['x'], pairs['feature_names'].tolist(), pairs['group_sizes']
        if args.action == 'train':
            if args.model.exists():
                raise FileExistsError(args.model)
            model = SetDecisionModel(epochs=args.epochs, device=args.device).fit(
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
```

