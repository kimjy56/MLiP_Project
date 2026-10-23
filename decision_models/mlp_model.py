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
