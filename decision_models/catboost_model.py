"""Independent CatBoost candidate scorer; python -m decision_models.catboost_model."""
import json
from pathlib import Path
import numpy as np
from .common import run_cli, validate_training


class CatBoostDecisionModel:
    def __init__(self, task_type='CPU', thread_count=None, **kwargs):
        import os
        self.task_type = task_type
        self.thread_count = thread_count or min(6, os.cpu_count() or 4)
        self.feature_names = []
        self.custom_params = kwargs

    def fit(self, x, y, w, feature_names):
        from catboost import CatBoostClassifier
        x, y, w = validate_training(x, y, w)
        self.feature_names = list(feature_names)
        params = {
            'iterations': 350,
            'depth': 6,
            'learning_rate': 0.05,
            'loss_function': 'Logloss',
            'random_seed': 20261016,
            'verbose': False,
            'allow_writing_files': False
        }
        params.update(self.custom_params)
        if self.task_type == 'GPU':
            params['task_type'] = 'GPU'
        else:
            params['task_type'] = 'CPU'
            params['thread_count'] = self.thread_count
        self.model = CatBoostClassifier(**params)
        self.model.fit(x, y, sample_weight=w)
        return self

    def predict(self, x, num_threads=None):
        x = np.asarray(x, dtype=np.float32)
        if not len(x):
            return np.empty(0, dtype=np.float32)
        threads = num_threads or self.thread_count
        return self.model.predict_proba(x, thread_count=threads)[:, 1]

    def save(self, directory):
        directory = Path(directory)
        self.model.save_model(str(directory / 'catboost.cbm'))
        (directory / 'features.json').write_text(json.dumps(self.feature_names), encoding='utf-8')

    @classmethod
    def load(cls, directory):
        from catboost import CatBoostClassifier
        directory = Path(directory)
        result = cls()
        result.feature_names = json.loads((directory / 'features.json').read_text(encoding='utf-8'))
        result.model = CatBoostClassifier()
        result.model.load_model(str(directory / 'catboost.cbm'))
        return result


if __name__ == '__main__':
    run_cli(CatBoostDecisionModel)
