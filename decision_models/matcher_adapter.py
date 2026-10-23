"""Reuse the existing retrieval and tie-breaking code with either scorer."""
import json
from pathlib import Path
import sys
_model_path = str(Path(__file__).resolve().parents[1] / 'model')
if _model_path not in sys.path:
    sys.path.insert(0, _model_path)
from model.matcher import Matcher, CatalogIndex, feature_names


class DecisionMatcher(Matcher):
    def __init__(self, catalog, assets_path, scorer):
        assets = json.loads(Path(assets_path).read_text(encoding='utf-8'))
        if assets['features'] != feature_names(assets.get('name_mode', 'baseline')):
            raise ValueError('Feature version mismatch')
        if scorer.feature_names != assets['features']:
            raise ValueError('Scorer feature order differs from retrieval assets')
        self.index = CatalogIndex(catalog, assets)
        self.model = scorer
