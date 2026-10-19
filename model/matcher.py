"""Independent-row matching using only the official catalog and fitted assets."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode
from name_processing import NameFeatureIndex, STRUCTURE_FEATURES, PHONETIC_FEATURES, name_view
from ngram_retrieval import CharacterNgramIndex

CHANNELS = ('S', 'N', 'P', 'T')


def validate_channels(channels):
    if channels is None:
        return None
    if len(set(channels)) != len(channels) or not set(channels).issubset(CHANNELS):
        raise ValueError('Retrieval channels must be a unique subset of S/N/P/T')
    return [c for c in CHANNELS if c in channels]

FEATURES = [
    'exact', 'ratio', 'partial', 'roman_ratio', 'phonetic_ratio',
    'phonetic_partial', 'length_ratio', 'digits_equal', 'digits_missing',
    'gross_gap', 'international_gap', 'best_gap', 'gross_exact',
    'international_exact', 'gross_missing', 'international_missing',
    'catalog_gross_missing', 'type_probability', 'flag_probability',
    'flag_missing', 'is_hangul', 'name_length', 'catalog_name_length',
]
QUERY_COLUMNS = ['sample_id', 'vessel_name', 'call_sign', 'gross_tonnage',
                 'international_gross_tonnage', 'flag_name', 'vessel_type']


def feature_names(mode='baseline'):
    if mode not in {'baseline', 'structure', 'all'}:
        raise ValueError(f'Unknown name feature mode: {mode}')
    return FEATURES + (STRUCTURE_FEATURES if mode != 'baseline' else []) + (
        PHONETIC_FEATURES if mode == 'all' else [])


def norm(value):
    if not isinstance(value, str):
        return ''
    return ''.join(c for c in unicodedata.normalize('NFKC', value).upper() if c.isalnum())


def hangul(value):
    # Same definition as the supplied educational baseline.
    return isinstance(value, str) and any('가' <= c <= '힣' for c in value)


def roman(value):
    return norm(unidecode(value if isinstance(value, str) else ''))


def phonetic(value):
    s = roman(value)
    for old, new in [('PH', 'F'), ('SH', 'S'), ('CH', 'C'), ('TH', 'T'),
                     ('NG', 'N'), ('CK', 'K'), ('QU', 'K')]:
        s = s.replace(old, new)
    s = s.translate(str.maketrans({'V': 'B', 'F': 'P', 'L': 'R', 'G': 'K',
                                 'D': 'T', 'Z': 'J', 'C': 'K'}))
    s = re.sub('[AEIOUWYH]', '', s)
    return re.sub(r'(.)\1+', r'\1', s)


def number(value):
    try:
        n = float(value)
    except (ValueError, TypeError):
        return np.nan
    return n if np.isfinite(n) and n > 0 else np.nan


def text(value):
    return value if isinstance(value, str) else ''


def make_assets(train, catalog, name_mode='baseline', phonetic_search=0, retrieval_channels=None):
    """Must be called on a training fold only during validation."""
    c = catalog.set_index('vessel_id')
    calls, names = defaultdict(set), defaultdict(set)
    for row in train.to_dict('records'):
        if text(row['call_sign']):
            calls[text(row['call_sign'])].add(row['target_vessel_id'])
        if norm(row['vessel_name']):
            names[norm(row['vessel_name'])].add(row['target_vessel_id'])
    tables = {}
    for col in ['vessel_type', 'flag_name']:
        counts = defaultdict(Counter)
        # Repeated port calls must not multiply a ship's contribution.
        for row in train.drop_duplicates(['target_vessel_id', col]).to_dict('records'):
            q = text(row[col])
            if q:
                counts[q][text(c.loc[row['target_vessel_id'], col])] += 1
        tables[col] = {q: {k: n / sum(v.values()) for k, n in v.items()}
                       for q, v in counts.items()}
    return {'version': 3, 'features': feature_names(name_mode),
            'name_mode': name_mode, 'phonetic_search': int(phonetic_search),
            'retrieval_channels': validate_channels(retrieval_channels),
            'calls': {k: next(iter(v)) for k, v in calls.items() if len(v) == 1},
            'aliases': {k: sorted(v) for k, v in names.items()}, 'tables': tables}


class CatalogIndex:
    def __init__(self, catalog, assets):
        required = {'vessel_id', 'vessel_name', 'gross_tonnage', 'vessel_type', 'flag_name'}
        if not required.issubset(catalog.columns):
            raise ValueError('Catalog schema is incomplete')
        if catalog.empty or catalog.vessel_id.isna().any() or not catalog.vessel_id.is_unique:
            raise ValueError('Catalog IDs must be nonempty and unique')
        # Explicit ordering gives deterministic score ties and retrieval ties.
        self.catalog = catalog.sort_values('vessel_id', kind='stable').reset_index(drop=True)
        self.assets = assets
        self.name_mode = assets.get('name_mode', 'baseline')
        self.phonetic_search = assets.get('phonetic_search', 0)
        self.retrieval_channels = validate_channels(assets.get('retrieval_channels'))
        if not isinstance(self.phonetic_search, int) or self.phonetic_search < 0:
            raise ValueError('phonetic_search must be a nonnegative integer')
        self.feature_names = feature_names(self.name_mode)
        self.ids = self.catalog.vessel_id.astype(str).tolist()
        self.lookup = {v: i for i, v in enumerate(self.ids)}
        self.names = [norm(x) for x in self.catalog.vessel_name]
        self.romans = [roman(x) for x in self.catalog.vessel_name]
        self.phones = [phonetic(x) for x in self.catalog.vessel_name]
        self.name_index = (NameFeatureIndex(self.catalog.vessel_name)
                           if self.name_mode != 'baseline' or self.phonetic_search else None)
        self.ngram_index = (CharacterNgramIndex(self.catalog.vessel_name)
                            if self.retrieval_channels and 'N' in self.retrieval_channels else None)
        self.digits = [re.sub(r'\D', '', x) for x in self.names]
        self.gt = np.array([number(x) for x in self.catalog.gross_tonnage])
        self.types = self.catalog.vessel_type.fillna('').tolist()
        self.flags = self.catalog.flag_name.fillna('').tolist()
        valid = np.flatnonzero(np.isfinite(self.gt))
        self.gt_order = valid[np.argsort(self.gt[valid], kind='stable')]
        self.type_indices = {k: np.flatnonzero(np.array(self.types) == k)
                             for k in sorted(set(self.types))}

    def direct(self, row):
        vessel = self.assets['calls'].get(text(row.get('call_sign')))
        return vessel if vessel in self.lookup else None

    def nearest(self, value, count=40, subset=None):
        if not np.isfinite(value):
            return []
        if subset is None:
            p = np.searchsorted(self.gt[self.gt_order], value)
            subset = self.gt_order[max(0, p-count):p+count]
        gaps = np.abs(self.gt[subset] - value)
        order = np.lexsort((subset, np.nan_to_num(gaps, nan=np.inf)))[:count]
        return subset[order].tolist()

    def channel_candidates(self, row, channel):
        if channel not in CHANNELS:
            raise ValueError(f'Unknown retrieval channel: {channel}')
        q = norm(row.get('vessel_name'))
        ko = hangul(row.get('vessel_name'))
        result = set()
        if channel == 'N':
            if self.ngram_index is None:
                raise ValueError('N-gram index not configured')
            return self.ngram_index.search(row.get('vessel_name'), 60 if ko else 24)
        if channel == 'S':
            if q and not ko:
                result.update(i for _, _, i in process.extract(q, self.names, scorer=fuzz.ratio, limit=24))
            return sorted(result)
        if channel == 'P':
            if ko:
                for query, choices in [(roman(row.get('vessel_name')), self.romans),
                                       (phonetic(row.get('vessel_name')), self.phones)]:
                    if query:
                        result.update(i for _, _, i in process.extract(
                            query, choices, scorer=fuzz.ratio, limit=60))
            if q and self.phonetic_search:
                key = name_view(row['vessel_name']).search_key
                if key:
                    result.update(i for _, _, i in process.extract(
                        key, self.name_index.search_keys, scorer=fuzz.ratio,
                        limit=self.phonetic_search))
            return sorted(result)
        probs = self.assets['tables']['vessel_type'].get(text(row.get('vessel_type')), {})
        likely_types = sorted(probs, key=lambda x: (-probs[x], x))[:2]
        for col in ['gross_tonnage', 'international_gross_tonnage']:
            value = number(row.get(col))
            result.update(self.nearest(value, 60 if ko else 24))
            if ko:
                for typ in likely_types:
                    result.update(self.nearest(value, 40, self.type_indices.get(typ, np.array([], dtype=int))))
        return sorted(result)

    def candidates(self, row, use_aliases=True):
        result = set()
        channels = ('S', 'P', 'T') if self.retrieval_channels is None else self.retrieval_channels
        for channel in channels:
            result.update(self.channel_candidates(row, channel))
        if use_aliases:
            result.update(self.lookup[v] for v in self.assets['aliases'].get(norm(row.get('vessel_name')), [])
                          if v in self.lookup)
        return sorted(result) if result else [0]

    def features(self, row, candidates, include_name_features=True):
        name = row.get('vessel_name')
        q, r, p = norm(name), roman(name), phonetic(name)
        names = [self.names[i] for i in candidates]
        romans = [self.romans[i] for i in candidates]
        phones = [self.phones[i] for i in candidates]
        def similarity(query, choices, scorer):
            if not query:
                return np.zeros(len(choices), dtype=np.float32)
            return process.cdist([query], choices, scorer=scorer, workers=1)[0] / 100
        gt = self.gt[candidates]
        gross, inter = number(row.get('gross_tonnage')), number(row.get('international_gross_tonnage'))
        def gap(value):
            return np.nan_to_num(np.abs(gt-value) / np.maximum(np.maximum(gt, value), 1), nan=2.0)
        g1, g2 = gap(gross), gap(inter)
        tp = self.assets['tables']['vessel_type'].get(text(row.get('vessel_type')), {})
        fp = self.assets['tables']['flag_name'].get(text(row.get('flag_name')), {})
        digits = re.sub(r'\D', '', q)
        n = len(candidates)
        base = np.column_stack([
            [float(bool(q) and q == s) for s in names],
            similarity(q, names, fuzz.ratio), similarity(q, names, fuzz.partial_ratio),
            similarity(r, romans, fuzz.ratio), similarity(p, phones, fuzz.ratio),
            similarity(p, phones, fuzz.partial_ratio),
            [min(len(q), len(s))/max(len(q), len(s), 1) for s in names],
            [float(digits == self.digits[i]) for i in candidates],
            np.full(n, not digits), g1, g2, np.minimum(g1, g2),
            (g1 < 1e-6), (g2 < 1e-6), np.full(n, not np.isfinite(gross)),
            np.full(n, not np.isfinite(inter)), ~np.isfinite(gt),
            [tp.get(self.types[i], 0) for i in candidates],
            [fp.get(self.flags[i], 0) for i in candidates],
            np.full(n, not text(row.get('flag_name'))), np.full(n, hangul(name)),
            np.full(n, len(q)), [len(s) for s in names],
        ]).astype(np.float32)
        if self.name_mode == 'baseline' or not include_name_features:
            return base
        return np.column_stack([base, self.name_index.features(name, candidates, self.name_mode)])


class Matcher:
    def __init__(self, catalog, model_dir):
        model_dir = Path(model_dir)
        assets = json.loads((model_dir / 'assets.json').read_text(encoding='utf-8'))
        if assets['features'] != feature_names(assets.get('name_mode', 'baseline')):
            raise ValueError('Feature version mismatch')
        self.index = CatalogIndex(catalog, assets)
        self.model = lgb.Booster(model_file=str(model_dir / 'ranker.txt'))
        if self.model.feature_name() != assets['features']:
            raise ValueError('Model and assets feature names differ')

    def predict(self, queries, use_direct=True, diagnostics=False):
        missing = set(QUERY_COLUMNS) - set(queries.columns)
        if missing:
            raise ValueError(f'Missing input columns: {sorted(missing)}')
        predictions, recalled, direct_flags = [], [], []
        # No fitting, accumulated aliases, query aggregation, or mutable cache.
        for row in queries.to_dict('records'):
            direct = self.index.direct(row) if use_direct else None
            if direct:
                predictions.append(direct)
                recalled.append(direct == row.get('target_vessel_id'))
                direct_flags.append(True)
                continue
            candidates = self.index.candidates(row)
            features = self.index.features(row, candidates)
            scores = self.model.predict(features, num_threads=1)
            predictions.append(self.index.ids[candidates[int(np.argmax(scores))]])
            recalled.append(row.get('target_vessel_id') in {self.index.ids[i] for i in candidates})
            direct_flags.append(False)
        if diagnostics:
            return predictions, recalled, direct_flags
        return predictions
