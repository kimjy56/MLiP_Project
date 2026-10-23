"""Independent-row matching using only the official catalog and fitted assets."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

import concurrent.futures
import os
import time

try:
    import torch
    try:
        from model.torch_indexer import TorchNgramIndexer
    except ImportError:
        from torch_indexer import TorchNgramIndexer
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode
from threadpoolctl import threadpool_limits

try:
    from model.name_processing import NameFeatureIndex, STRUCTURE_FEATURES, PHONETIC_FEATURES, name_view, rapidfuzz_phone_similarity, expand_korean_name
    from model.ngram_retrieval import CharacterNgramIndex
except ImportError:
    from name_processing import NameFeatureIndex, STRUCTURE_FEATURES, PHONETIC_FEATURES, name_view, rapidfuzz_phone_similarity, expand_korean_name
    from ngram_retrieval import CharacterNgramIndex

CHANNELS = ('S', 'N', 'P', 'T')


def runtime_int(name, default, maximum=None, minimum=1):
    value = int(os.environ.get(name, default))
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return min(value, maximum) if maximum else value


def worker_count(name):
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            available = min(available, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return runtime_int(name, min(4, available), min(6, available))


def validate_channels(channels):
    if channels is None:
        return None
    if len(set(channels)) != len(channels) or not set(channels).issubset(CHANNELS):
        raise ValueError('Retrieval channels must be a unique subset of S/N/P/T')
    return [c for c in CHANNELS if c in channels]


FEATURES = [
    'ratio', 'roman_ratio', 'phonetic_ratio',
    'phonetic_partial', 'length_ratio', 'digits_equal', 'digits_missing',
    'gross_gap', 'international_gap', 'best_gap', 'gross_exact',
    'international_exact', 'gross_missing', 'international_missing',
    'type_probability', 'flag_probability',
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
    return isinstance(value, str) and any('가' <= c <= '힣' for c in value)


def roman(value):
    return norm(unidecode(value if isinstance(value, str) else ''))


def phonetic_from_roman(s):
    for old, new in [('PH', 'F'), ('SH', 'S'), ('CH', 'C'), ('TH', 'T'),
                     ('NG', 'N'), ('CK', 'K'), ('QU', 'K')]:
        s = s.replace(old, new)
    s = s.translate(str.maketrans({'V': 'B', 'F': 'P', 'L': 'R', 'G': 'K',
                                 'D': 'T', 'Z': 'J', 'C': 'K'}))
    s = re.sub('[AEIOUWYH]', '', s)
    return re.sub(r'(.)\1+', r'\1', s)


def phonetic(value):
    return phonetic_from_roman(roman(value))


def prepare_row(row, expand_korean=False):
    row = dict(row)
    name = row.get('vessel_name')
    is_ko = hangul(name)
    r_name = expand_korean_name(name) if (expand_korean and is_ko) else name
    r_roman = roman(r_name)
    row['_name'] = (norm(name), r_roman, phonetic_from_roman(r_roman), is_ko)
    row['_view'] = name_view(r_name, expand_korean=False)
    row['_tons'] = (number(row.get('gross_tonnage')), number(row.get('international_gross_tonnage')))
    return row


def number(value):
    try:
        n = float(value)
    except (ValueError, TypeError):
        return np.nan
    return n if np.isfinite(n) and n > 0 else np.nan


def text(value):
    return value if isinstance(value, str) else ''


def make_assets(train, catalog, name_mode='baseline', phonetic_search=0, retrieval_channels=None,
                use_train_lookup=True, relax_tonnage_penalty=False, hangul_ignore_type_flag=False,
                korean_expansion=False, hangul_soft_meta=False):
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
    if not isinstance(use_train_lookup, bool):
        raise ValueError('use_train_lookup must be a boolean')
    return {'version': 5, 'search_algorithm': os.environ.get('VESSEL_SEARCH_ALGORITHM', 'rapidfuzz'), 'features': feature_names(name_mode),
            'use_train_lookup': use_train_lookup,
            'relax_tonnage_penalty': bool(relax_tonnage_penalty),
            'hangul_ignore_type_flag': bool(hangul_ignore_type_flag),
            'hangul_soft_meta': bool(hangul_soft_meta),
            'korean_expansion': bool(korean_expansion),
            'name_mode': name_mode, 'phonetic_search': int(phonetic_search),
            'retrieval_channels': validate_channels(retrieval_channels),
            'calls': {k: next(iter(v)) for k, v in calls.items() if len(v) == 1},
            'aliases': {k: sorted(v) for k, v in names.items()}, 'tables': tables}


class CatalogIndex:
    def __init__(self, catalog, assets, use_train_lookup=None,
                 relax_tonnage_penalty=None, hangul_ignore_type_flag=None,
                 korean_expansion=None, hangul_soft_meta=None):
        required = {'vessel_id', 'vessel_name', 'gross_tonnage', 'vessel_type', 'flag_name'}
        if not required.issubset(catalog.columns):
            raise ValueError('Catalog schema is incomplete')
        if (catalog.empty or catalog.vessel_id.isna().any()
                or catalog.vessel_id.eq('').any() or not catalog.vessel_id.is_unique):
            raise ValueError('Catalog IDs must be nonempty and unique')
        # Explicit ordering gives deterministic score ties and retrieval ties.
        self.catalog = catalog.sort_values('vessel_id', kind='stable').reset_index(drop=True)
        self.assets = assets
        self.use_train_lookup = (assets.get('use_train_lookup', True)
                                 if use_train_lookup is None else use_train_lookup)
        if not isinstance(self.use_train_lookup, bool):
            raise ValueError('use_train_lookup must be a boolean')
        self.relax_tonnage_penalty = (assets.get('relax_tonnage_penalty', False)
                                      if relax_tonnage_penalty is None else relax_tonnage_penalty) or (os.environ.get('VESSEL_RELAX_TONNAGE_PENALTY', '0') == '1')
        self.hangul_ignore_type_flag = (assets.get('hangul_ignore_type_flag', False)
                                        if hangul_ignore_type_flag is None else hangul_ignore_type_flag) or (os.environ.get('VESSEL_HANGUL_IGNORE_TYPE_FLAG', '0') == '1')
        self.hangul_soft_meta = (assets.get('hangul_soft_meta', False)
                                 if hangul_soft_meta is None else hangul_soft_meta) or (os.environ.get('VESSEL_HANGUL_SOFT_META', '0') == '1')
        self.korean_expansion = (assets.get('korean_expansion', False)
                                 if korean_expansion is None else korean_expansion) or (os.environ.get('VESSEL_KOREAN_EXPANSION', '0') == '1')
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
        # Historical assets can represent different search algorithms; require an explicit migration choice.
        self.search_algorithm = assets.get('search_algorithm') or os.environ.get('VESSEL_LEGACY_SEARCH_ALGORITHM')
        if self.search_algorithm not in {'rapidfuzz', 'tfidf'}:
            raise ValueError('Set saved search_algorithm or VESSEL_LEGACY_SEARCH_ALGORITHM to rapidfuzz/tfidf; old assets do not identify the algorithm')
        self.torch_names = self.torch_romans = self.torch_phones = self.torch_keys = None
        channels = self.retrieval_channels if self.retrieval_channels is not None else ('S', 'P', 'T')
        if self.search_algorithm == 'tfidf':
            if not HAS_TORCH:
                raise RuntimeError('Saved TF-IDF retrieval requires torch; no algorithm fallback allowed')
            device = os.environ.get('VESSEL_SEARCH_DEVICE', 'cpu')
            if 'S' in channels:
                self.torch_names = TorchNgramIndexer(self.names, device=device)
            if 'P' in channels:
                self.torch_romans = TorchNgramIndexer(self.romans, device=device)
                self.torch_phones = TorchNgramIndexer(self.phones, device=device)
                if self.name_index is not None and self.phonetic_search:
                    self.torch_keys = TorchNgramIndexer(self.name_index.search_keys, device=device)
        self.name_lengths = np.array([len(x) for x in self.names])
        self.digits = [re.sub(r'\D', '', x) for x in self.names]
        self.gt = np.array([number(x) for x in self.catalog.gross_tonnage])
        self.types = self.catalog.vessel_type.fillna('').tolist()
        self.flags = self.catalog.flag_name.fillna('').tolist()
        valid = np.flatnonzero(np.isfinite(self.gt))
        self.gt_order = valid[np.argsort(self.gt[valid], kind='stable')]
        self.sorted_gt = self.gt[self.gt_order]
        self.type_indices = {k: np.flatnonzero(np.array(self.types) == k)
                             for k in sorted(set(self.types))}

    def direct(self, row):
        if not self.use_train_lookup:
            return None
        vessel = self.assets['calls'].get(text(row.get('call_sign')))
        return vessel if vessel in self.lookup else None

    def nearest(self, value, count=40, subset=None):
        if not np.isfinite(value):
            return []
        if subset is None:
            p = np.searchsorted(self.sorted_gt, value)
            subset = self.gt_order[max(0, p-count):p+count]
        gaps = np.abs(self.gt[subset] - value)
        order = np.lexsort((subset, np.nan_to_num(gaps, nan=np.inf)))[:count]
        return subset[order].tolist()

    def channel_candidates(self, row, channel):
        if channel not in CHANNELS:
            raise ValueError(f'Unknown retrieval channel: {channel}')
        q, r_q, p_q, ko = row['_name'] if '_name' in row else prepare_row(row, self.korean_expansion)['_name']
        result = set()
        if channel == 'N':
            if self.ngram_index is None:
                raise ValueError('N-gram index not configured')
            return self.ngram_index.search(row.get('vessel_name'), 60 if ko else 24)
        if channel == 'S':
            if q and not ko:
                if self.torch_names is not None:
                    result.update(row['_retrieval']['names'] if '_retrieval' in row and 'names' in row['_retrieval'] else self.torch_names.search(q, count=24))
                else:
                    result.update(i for _, _, i in process.extract(q, self.names, scorer=fuzz.ratio, limit=24, score_cutoff=35))
            return sorted(result)
        if channel == 'P':
            if ko:
                if r_q:
                    if self.torch_romans is not None:
                        result.update(row['_retrieval']['romans'] if '_retrieval' in row and 'romans' in row['_retrieval'] else self.torch_romans.search(r_q, count=60))
                    else:
                        result.update(i for _, _, i in process.extract(r_q, self.romans, scorer=fuzz.ratio, limit=60, score_cutoff=35))
                if p_q:
                    if self.torch_phones is not None:
                        result.update(row['_retrieval']['phones'] if '_retrieval' in row and 'phones' in row['_retrieval'] else self.torch_phones.search(p_q, count=60))
                    else:
                        result.update(i for _, _, i in process.extract(p_q, self.phones, scorer=fuzz.ratio, limit=60, score_cutoff=35))
                if q and self.phonetic_search:
                    key = (row['_view'] if '_view' in row else name_view(row['vessel_name'])).search_key
                    if key:
                        if self.torch_keys is not None:
                            result.update(row['_retrieval']['keys'] if '_retrieval' in row and 'keys' in row['_retrieval'] else self.torch_keys.search(key, count=self.phonetic_search))
                        else:
                            result.update(i for _, _, i in process.extract(
                                key, self.name_index.search_keys, scorer=fuzz.ratio,
                                limit=self.phonetic_search, score_cutoff=35))
            return sorted(result)
        probs = self.assets['tables']['vessel_type'].get(text(row.get('vessel_type')), {})
        likely_types = sorted(probs, key=lambda x: (-probs[x], x))[:2]
        val_gross = number(row.get('gross_tonnage'))
        val_inter = number(row.get('international_gross_tonnage'))
        ton_cols = ['gross_tonnage']
        if val_gross != val_inter and not (np.isnan(val_gross) and np.isnan(val_inter)):
            ton_cols.append('international_gross_tonnage')
        for col in ton_cols:
            value = number(row.get(col))
            result.update(self.nearest(value, 60 if ko else 24))
            if ko:
                for typ in likely_types:
                    result.update(self.nearest(value, 40, self.type_indices.get(typ, np.array([], dtype=int))))
        return sorted(result)

    def prepare_retrieval(self, rows, use_direct=False):
        for row in rows:
            row['_retrieval'] = {}
        for field, index, position, count in (
                ('names', self.torch_names, 0, 24), ('romans', self.torch_romans, 1, 60),
                ('phones', self.torch_phones, 2, 60), ('keys', self.torch_keys, None, self.phonetic_search)):
            if index is None:
                continue
            selected = [r for r in rows if not (use_direct and self.direct(r)) and
                        (r['_name'][3] if field != 'names' else not r['_name'][3])]
            queries = [r['_view'].search_key if position is None else r['_name'][position] for r in selected]
            for row, result in zip(selected, index.batch_search(queries, count)):
                row['_retrieval'][field] = result

    def prune_candidates_by_tonnage(self, row, candidates, max_candidates=30):
        if len(candidates) <= max_candidates:
            return sorted(candidates)
        gross, inter = row['_tons'] if '_tons' in row else prepare_row(row, self.korean_expansion)['_tons']
        ton = inter if inter > 0 else (gross if gross > 0 else -1.0)
        alt_ton = gross if (gross > 0 and gross != ton) else -1.0
        q, r_q, p_q, ko = row['_name'] if '_name' in row else prepare_row(row, self.korean_expansion)['_name']

        alias_cands = set()
        if self.use_train_lookup:
            for v in self.assets.get('aliases', {}).get(norm(row.get('vessel_name')), []):
                if v in self.lookup:
                    alias_cands.add(self.lookup[v])

        scored = []
        for c_i in candidates:
            if c_i in alias_cands:
                scored.append((-999.0, c_i))
                continue
            c_gt = self.gt[c_i]
            if ton > 0 and c_gt > 0:
                diff = abs(c_gt - ton)
                if alt_ton > 0:
                    diff = min(diff, abs(c_gt - alt_ton))
                rel = diff / max(1.0, c_gt)
                # 한글 선박의 경우 9.8% 초과 시 페널티 대폭 부여 (AR-02 기준)
                if self.relax_tonnage_penalty:
                    ton_pen = rel
                else:
                    ton_pen = rel if rel <= 0.098 else rel * 3.5
            elif ton > 0 and c_gt <= 0:
                ton_pen = 0.25
            else:
                ton_pen = 0.0

            if ko:
                c_p = self.phones[c_i]
                sim = rapidfuzz_phone_similarity(p_q, c_p) if (p_q and c_p) else 0.0
            else:
                c_n = self.names[c_i]
                sim = (fuzz.ratio(q, c_n) / 100.0) if (q and c_n) else 0.0

            priority = sim - 1.5 * ton_pen
            scored.append((-priority, c_i))

        scored.sort()
        chosen = [c_i for _, c_i in scored[:max_candidates]]
        return sorted(chosen)

    def candidates(self, row, use_aliases=True, max_candidates=None):
        result = set()
        channels = ('S', 'P', 'T') if self.retrieval_channels is None else self.retrieval_channels
        for channel in channels:
            result.update(self.channel_candidates(row, channel))
        if self.use_train_lookup and use_aliases:
            result.update(self.lookup[v] for v in self.assets['aliases'].get(norm(row.get('vessel_name')), [])
                          if v in self.lookup)
        if not result:
            return [0]
        if max_candidates is None:
            max_candidates = runtime_int('VESSEL_MAX_CANDIDATES', 0, minimum=0)
        if max_candidates > 0 and len(result) > max_candidates:
            return self.prune_candidates_by_tonnage(row, result, max_candidates)
        return sorted(result)

    def features(self, row, candidates, include_name_features=True):
        row = row if '_name' in row else prepare_row(row, self.korean_expansion)
        name = row.get('vessel_name')
        q, r, p, ko = row['_name']
        names = [self.names[i] for i in candidates]
        romans = [self.romans[i] for i in candidates]
        phones = [self.phones[i] for i in candidates]
        def similarity(query, choices, scorer):
            if not query:
                return np.zeros(len(choices), dtype=np.float32)
            return process.cdist([query], choices, scorer=scorer, workers=1)[0] / 100
        gt = self.gt[candidates]
        gross, inter = row['_tons']
        def gap(value):
            return np.nan_to_num(np.abs(gt-value) / np.maximum(np.maximum(gt, value), 1), nan=2.0)
        g1 = gap(gross)
        g2 = g1 if (gross == inter or (not np.isfinite(gross) and not np.isfinite(inter))) else gap(inter)
        tp = self.assets['tables']['vessel_type'].get(text(row.get('vessel_type')), {})
        fp = self.assets['tables']['flag_name'].get(text(row.get('flag_name')), {})
        names_sim = similarity(q, names, fuzz.ratio)
        if self.hangul_ignore_type_flag and ko:
            type_probs = [0.0 for _ in candidates]
            flag_probs = [0.0 for _ in candidates]
        elif self.hangul_soft_meta and ko:
            type_probs = [0.5 * tp.get(self.types[i], 0.0) for i in candidates]
            flag_probs = [max(0.5, fp.get(self.flags[i], 0.0)) if sim >= 0.85 else 0.5 * fp.get(self.flags[i], 0.0)
                          for i, sim in zip(candidates, names_sim)]
        else:
            type_probs = [tp.get(self.types[i], 0) for i in candidates]
            flag_probs = [fp.get(self.flags[i], 0) for i in candidates]
        digits = re.sub(r'\D', '', q)
        n = len(candidates)
        base = np.column_stack([
            names_sim,
            similarity(r, romans, fuzz.ratio), similarity(p, phones, fuzz.ratio),
            similarity(p, phones, fuzz.partial_ratio),
            [min(len(q), len(s))/max(len(q), len(s), 1) for s in names],
            [float(digits == self.digits[i]) for i in candidates],
            np.full(n, not digits), g1, g2, np.minimum(g1, g2),
            (g1 < 1e-6), (g2 < 1e-6), np.full(n, not np.isfinite(gross)),
            np.full(n, not np.isfinite(inter)),
            type_probs,
            flag_probs,
            np.full(n, not text(row.get('flag_name'))), np.full(n, hangul(name)),
            np.full(n, len(q)), self.name_lengths[candidates],
        ]).astype(np.float32)
        if self.name_mode == 'baseline' or not include_name_features:
            return base
        return np.column_stack([base, self.name_index.features(name, candidates, self.name_mode, query=row['_view'])])


class Matcher:
    @threadpool_limits.wrap(limits=1)
    def __init__(self, catalog, model_dir, use_train_lookup=None,
                 relax_tonnage_penalty=None, hangul_ignore_type_flag=None,
                 korean_expansion=None, hangul_soft_meta=None):
        model_dir = Path(model_dir)
        assets = json.loads((model_dir / 'assets.json').read_text(encoding='utf-8'))
        if assets['features'] != feature_names(assets.get('name_mode', 'baseline')):
            raise ValueError('Feature version mismatch')
        self.index = CatalogIndex(catalog, assets, use_train_lookup=use_train_lookup,
                                  relax_tonnage_penalty=relax_tonnage_penalty,
                                  hangul_ignore_type_flag=hangul_ignore_type_flag,
                                  korean_expansion=korean_expansion,
                                  hangul_soft_meta=hangul_soft_meta)
        if assets.get('decision_model') == 'set_mlp':
            try:
                from decision_models.set_model import SetDecisionModel
            except ImportError:
                from model.set_model import SetDecisionModel
            self.model = SetDecisionModel.load(model_dir, device='cpu')
            if self.model.feature_names != assets['features']:
                raise ValueError('Set model and assets feature names differ')
            return
        cb_path = model_dir / 'catboost.cbm'
        mlp_path = model_dir / 'mlp.pt'

        scorers = []
        if cb_path.is_file():
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier()
            cb.load_model(str(cb_path))
            class _CatBoostScorer:
                def __init__(self, model):
                    self.model = model
                def predict(self, x, num_threads=None):
                    x = np.asarray(x, dtype=np.float32)
                    if not len(x):
                        return np.empty(0, dtype=np.float32)
                    threads = num_threads or 4
                    return self.model.predict_proba(x, thread_count=threads)[:, 1]
            scorers.append(('cb', _CatBoostScorer(cb)))

        if mlp_path.is_file():
            try:
                from decision_models.mlp_model import MLPDecisionModel
            except ImportError:
                from model.mlp_model import MLPDecisionModel
            mlp = MLPDecisionModel.load(model_dir, device='cpu')
            scorers.append(('mlp', mlp))

        if len(scorers) >= 2:
            class _EnsembleScorer:
                def __init__(self, model_list):
                    self.scorers = model_list
                    self.weights = [1.0 / len(model_list)] * len(model_list)
                def predict(self, x, num_threads=None):
                    x = np.asarray(x, dtype=np.float32)
                    if not len(x):
                        return np.empty(0, dtype=np.float32)
                    total_scores = np.zeros(len(x), dtype=np.float32)
                    for (_, s), w in zip(self.scorers, self.weights):
                        total_scores += w * s.predict(x, num_threads=num_threads)
                    return total_scores
            self.model = _EnsembleScorer(scorers)
        elif len(scorers) == 1:
            self.model = scorers[0][1]
        else:
            # Read through Python so Unicode Windows project paths also work.
            self.model = lgb.Booster(model_str=(model_dir / 'ranker.txt').read_text(encoding='utf-8'))
            if self.model.feature_name() != assets['features']:
                raise ValueError('Model and assets feature names differ')

    def predict(self, queries, use_direct=True, diagnostics=False):
        """Use only query columns; legacy label diagnostics run after prediction.

        use_direct=False disables calls only for historical experiments.
        Matcher(use_train_lookup=False) disables both calls and name aliases.
        New validation code uses predict_details with label-free queries.
        """
        pred, candidates, direct = self._predict(queries, use_direct, diagnostics)
        if diagnostics:
            truth = queries.get('target_vessel_id', [None] * len(queries))
            recall = [gold in ids for gold, ids in zip(truth, candidates)]
            return pred, recall, direct
        return pred

    def predict_details(self, queries, use_direct=True):
        """Public holdout diagnostics: predictions, candidate IDs, lookup flags.

        No labels are needed, and no rows or predictions are written to disk.
        """
        return self._predict(queries, use_direct, True)

    @threadpool_limits.wrap(limits=1)
    def _predict(self, queries, use_direct, details):
        missing = set(QUERY_COLUMNS) - set(queries.columns)
        if missing:
            raise ValueError(f'Missing input columns: {sorted(missing)}')
        records = queries.loc[:, QUERY_COLUMNS].to_dict('records')
        n_records = len(records)
        if n_records == 0:
            return [], [], []

        batch_size = runtime_int('VESSEL_PREDICT_BATCH', 64)
        workers = worker_count('VESSEL_PREDICT_WORKERS')
        predictions, candidate_ids, direct_flags = [], [], []
        timings = {'prepare_seconds': 0., 'feature_seconds': 0., 'score_seconds': 0.}
        pred_max_cands = runtime_int('VESSEL_PREDICT_CANDIDATES', 30, minimum=0)
        def build(row):
            direct = self.index.direct(row) if use_direct else None
            if direct:
                return direct, None, None
            candidates = self.index.candidates(row, max_candidates=pred_max_cands)
            return None, candidates, self.index.features(row, candidates)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, n_records, batch_size):
                tick = time.monotonic()
                rows = [prepare_row(r, self.index.korean_expansion) for r in records[start:start + batch_size]]
                # Fixed catalog transform only: no test-fitted statistics or cross-row features.
                self.index.prepare_retrieval(rows, use_direct)
                timings['prepare_seconds'] += time.monotonic() - tick
                tick = time.monotonic()
                built = list(executor.map(build, rows)) if len(rows) > 10 and workers > 1 else [build(r) for r in rows]
                timings['feature_seconds'] += time.monotonic() - tick
                tick = time.monotonic()
                matrices = [f for _, _, f in built if f is not None]
                if matrices:
                    score_threads = runtime_int('VESSEL_SCORE_THREADS', 1, 6)
                    if hasattr(self.model, 'predict_groups'):
                        scores = self.model.predict_groups(
                            np.concatenate(matrices), [len(f) for f in matrices], num_threads=score_threads)
                    else:
                        scores = self.model.predict(np.concatenate(matrices), num_threads=score_threads)
                else:
                    scores = []
                offset = 0
                for direct, candidates, features in built:
                    if direct:
                        pred, ids = direct, [direct]
                    else:
                        n = len(candidates)
                        pred = self.index.ids[candidates[int(np.argmax(scores[offset:offset+n]))]]
                        ids = [self.index.ids[i] for i in candidates] if details else None
                        offset += n
                    predictions.append(pred)
                    if details:
                        candidate_ids.append(ids)
                        direct_flags.append(direct is not None)
                timings['score_seconds'] += time.monotonic() - tick
        self.last_timings = timings
        return predictions, candidate_ids, direct_flags
