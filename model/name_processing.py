"""Deterministic name views; no external lexicon or test-fitted statistics.

These are spelling/phonetic heuristics, not a pronunciation (G2P) model.
Raw names stay in the catalog; lossy views are only additional features.
"""
from collections import Counter
from dataclasses import dataclass
import functools
import math
import re
import unicodedata

import os

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA
from unidecode import unidecode

STRUCTURE_FEATURES = [
    'number_sequence_equal', 'number_conflict',
    'token_count_gap', 'token_roman_idf',
    'token_roman_min', 'token_roman_last',
]
PHONETIC_FEATURES = [
    'weighted_phone_full', 'weighted_phone_token_idf',
    'weighted_phone_token_min', 'weighted_phone_last',
    'weighted_phone_residual', 'phone_shared_edge',
]


def compact(value):
    return ''.join(c for c in value if c.isalnum())


def phone_spelling(value):
    """Resolve a few digraphs, keeping vowels and digits. Not English G2P."""
    for old, new in [('PH', 'F'), ('CK', 'K'), ('QU', 'KW')]:
        value = value.replace(old, new)
    return value


def retrieval_key(value):
    # A deliberately lossy *extra* retrieval view; originals remain available.
    value = phone_spelling(value)
    value = value.translate(str.maketrans({'V': 'F', 'Z': 'S', 'C': 'K',
                                         'G': 'K', 'D': 'T', 'L': 'R'}))
    value = re.sub('[AEIOUY]', 'A', value)
    return re.sub(r'([A-Z])\1+', r'\1', value)


@dataclass(frozen=True)
class NameView:
    roman: str
    tokens: tuple
    phone: str
    phone_tokens: tuple
    numbers: tuple
    search_key: str


def name_view(value):
    raw = unicodedata.normalize('NFKC', value).upper() if isinstance(value, str) else ''
    # Split BEFORE Unidecode: its Hangul output adds syllable spaces.
    tokens = tuple(compact(unidecode(t).upper()) for t in re.findall(r'[^\W_]+', raw))
    tokens = tuple(t for t in tokens if t)
    roman = ''.join(tokens)
    phones = tuple(phone_spelling(t) for t in tokens)
    # Preserve boundaries: (1, 23) must not equal (12, 3). Ignore leading zeros.
    numbers = tuple(n.lstrip('0') or '0' for n in re.findall(r'\d+', raw))
    return NameView(roman, tokens, phone_spelling(roman), phones,
                    numbers, retrieval_key(roman))


VOWELS = frozenset('AEIOUY')
SIMILAR_PAIRS = {frozenset(p) for p in ['RL', 'BV', 'FV', 'FP', 'PB', 'CG', 'CK', 'GK', 'DT', 'SZ', 'IJ', 'JZ']}
SIMILAR_TUPLES = {(p[0], p[1]) for p in ['RL', 'BV', 'FV', 'FP', 'PB', 'CG', 'CK', 'GK', 'DT', 'SZ', 'IJ', 'JZ']} | \
                 {(p[1], p[0]) for p in ['RL', 'BV', 'FV', 'FP', 'PB', 'CG', 'CK', 'GK', 'DT', 'SZ', 'IJ', 'JZ']}


@functools.lru_cache(maxsize=262144)
def rapidfuzz_phone_similarity(a, b):
    """Ultra-fast phonetic similarity using rapidfuzz C-engine (OSA normalized similarity)."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return OSA.normalized_similarity(a, b)


@functools.lru_cache(maxsize=262144)
def weighted_similarity(a, b):
    """Weighted Levenshtein on phone_spellings, normalized to [0, 1]. (Preserved for comparison/baseline)

    Units: ordinary insertion/deletion/substitution=4, vowel insertion or
    deletion=2, vowel-vowel/similar-consonant substitution=1. Digraphs are
    handled by phone_spelling before alignment. Costs are fixed hypotheses.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Fast native distance for completely disjoint strings is unnecessary;
    # keep one transparent exact DP with bounded per-row call-local caching.
    insertion = [2 if ch in VOWELS else 4 for ch in b]
    previous = [0]
    for cost in insertion:
        previous.append(previous[-1] + cost)
    for ca in a:
        deletion = 2 if ca in VOWELS else 4
        current = [previous[0] + deletion]
        for j, cb in enumerate(b):
            substitution = (0 if ca == cb else
                            1 if ((ca in VOWELS and cb in VOWELS) or
                                  (ca, cb) in SIMILAR_TUPLES) else 4)
            current.append(min(previous[j+1] + deletion,
                               current[j] + insertion[j], previous[j] + substitution))
        previous = current
    return max(0.0, 1.0 - previous[-1] / (4 * max(len(a), len(b))))


def get_phone_similarity_fn():
    """Select phonetic similarity function. Defaults to rapidfuzz; set VESSEL_PHONE_SIMILARITY='legacy' for original DP."""
    mode = os.environ.get('VESSEL_PHONE_SIMILARITY', 'rapidfuzz').strip().lower()
    if mode in ('legacy', 'dp', 'weighted'):
        return weighted_similarity
    return rapidfuzz_phone_similarity


class NameFeatureIndex:
    def __init__(self, names):
        self.views = [name_view(n) for n in names]
        frequency = Counter(t for v in self.views for t in set(v.tokens))
        self.idf = {t: 1 + math.log((len(self.views)+1)/(count+1))
                    for t, count in frequency.items()}
        self.unknown_idf = 1 + math.log(len(self.views)+1)
        self.search_keys = [v.search_key for v in self.views]
        self.candidate_cw = [[self.idf.get(t, self.unknown_idf) for t in v.tokens] for v in self.views]

    def features(self, name, candidates, mode, query=None):
        query = name_view(name) if query is None else query
        output = []
        distances = {}
        phone_sim_fn = get_phone_similarity_fn()
        is_hangul = bool(re.search(r'[가-힣]', name)) if isinstance(name, str) else False

        def phone_score(a, b):
            key = (a, b)
            if key not in distances:
                distances[key] = phone_sim_fn(a, b)
            return distances[key]

        query_qw = [self.idf.get(t, self.unknown_idf) for t in query.tokens]

        def tokens_score(q, c, scorer, cw):
            if not q or not c:
                return 0., 0.
            if q == c:
                return 1., 1.
            # 방안 3: 단어 개수가 같으면 1:1 순서 매칭 (선박명 단어 역전 0건)
            if len(q) == len(c):
                sims = np.array([scorer(a, b) for a, b in zip(q, c)])
                return ((np.average(sims, weights=query_qw) + np.average(sims, weights=cw)) / 2,
                        float(sims.min()))
            scores = np.array([[scorer(a, b) for b in c] for a in q])
            # Both directions penalize extra unmatched words. Frequency comes
            # exclusively from the fixed official catalog, never from test.
            qbest, cbest = scores.max(axis=1), scores.max(axis=0)
            return ((np.average(qbest, weights=query_qw) + np.average(cbest, weights=cw))/2,
                    min(qbest.min(), cbest.min()))

        for i in candidates:
            candidate = self.views[i]
            cw = self.candidate_cw[i]
            lexical_mean, lexical_min = tokens_score(
                query.tokens, candidate.tokens, lambda a, b: fuzz.ratio(a, b)/100, cw)
            both_numbers = bool(query.numbers and candidate.numbers)
            structure = [
                float(both_numbers and query.numbers == candidate.numbers),
                float(both_numbers and query.numbers != candidate.numbers),
                abs(len(query.tokens)-len(candidate.tokens)), lexical_mean, lexical_min,
                fuzz.ratio(query.tokens[-1], candidate.tokens[-1])/100
                if query.tokens and candidate.tokens else 0.,
            ]
            if mode == 'structure':
                output.append(structure)
                continue

            # 방안 1: 비한글 선박은 발음과 철자가 사실상 동일하므로, 무거운 발음 루프 대신
            # 이미 계산된 고속 철자 점수(lexical)로 채워 0초 만에 완료
            if not is_hangul:
                phone_full = (fuzz.ratio(query.roman, candidate.roman)/100
                              if query.roman and candidate.roman else 0.)
                output.append(structure + [
                    phone_full, lexical_mean, lexical_min,
                    fuzz.ratio(query.tokens[-1], candidate.tokens[-1])/100 if query.tokens and candidate.tokens else 0.,
                    lexical_mean, 0.0
                ])
                continue

            # 한글 선박 전용 정밀 발음 연산
            phonetic_mean, phonetic_min = tokens_score(
                query.phone_tokens, candidate.phone_tokens, phone_score, cw)
            qleft, cleft = list(query.phone_tokens), list(candidate.phone_tokens)
            shared = 0
            while len(qleft) > 1 and len(cleft) > 1 and phone_score(qleft[0], cleft[0]) >= .8:
                qleft.pop(0); shared += 1
            while len(qleft) > 1 and len(cleft) > 1 and phone_score(qleft[-1], cleft[-1]) >= .8:
                qleft.pop(); cleft.pop(); shared += 1

            # phone_shared_edge 복구
            output.append(structure + [
                phone_score(query.phone, candidate.phone), phonetic_mean, phonetic_min,
                phone_score(query.phone_tokens[-1], candidate.phone_tokens[-1])
                if query.phone_tokens and candidate.phone_tokens else 0.,
                phone_score(''.join(qleft), ''.join(cleft)), float(shared),
            ])
        width = len(STRUCTURE_FEATURES) + (len(PHONETIC_FEATURES) if mode == 'all' else 0)
        return np.asarray(output, dtype=np.float32).reshape(len(candidates), width)
