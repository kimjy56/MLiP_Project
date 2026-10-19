"""Deterministic name views; no external lexicon or test-fitted statistics.

These are spelling/phonetic heuristics, not a pronunciation (G2P) model.
Raw names stay in the catalog; lossy views are only additional features.
"""
from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz
from unidecode import unidecode

STRUCTURE_FEATURES = [
    'number_sequence_equal', 'number_conflict', 'catalog_number_missing',
    'number_count_gap', 'token_count_gap', 'token_roman_idf',
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


def weighted_similarity(a, b):
    """Weighted Levenshtein on phone_spellings, normalized to [0, 1].

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
                                  frozenset((ca, cb)) in SIMILAR_PAIRS) else 4)
            current.append(min(previous[j+1] + deletion,
                               current[j] + insertion[j], previous[j] + substitution))
        previous = current
    return max(0.0, 1.0 - previous[-1] / (4 * max(len(a), len(b))))


class NameFeatureIndex:
    def __init__(self, names):
        self.views = [name_view(n) for n in names]
        frequency = Counter(t for v in self.views for t in set(v.tokens))
        self.idf = {t: 1 + math.log((len(self.views)+1)/(count+1))
                    for t, count in frequency.items()}
        self.unknown_idf = 1 + math.log(len(self.views)+1)
        self.search_keys = [v.search_key for v in self.views]

    def features(self, name, candidates, mode):
        query = name_view(name)
        output = []
        # Local to this input row; no cross-test-row fitted state/cache.
        distances = {}

        def phone_score(a, b):
            key = (a, b)
            if key not in distances:
                distances[key] = weighted_similarity(a, b)
            return distances[key]

        def tokens_score(q, c, scorer):
            if not q or not c:
                return 0., 0.
            scores = np.array([[scorer(a, b) for b in c] for a in q])
            # Both directions penalize extra unmatched words. Frequency comes
            # exclusively from the fixed official catalog, never from test.
            qw = [self.idf.get(t, self.unknown_idf) for t in query.tokens]
            cw = [self.idf.get(t, self.unknown_idf) for t in candidate.tokens]
            qbest, cbest = scores.max(axis=1), scores.max(axis=0)
            return ((np.average(qbest, weights=qw) + np.average(cbest, weights=cw))/2,
                    min(qbest.min(), cbest.min()))

        for i in candidates:
            candidate = self.views[i]
            lexical_mean, lexical_min = tokens_score(
                query.tokens, candidate.tokens, lambda a, b: fuzz.ratio(a, b)/100)
            both_numbers = bool(query.numbers and candidate.numbers)
            structure = [
                float(both_numbers and query.numbers == candidate.numbers),
                float(both_numbers and query.numbers != candidate.numbers),
                float(not candidate.numbers),
                abs(len(query.numbers)-len(candidate.numbers)),
                abs(len(query.tokens)-len(candidate.tokens)), lexical_mean, lexical_min,
                fuzz.ratio(query.tokens[-1], candidate.tokens[-1])/100
                if query.tokens and candidate.tokens else 0.,
            ]
            if mode == 'structure':
                output.append(structure)
                continue
            phonetic_mean, phonetic_min = tokens_score(
                query.phone_tokens, candidate.phone_tokens, phone_score)
            qleft, cleft = list(query.phone_tokens), list(candidate.phone_tokens)
            shared = 0
            # Drop matching edges only in this extra comparison, never in the
            # original name, and always leave at least one word on each side.
            while len(qleft) > 1 and len(cleft) > 1 and phone_score(qleft[0], cleft[0]) >= .8:
                qleft.pop(0); cleft.pop(0); shared += 1
            while len(qleft) > 1 and len(cleft) > 1 and phone_score(qleft[-1], cleft[-1]) >= .8:
                qleft.pop(); cleft.pop(); shared += 1
            output.append(structure + [
                phone_score(query.phone, candidate.phone), phonetic_mean, phonetic_min,
                phone_score(query.phone_tokens[-1], candidate.phone_tokens[-1])
                if query.phone_tokens and candidate.phone_tokens else 0.,
                phone_score(''.join(qleft), ''.join(cleft)), float(shared),
            ])
        width = len(STRUCTURE_FEATURES) + (len(PHONETIC_FEATURES) if mode == 'all' else 0)
        return np.asarray(output, dtype=np.float32).reshape(len(candidates), width)
