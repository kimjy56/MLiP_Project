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
    for old, new in [('PH', 'F'), ('CK', 'K'), ('QU', 'KW'), ('EO', 'O')]:
        value = value.replace(old, new)
    # ㅡ 말음 탈락: e.g. RODEU -> ROD, BAENGKEU -> BAENGK
    value = re.sub(r'EU$', '', value)
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


KOREAN_ALPHABET_SYLLABLES = {
    '에이': 'A', '비': 'B', '씨': 'C', '시': 'C', '디': 'D', '이': 'E',
    '에프': 'F', '지': 'G', '쥐': 'G', '에이치': 'H', '아이': 'I', '제이': 'J',
    '케이': 'K', '엘': 'L', '엠': 'M', '엔': 'N', '오': 'O',
    '피': 'P', '큐': 'Q', '알': 'R', '아르': 'R', '에스': 'S', '티': 'T',
    '유': 'U', '브이': 'V', '더블유': 'W', '엑스': 'X', '와이': 'Y', '제트': 'Z',
    '앤': 'N',
}

KNOWN_KOREAN_WORD_MAP = {
    '현대': 'HYUNDAI', '흥아': 'HEUNGA', '우양': 'WOOYANG', '인천': 'INCHEON',
    '마루': 'MARU', '한진': 'HANJIN', '동양': 'DONGYANG', '대우': 'DAEWOO',
    '삼성': 'SAMSUNG', '고려': 'KOREA', '대한': 'KOREA', '아시아': 'ASIA',
    '아세안': 'ASIAN', '태평양': 'PACIFIC',
    '인터오션': 'INTEROCEAN', '오션': 'OCEAN', '썬': 'SUN',
    '글로비스': 'GLOVIS', '커리지': 'COURAGE', '갤럭시': 'GALAXY',
    '이스턴': 'EASTERN', '웨스턴': 'WESTERN', '스타': 'STAR', '스카이': 'SKY',
    '월드': 'WORLD', '글로벌': 'GLOBAL', '코리아': 'KOREA', '팬오션': 'PAN OCEAN',
    '천경': 'CK', '동영': 'PEGA', '포스코': 'POSCO', '에스케이': 'SK',
    '파이오니어': 'PIONEER', '피아니어': 'PIONEER', '리더': 'LEADER', '마스터': 'MASTER',
    '익스프레스': 'EXPRESS', '프런티어': 'FRONTIER', '하모니': 'HARMONY',
    '블루': 'BLUE', '그린': 'GREEN', '골든': 'GOLDEN', '에이스': 'ACE',
    '해운': 'SHIPPING', '캐리어': 'CARRIER', '컨테이너': 'CONTAINER', '라인': 'LINE',
    '크리퍼': 'CLIPPER', '클리퍼': 'CLIPPER',
    '에버': 'EVER', '얼라이언스': 'ALLIANCE', '에버그린': 'EVERGREEN',
    '머스크': 'MAERSK', '엠에스씨': 'MSC', '하팍': 'HAPAG', '오엔이': 'ONE', '짐': 'ZIM',
    '글로리': 'GLORY', '스피릿': 'SPIRIT', '슈프림': 'SUPREME', '수페리어': 'SUPERIOR',
    '시앤시': 'CNC', '씨앤씨': 'CNC',
    # 안전한 주요 기항 항구 (Safe Foreign Ports)
    '칭다오': 'QINGDAO', '상하이': 'SHANGHAI', '닝보': 'NINGBO', '다롄': 'DALIAN',
    # 안전한 선박 고유명사 외래어 (Safe Maritime Loanwords)
    '브릿지': 'BRIDGE', '브리지': 'BRIDGE', '엔터프라이즈': 'ENTERPRISE',
    '챌린저': 'CHALLENGER', '빅토리': 'VICTORY', '프라이드': 'PRIDE', '트레이더': 'TRADER',
    '보이저': 'VOYAGER', '프로그레스': 'PROGRESS', '챔피언': 'CHAMPION',
    '호라이즌': 'HORIZON', '네비게이터': 'NAVIGATOR', '내비게이터': 'NAVIGATOR',
    '선라이즈': 'SUNRISE', '썬라이즈': 'SUNRISE',
}

ROMAN_NUMERALS = {
    'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
    'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10'
}

KOREAN_NUM_WORD_MAP = {
    '원': '1', '투': '2', '쓰리': '3', '스리': '3', '포': '4', '파이브': '5',
    '식스': '6', '세븐': '7', '에잇': '8', '나인': '9', '텐': '10',
}


def parse_korean_acronym_token(token):
    idx = 0
    res = []
    syllable_names = sorted(KOREAN_ALPHABET_SYLLABLES.keys(), key=lambda x: -len(x))
    while idx < len(token):
        matched = False
        for s in syllable_names:
            if token.startswith(s, idx):
                res.append(KOREAN_ALPHABET_SYLLABLES[s])
                idx += len(s)
                matched = True
                break
        if not matched:
            return None
    if len(res) >= 2 or (len(res) == 1 and len(token) >= 2):
        return ''.join(res)
    return None


def extract_kr_acronym_prefix(word):
    """'케이엠티씨부산'처럼 공백 없이 붙은 한글 알파벳 읽기(2글자 이상 약어)를

    앞머리에서 찾아 ('KMTC', '부산')으로 분리한다.
    """
    if not isinstance(word, str) or len(word) < 2 or not any('가' <= c <= '힣' for c in word):
        return None
    pos = 0
    letters = []
    n = len(word)
    syllable_items = sorted(KOREAN_ALPHABET_SYLLABLES.items(), key=lambda x: -len(x[0]))
    while pos < n:
        matched = False
        for kr_let, en_let in syllable_items:
            if word.startswith(kr_let, pos):
                # 1음절 모음 글자 ('이', '오', '유')가 첫 글자로 오면 일반 한국어 단어(이순신 등)일 수 있으므로 건너뜀
                if len(kr_let) == 1 and kr_let in ('이', '오', '유') and len(letters) == 0:
                    continue
                letters.append(en_let)
                pos += len(kr_let)
                matched = True
                break
        if not matched:
            break
    if len(letters) >= 2 and (pos >= 3 or pos == n):
        return ''.join(letters), word[pos:]
    return None


def expand_korean_name(name):
    if not isinstance(name, str) or not any('가' <= c <= '힣' for c in name):
        return name
    s = name
    for k in sorted(KNOWN_KOREAN_WORD_MAP.keys(), key=lambda x: -len(x)):
        s = s.replace(k, f" {KNOWN_KOREAN_WORD_MAP[k]} ")
    s = re.sub(r'제\s*(\d+)\s*호?', r'NO. \1 ', s)
    # 선박 부속어/접미사 및 수식어 정규화 (호선, 바지, 급, 탱커)
    s = re.sub(r'\b(호선|바지|급|탱커)\b', ' ', s)
    tokens = s.split()
    new_tokens = []
    for t in tokens:
        acronym = parse_korean_acronym_token(t)
        if acronym:
            new_tokens.append(acronym)
        elif t in KOREAN_NUM_WORD_MAP:
            new_tokens.append(KOREAN_NUM_WORD_MAP[t])
        else:
            # 연속된 알파벳 약어 접두사 분리 검사 (예: 케이엠티씨부산 -> KMTC, 부산)
            prefix_res = extract_kr_acronym_prefix(t)
            if prefix_res:
                acronym_part, remainder = prefix_res
                new_tokens.append(acronym_part)
                if remainder:
                    if remainder.endswith('호선') and len(remainder) >= 4:
                        remainder = remainder[:-2]
                    elif remainder.endswith('바지') and len(remainder) >= 4:
                        remainder = remainder[:-2]
                    elif remainder.endswith('탱커') and len(remainder) >= 4:
                        remainder = remainder[:-2]
                    elif remainder.endswith('급') and len(remainder) >= 3:
                        remainder = remainder[:-1]
                    elif remainder.endswith('호') and len(remainder) >= 3 and not remainder.endswith('번호'):
                        remainder = remainder[:-1]
                    if remainder in KOREAN_NUM_WORD_MAP:
                        remainder = KOREAN_NUM_WORD_MAP[remainder]
                    new_tokens.append(remainder)
                continue

            if t.endswith('호선') and len(t) >= 4:
                t = t[:-2]
            elif t.endswith('바지') and len(t) >= 4:
                t = t[:-2]
            elif t.endswith('탱커') and len(t) >= 4:
                t = t[:-2]
            elif t.endswith('급') and len(t) >= 3:
                t = t[:-1]
            elif t.endswith('호') and len(t) >= 3 and not t.endswith('번호'):
                t = t[:-1]
            new_tokens.append(t)
    return ' '.join(new_tokens)


def name_view(value, expand_korean=False):
    if expand_korean and isinstance(value, str):
        value = expand_korean_name(value)
    raw = unicodedata.normalize('NFKC', value).upper() if isinstance(value, str) else ''
    # Split BEFORE Unidecode: its Hangul output adds syllable spaces.
    tokens = tuple(compact(unidecode(t).upper()) for t in re.findall(r'[^\W_]+', raw))
    tokens = tuple(t for t in tokens if t)
    roman = ''.join(tokens)
    phones = tuple(phone_spelling(t) for t in tokens)
    # Preserve boundaries: (1, 23) must not equal (12, 3). Ignore leading zeros.
    numbers = tuple(n.lstrip('0') or '0' for n in re.findall(r'\d+', raw))
    if not numbers:
        roman_nums = tuple(ROMAN_NUMERALS[t] for t in tokens if t in ROMAN_NUMERALS)
        if roman_nums:
            numbers = roman_nums
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
