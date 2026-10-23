# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz.distance import OSA

# TEAM_RULES §6: train 사전 조회 스위치
USE_TRAIN_LOOKUP: bool = True

ROMAN_CHAR_MAP = {
    "Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5",
    "Ⅵ": "6", "Ⅶ": "7", "Ⅷ": "8", "Ⅸ": "9", "Ⅹ": "10",
    "ⅰ": "1", "ⅱ": "2", "ⅲ": "3", "ⅳ": "4", "ⅴ": "5",
}
ROMAN_TOKEN_MAP = {
    "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
    "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10",
}

# 한글 초/중/종성 기본 로마자 및 발음 코드
CHOSUNG = ["G", "KK", "N", "D", "TT", "R", "M", "B", "PP", "S", "SS", "", "J", "JJ", "CH", "K", "T", "P", "H"]
JUNGSUNG = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WO", "WE", "WI", "YU", "EU", "UI", "I"]
JONGSUNG = ["", "K", "K", "K", "N", "N", "N", "T", "L", "K", "M", "P", "L", "L", "P", "L", "M", "P", "P", "T", "T", "NG", "T", "T", "K", "T", "P", "T"]

# 한국어 외래어 발음 통합형 (영문 발음 정규화본과 1:1 비교용)
CHOSUNG_PHON = ["K", "K", "N", "T", "T", "R", "M", "P", "P", "S", "S", "", "J", "J", "C", "K", "T", "P", "H"]
JUNGSUNG_PHON = ["A", "E", "YA", "YE", "O", "E", "YO", "YE", "O", "WA", "WE", "WE", "YO", "U", "WO", "WE", "WI", "YU", "U", "I", "I"]
JONGSUNG_PHON = ["", "K", "K", "K", "N", "N", "N", "T", "R", "K", "M", "P", "R", "R", "P", "R", "M", "P", "P", "T", "T", "N", "T", "T", "K", "T", "P", "T"]

KR_LETTER_MAP = {
    "에이": "A", "비": "B", "씨": "C", "시": "C", "디": "D", "이": "E",
    "에프": "F", "쥐": "G", "지": "G", "에이치": "H", "아이": "I", "제이": "J",
    "케이": "K", "엘": "L", "엠": "M", "엔": "N", "오": "O", "피": "P",
    "큐": "Q", "알": "R", "에스": "S", "티": "T", "유": "U", "브이": "V",
    "더블유": "W", "엑스": "X", "와이": "Y", "제트": "Z",
}

KR_NUM_WORD_MAP = {
    "원": "1", "투": "2", "쓰리": "3", "스리": "3", "포": "4", "파이브": "5",
    "식스": "6", "세븐": "7", "에잇": "8", "나인": "9", "텐": "10",
    "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5", "육": "6", "칠": "7", "팔": "8", "구": "9", "십": "10",
}

re_hangul = re.compile(r"[가-힣]")
re_non_alnum_kr = re.compile(r"[^A-Z0-9가-힣]+")


def is_korean_name(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    return bool(re_hangul.search(name))


def syllable_to_rom(ch: str) -> str:
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return ch
    cho = code // 588
    jung = (code % 588) // 28
    jong = code % 28
    return CHOSUNG[cho] + JUNGSUNG[jung] + JONGSUNG[jong]


def syllable_to_phon(ch: str) -> str:
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return ch
    cho = code // 588
    jung = (code % 588) // 28
    jong = code % 28
    return CHOSUNG_PHON[cho] + JUNGSUNG_PHON[jung] + JONGSUNG_PHON[jong]


def en_to_phonetic(word: str) -> str:
    """영문 단어를 한국어 외래어 표기 발음에 가깝게 정규화"""
    if not word:
        return ""
    s = re.sub(r"[^A-Z0-9]", "", word.upper())
    if not s:
        return ""
    # 주요 이중자음/이중모음 치환
    for pat, rep in [
        ("TCH", "C"), ("SCH", "S"), ("SH", "S"), ("CH", "C"),
        ("PH", "P"), ("GH", ""), ("CK", "K"), ("QU", "KW"),
        ("WR", "R"), ("KN", "N"), ("WH", "W"), ("TH", "S"),
        ("NG", "N"),
        ("OO", "U"), ("EE", "I"), ("EA", "I"), ("OA", "O"),
        ("OU", "U"), ("OW", "O"), ("AI", "E"), ("AY", "E"),
        ("AU", "O"), ("AW", "O"), ("EW", "U"), ("IE", "I"),
    ]:
        s = s.replace(pat, rep)
    # 어말 무음 E 제거 (길이 4 이상이고 모음이 앞에 있을 때)
    if len(s) >= 4 and s.endswith("E") and s[-2] not in "AEIOU":
        s = s[:-1]
    # 자음/모음 통합 (G/C/Q->K, B/V/F->P, D->T, L->R, Z->S)
    trans = str.maketrans({
        "G": "K", "C": "K", "Q": "K",
        "B": "P", "V": "P", "F": "P",
        "D": "T", "L": "R", "Z": "S", "X": "KS",
    })
    s = s.translate(trans)
    if not s:
        return ""
    # 연속된 동일 글자 압축
    out = [s[0]]
    for ch in s[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def kr_to_phonetic(word: str) -> str:
    """한글 단어를 발음 통합 코드로 변환 (스/즈/츠 등의 으 모음 제거 포함)"""
    if not word:
        return ""
    raw_pieces = []
    for ch in word:
        if "가" <= ch <= "힣":
            code = ord(ch) - 0xAC00
            cho = code // 588
            jung = (code % 588) // 28
            jong = code % 28
            # 외래어 마찰음/파열음 뒤의 'ㅡ'(jung==18)이고 종성이 없으면 모음 생략 (예: 스타 -> STA, 프린스 -> PRINS)
            if jung == 18 and jong == 0 and CHOSUNG_PHON[cho] in ("S", "T", "P", "K", "J", "C", "R"):
                raw_pieces.append(CHOSUNG_PHON[cho])
            else:
                raw_pieces.append(CHOSUNG_PHON[cho] + JUNGSUNG_PHON[jung] + JONGSUNG_PHON[jong])
        else:
            raw_pieces.append(ch)
    s = "".join(raw_pieces)
    if not s:
        return ""
    out = [s[0]]
    for ch in s[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def consonant_skeleton(text: str) -> str:
    """자음 골격 추출 (첫 글자가 모음이면 유지하되 이후 모든 모음 제거)"""
    if not text:
        return ""
    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not text:
        return ""
    text = text.replace("PH", "P").replace("SH", "S").replace("CH", "C").replace("CK", "K").replace("NG", "N")
    trans = str.maketrans({
        "P": "P", "F": "P", "V": "P", "B": "P",
        "L": "R", "C": "K", "Q": "K", "G": "K",
        "Z": "S", "D": "T",
    })
    t = text.translate(trans)
    out = []
    for i, ch in enumerate(t):
        if ch in "AEIOUWY":
            if i == 0:
                out.append("V")  # 첫 글자 모음 표시
            continue
        if not out or ch != out[-1]:
            out.append(ch)
    return "".join(out)


def normalize_vessel_name(raw_name: Any) -> Tuple[str, List[str], List[str]]:
    if not isinstance(raw_name, str):
        return "", [], []
    s = raw_name.strip().upper()
    for k, v in ROMAN_CHAR_MAP.items():
        s = s.replace(k, f" {v} ")

    # NO.3, NO-3, NO3, #3 -> 공백 + 숫자
    s = re.sub(r"\bNO\.?\s*(\d+)\b", r" \1 ", s)
    s = re.sub(r"#\s*(\d+)\b", r" \1 ", s)
    # 제3호, 3호 -> 3
    s = re.sub(r"제\s*(\d+)\s*호", r" \1 ", s)
    s = re.sub(r"(\d+)\s*호\b", r" \1 ", s)

    # 문자와 숫자 분리, 영문과 한글 분리
    s = re.sub(r"([A-Z가-힣])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Z가-힣])", r"\1 \2", s)
    s = re.sub(r"([A-Z])([가-힣])", r"\1 \2", s)
    s = re.sub(r"([가-힣])([A-Z])", r"\1 \2", s)

    raw_tokens = [tok for tok in re_non_alnum_kr.split(s) if tok]
    word_tokens: List[str] = []
    num_tokens: List[str] = []

    for i, tok in enumerate(raw_tokens):
        if tok in ROMAN_TOKEN_MAP and (i == len(raw_tokens) - 1 or len(raw_tokens) > 1):
            num_tokens.append(str(int(ROMAN_TOKEN_MAP[tok])))
            continue
        if tok.isdigit():
            # 03 -> 3 정규화
            num_tokens.append(str(int(tok)))
            continue
        if tok == "NO":
            continue
        if tok == "호":
            continue
        # 마지막 토큰이 '호'로 끝나면 제거
        if i == len(raw_tokens) - 1 and len(tok) >= 2 and tok.endswith("호") and re_hangul.search(tok):
            tok = tok[:-1]
        elif len(tok) >= 3 and tok.endswith("호") and re_hangul.search(tok):
            tok = tok[:-1]

        # 한글 숫자 단어(원, 투, 쓰리...)가 끝에 오면 숫자 토큰으로도 보조 처리
        if tok in KR_NUM_WORD_MAP and i == len(raw_tokens) - 1 and len(raw_tokens) >= 2:
            num_tokens.append(KR_NUM_WORD_MAP[tok])
            continue

        if tok:
            word_tokens.append(tok)

    # 연속된 1글자 한글 알파벳 발음 결합 (예: 에이 치 -> 에이치)
    merged_words: List[str] = []
    idx = 0
    while idx < len(word_tokens):
        if idx + 1 < len(word_tokens) and (word_tokens[idx] + word_tokens[idx + 1]) in KR_LETTER_MAP:
            merged_words.append(KR_LETTER_MAP[word_tokens[idx] + word_tokens[idx + 1]])
            idx += 2
        elif word_tokens[idx] in KR_LETTER_MAP:
            merged_words.append(KR_LETTER_MAP[word_tokens[idx]])
            idx += 1
        else:
            merged_words.append(word_tokens[idx])
            idx += 1

    compact = "".join(merged_words) + ("".join(num_tokens) if num_tokens else "")
    return compact, merged_words, num_tokens


def dp_align_kr_en(kr_word: str, en_word: str) -> List[Tuple[str, str]]:
    """
    한글 단어(음절 N개)와 영문 단어(길이 M)를 동적 계획법(DP)으로 정렬하여
    각 한글 음절에 대응하는 영문 조각(길이 1~4)을 찾는다.
    """
    n = len(kr_word)
    m = len(en_word)
    if n == 0 or m == 0 or m < n:
        return []
    if n == 1:
        return [(kr_word, en_word)]

    syl_roms = [syllable_to_rom(ch) for ch in kr_word]
    INF = 1e9
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        s_rom = syl_roms[i - 1]
        for j in range(i, m - (n - i) + 1):
            for k in range(max(i - 1, j - 5), j):
                if dp[i - 1][k] >= INF:
                    continue
                chunk = en_word[k:j]
                # 기본 로마자와의 편집거리 기반 비용
                sim = OSA.normalized_similarity(s_rom, chunk)
                cost = (1.0 - sim) + 0.08 * abs(len(chunk) - len(s_rom))
                if dp[i - 1][k] + cost < dp[i][j]:
                    dp[i][j] = dp[i - 1][k] + cost
                    back[i][j] = k

    if dp[n][m] >= INF:
        return []

    pairs: List[Tuple[str, str]] = []
    curr_j = m
    for i in range(n, 0, -1):
        prev_k = back[i][curr_j]
        pairs.append((kr_word[i - 1], en_word[prev_k:curr_j]))
        curr_j = prev_k
    pairs.reverse()
    return pairs


def kr_word_to_en_candidates(
    word: str,
    syl_top2: Dict[str, List[str]],
    word_map: Dict[str, str],
) -> Tuple[str, str]:
    """한글 단어를 (1순위 영문 근사, 규칙 기반 로마자 근사) 2가지 버전으로 변환"""
    if not re_hangul.search(word):
        return word, word
    if word in word_map:
        mapped = word_map[word]
        return mapped, mapped
    if word in KR_LETTER_MAP:
        mapped = KR_LETTER_MAP[word]
        return mapped, mapped

    out1: List[str] = []
    out2: List[str] = []
    for ch in word:
        if "가" <= ch <= "힣":
            cands = syl_top2.get(ch, [])
            rom = syllable_to_rom(ch)
            out1.append(cands[0] if cands else rom)
            out2.append(cands[1] if len(cands) > 1 else rom)
        else:
            out1.append(ch)
            out2.append(ch)
    return "".join(out1), "".join(out2)


def year_bucket(year_val: Any) -> str:
    try:
        y = int(float(year_val))
    except (ValueError, TypeError):
        return "UNK"
    if y <= 2018:
        return "OLD"
    if y <= 2022:
        return "MID"
    return "NEW"


def fit_matching_model(train_df: pd.DataFrame, catalog_df: pd.DataFrame) -> Dict[str, Any]:
    merged = train_df.merge(
        catalog_df[["vessel_id", "vessel_name", "vessel_type", "flag_name", "gross_tonnage"]],
        left_on="target_vessel_id",
        right_on="vessel_id",
        suffixes=("_tr", "_cat"),
        how="inner",
    )

    # 1) 학습 사전 (call_sign, 정규화 이름)
    callsign_counts: Dict[str, Counter] = defaultdict(Counter)
    name_counts: Dict[str, Counter] = defaultdict(Counter)

    for row in train_df.itertuples(index=False):
        vid = str(row.target_vessel_id)
        cs = str(row.call_sign).strip().upper() if pd.notna(row.call_sign) else ""
        if cs and cs not in ("NAN", "NONE", ""):
            callsign_counts[cs][vid] += 1
        compact, _, _ = normalize_vessel_name(row.vessel_name)
        if compact:
            name_counts[compact][vid] += 1

    callsign_lookup = {k: v.most_common(1)[0][0] for k, v in callsign_counts.items()}
    name_lookup = {k: v.most_common(1)[0][0] for k, v in name_counts.items()}

    # 2) DP 정렬을 통한 한글 단어 / 음절 -> 영문 철자 대응 학습
    word_pair_counts: Dict[str, Counter] = defaultdict(Counter)
    syl_chunk_counts: Dict[str, Counter] = defaultdict(Counter)

    for row in merged.itertuples(index=False):
        tr_name = row.vessel_name_tr
        cat_name = row.vessel_name_cat
        if not is_korean_name(tr_name):
            continue
        _, kr_words, _ = normalize_vessel_name(tr_name)
        _, en_words, _ = normalize_vessel_name(cat_name)
        if not kr_words or not en_words:
            continue

        if len(kr_words) == len(en_words):
            pairs = list(zip(kr_words, en_words))
        else:
            pairs = [("".join(kr_words), "".join(en_words))]

        for kw, ew in pairs:
            if not re_hangul.search(kw) or not ew.isalpha():
                continue
            word_pair_counts[kw][ew] += 1
            for s_ch, e_chunk in dp_align_kr_en(kw, ew):
                if "가" <= s_ch <= "힣" and e_chunk:
                    syl_chunk_counts[s_ch][e_chunk] += 1

    learned_word_map = {
        k: v.most_common(1)[0][0]
        for k, v in word_pair_counts.items()
        if v.most_common(1)[0][1] >= 1
    }
    learned_syl_top2: Dict[str, List[str]] = {}
    for ch, cnt in syl_chunk_counts.items():
        top_items = [chunk for chunk, freq in cnt.most_common(2) if freq >= 1]
        if top_items:
            learned_syl_top2[ch] = top_items

    # 3) 선종 로그 우도비
    type_joint: Dict[str, Counter] = defaultdict(Counter)
    cat_type_prior: Counter = Counter(catalog_df["vessel_type"].fillna("UNK").astype(str))
    total_cat = max(1, len(catalog_df))

    for row in merged.itertuples(index=False):
        kt = str(row.vessel_type_tr).strip() if pd.notna(row.vessel_type_tr) else "UNK"
        ct = str(row.vessel_type_cat).strip() if pd.notna(row.vessel_type_cat) else "UNK"
        type_joint[kt][ct] += 1

    type_log_lift: Dict[str, Dict[str, float]] = {}
    for kt, cnts in type_joint.items():
        tot_kt = sum(cnts.values())
        type_log_lift[kt] = {}
        for ct, c in cnts.items():
            p_cond = (c + 0.5) / (tot_kt + 5.0)
            p_prior = max(1e-5, cat_type_prior.get(ct, 1) / total_cat)
            type_log_lift[kt][ct] = round(float(np.clip(math.log(p_cond / p_prior), -2.5, 4.0)), 4)

    # 4) 국적 + 도착연도 구간 로그 우도비
    flag_joint: Dict[str, Counter] = defaultdict(Counter)
    cat_flag_prior: Counter = Counter(catalog_df["flag_name"].fillna("UNK").astype(str))

    for row in merged.itertuples(index=False):
        kf = str(row.flag_name_tr).strip() if pd.notna(row.flag_name_tr) else "UNK"
        cf = str(row.flag_name_cat).strip() if pd.notna(row.flag_name_cat) else "UNK"
        yb = year_bucket(row.arrival_year)
        if kf != "UNK":
            flag_joint[f"{kf}|{yb}"][cf] += 1
            flag_joint[f"{kf}|ALL"][cf] += 1

    flag_log_lift: Dict[str, Dict[str, float]] = {}
    for key, cnts in flag_joint.items():
        tot_k = sum(cnts.values())
        flag_log_lift[key] = {}
        for cf, c in cnts.items():
            p_cond = (c + 0.5) / (tot_k + 5.0)
            p_prior = max(1e-5, cat_flag_prior.get(cf, 1) / total_cat)
            flag_log_lift[key][cf] = round(float(np.clip(math.log(p_cond / p_prior), -2.5, 4.5)), 4)

    return {
        "version": "2.0",
        "use_train_lookup_default": True,
        "callsign_lookup": callsign_lookup,
        "name_lookup": name_lookup,
        "learned_word_map": learned_word_map,
        "learned_syl_top2": learned_syl_top2,
        "type_log_lift": type_log_lift,
        "flag_log_lift": flag_log_lift,
    }


class CatalogIndex:
    def __init__(self, catalog_df: pd.DataFrame) -> None:
        self.vessel_ids: List[str] = catalog_df["vessel_id"].astype(str).tolist()
        self.n = len(self.vessel_ids)

        raw_names = catalog_df["vessel_name"].fillna("").astype(str).tolist()
        self.types = catalog_df["vessel_type"].fillna("UNK").astype(str).str.strip().tolist()
        self.flags = catalog_df["flag_name"].fillna("UNK").astype(str).str.strip().tolist()
        self.gt = pd.to_numeric(catalog_df["gross_tonnage"], errors="coerce").fillna(-1.0).to_numpy(dtype=np.float64)

        self.compact_names: List[str] = [""] * self.n
        self.word_Alpha: List[str] = [""] * self.n
        self.word_tokens: List[List[str]] = [[] for _ in range(self.n)]
        self.phon_words: List[List[str]] = [[] for _ in range(self.n)]
        self.phon_full: List[str] = [""] * self.n
        self.num_strs: List[str] = [""] * self.n
        self.skeletons: List[str] = [""] * self.n

        # 역색인
        self.exact_compact_idx: Dict[str, List[int]] = defaultdict(list)
        self.exact_alpha_idx: Dict[str, List[int]] = defaultdict(list)
        self.prefix3_idx: Dict[str, List[int]] = defaultdict(list)
        self.prefix2_idx: Dict[str, List[int]] = defaultdict(list)
        self.skel3_idx: Dict[str, List[int]] = defaultdict(list)
        self.skel2_idx: Dict[str, List[int]] = defaultdict(list)
        self.phon3_idx: Dict[str, List[int]] = defaultdict(list)
        self.num_idx: Dict[str, List[int]] = defaultdict(list)
        self.word_idx: Dict[str, List[int]] = defaultdict(list)
        # TEAM_RULES §2: 6글자 이상 단어의 앞2+뒤2 오타 불변 인덱스
        self.outer2_idx: Dict[str, List[int]] = defaultdict(list)

        for i, rname in enumerate(raw_names):
            comp, wtoks, ntoks = normalize_vessel_name(rname)
            alpha = "".join(wtoks)
            nstr = "|".join(ntoks)
            skel = consonant_skeleton(alpha)
            p_words = [en_to_phonetic(w) for w in wtoks]
            p_full = "".join(p_words)

            self.compact_names[i] = comp
            self.word_Alpha[i] = alpha
            self.word_tokens[i] = wtoks
            self.phon_words[i] = p_words
            self.phon_full[i] = p_full
            self.num_strs[i] = nstr
            self.skeletons[i] = skel

            if comp:
                self.exact_compact_idx[comp].append(i)
            if alpha:
                self.exact_alpha_idx[alpha].append(i)
                self.prefix3_idx[alpha[:3]].append(i)
                self.prefix2_idx[alpha[:2]].append(i)
            if skel:
                self.skel3_idx[skel[:3]].append(i)
                self.skel2_idx[skel[:2]].append(i)
            if p_full:
                self.phon3_idx[p_full[:3]].append(i)
            if nstr:
                self.num_idx[nstr].append(i)
            for w in wtoks:
                if len(w) >= 3:
                    self.word_idx[w[:4]].append(i)
                if len(w) >= 5:
                    self.outer2_idx[f"{w[:2]}_{w[-2:]}"].append(i)

        valid_mask = self.gt > 0
        self.sorted_gt_indices = np.argsort(self.gt[valid_mask])
        valid_positions = np.flatnonzero(valid_mask)
        self.sorted_gt_cat_idx = valid_positions[self.sorted_gt_indices]
        self.sorted_gt_vals = self.gt[self.sorted_gt_cat_idx]

    def query_tonnage_window(self, ton: float, rel_tol: float = 0.06, max_cands: Optional[int] = None) -> List[int]:
        if ton <= 0 or len(self.sorted_gt_vals) == 0:
            return []
        low = ton * (1.0 - rel_tol) - 2.0
        high = ton * (1.0 + rel_tol) + 2.0
        left = int(np.searchsorted(self.sorted_gt_vals, low, side="left"))
        right = int(np.searchsorted(self.sorted_gt_vals, high, side="right"))
        if right <= left:
            return []
        if max_cands is None or (right - left) <= max_cands:
            return self.sorted_gt_cat_idx[left:right].tolist()
        center = int(np.searchsorted(self.sorted_gt_vals, ton))
        half = max_cands // 2
        s = max(left, center - half)
        e = min(right, s + max_cands)
        s = max(left, e - max_cands)
        return self.sorted_gt_cat_idx[s:e].tolist()


def predict_rows(
    df: pd.DataFrame,
    model: Dict[str, Any],
    cat_idx: CatalogIndex,
    use_train_lookup: bool = USE_TRAIN_LOOKUP,
    gold_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[int]]:
    callsign_lookup = model.get("callsign_lookup", {})
    name_lookup = model.get("name_lookup", {})
    word_map = model.get("learned_word_map", {})
    syl_top2 = model.get("learned_syl_top2", {})
    type_lift = model.get("type_log_lift", {})
    flag_lift = model.get("flag_log_lift", {})

    preds: List[str] = []
    gold_ranks: List[int] = []

    names = df["vessel_name"].tolist()
    callsigns = df["call_sign"].tolist() if "call_sign" in df.columns else [None] * len(df)
    gts = pd.to_numeric(df["gross_tonnage"], errors="coerce").fillna(-1.0).tolist()
    igts = pd.to_numeric(df["international_gross_tonnage"], errors="coerce").fillna(-1.0).tolist()
    flags = df["flag_name"].tolist() if "flag_name" in df.columns else [None] * len(df)
    vtypes = df["vessel_type"].tolist() if "vessel_type" in df.columns else [None] * len(df)
    years = df["arrival_year"].tolist() if "arrival_year" in df.columns else [None] * len(df)

    for row_i in range(len(df)):
        raw_name = names[row_i]
        cs = str(callsigns[row_i]).strip().upper() if pd.notna(callsigns[row_i]) else ""
        gt = gts[row_i]
        igt = igts[row_i]
        # TEAM_RULES §2: 국제총톤수가 카탈로그와 97% 더 가깝다
        ton = igt if igt > 0 else (gt if gt > 0 else -1.0)
        alt_ton = gt if (gt > 0 and gt != ton) else -1.0

        kf = str(flags[row_i]).strip() if pd.notna(flags[row_i]) else "UNK"
        kt = str(vtypes[row_i]).strip() if pd.notna(vtypes[row_i]) else "UNK"
        yb = year_bucket(years[row_i])
        gold_vid = gold_ids[row_i] if gold_ids is not None else None

        # 0) 사전 조회 (스위치 켜진 경우)
        if use_train_lookup:
            if cs and cs in callsign_lookup:
                pred_vid = callsign_lookup[cs]
                preds.append(pred_vid)
                gold_ranks.append(1 if (gold_vid is None or pred_vid == gold_vid) else 0)
                continue

        comp, wtoks, ntoks = normalize_vessel_name(raw_name)
        if use_train_lookup and comp and comp in name_lookup:
            pred_vid = name_lookup[comp]
            preds.append(pred_vid)
            gold_ranks.append(1 if (gold_vid is None or pred_vid == gold_vid) else 0)
            continue

        kr_flag = is_korean_name(raw_name)
        nstr = "|".join(ntoks)

        if kr_flag:
            cands_pairs = [kr_word_to_en_candidates(w, syl_top2, word_map) for w in wtoks]
            en_wtoks1 = [p[0] for p in cands_pairs]
            en_wtoks2 = [p[1] for p in cands_pairs]
            query_alpha1 = "".join(en_wtoks1)
            query_alpha2 = "".join(en_wtoks2)
            # 한글 단어가 이미 영문으로 매핑된 경우 en_to_phonetic, 한글이면 kr_to_phonetic
            query_phon_words = [
                en_to_phonetic(w_en) if w in word_map else kr_to_phonetic(w)
                for w, w_en in zip(wtoks, en_wtoks1)
            ]
            query_phon_full = "".join(query_phon_words)
        else:
            en_wtoks1 = wtoks
            en_wtoks2 = wtoks
            query_alpha1 = "".join(wtoks)
            query_alpha2 = query_alpha1
            query_phon_words = [en_to_phonetic(w) for w in wtoks]
            query_phon_full = "".join(query_phon_words)

        query_skel1 = consonant_skeleton(query_alpha1)
        query_skel2 = consonant_skeleton(query_alpha2)

        # 1) 후보 검색 (Candidate Retrieval)
        cand_set = set()
        if not kr_flag and comp in cat_idx.exact_compact_idx:
            cand_set.update(cat_idx.exact_compact_idx[comp])
        if query_alpha1 in cat_idx.exact_alpha_idx:
            cand_set.update(cat_idx.exact_alpha_idx[query_alpha1])

        if kr_flag:
            # TEAM_RULES §2: 한글 행은 톤수가 100% 존재하고 최대 9.3% 차이 안 -> ±9.8% 무제한 검색으로 recall 100% 확보
            if ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.098, max_cands=None))
            if alt_ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(alt_ton, rel_tol=0.098, max_cands=None))
            if query_alpha1:
                cand_set.update(cat_idx.prefix3_idx.get(query_alpha1[:3], []))
            if query_skel1:
                cand_set.update(cat_idx.skel3_idx.get(query_skel1[:3], []))
            if query_phon_full:
                cand_set.update(cat_idx.phon3_idx.get(query_phon_full[:3], []))
            for w in en_wtoks1:
                if len(w) >= 3:
                    cand_set.update(cat_idx.word_idx.get(w[:4], []))
        else:
            # 비한글: 앞2+뒤2 오타 불변 인덱스 + prefix + 자음골격 + 톤수 창
            for w in en_wtoks1:
                if len(w) >= 5:
                    cand_set.update(cat_idx.outer2_idx.get(f"{w[:2]}_{w[-2:]}", []))
                if len(w) >= 3:
                    cand_set.update(cat_idx.word_idx.get(w[:4], []))
            if query_alpha1:
                cand_set.update(cat_idx.prefix3_idx.get(query_alpha1[:3], []))
                if len(cand_set) < 150:
                    cand_set.update(cat_idx.prefix2_idx.get(query_alpha1[:2], []))
            if query_skel1:
                cand_set.update(cat_idx.skel3_idx.get(query_skel1[:3], []))
            if ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.06, max_cands=300))
                if len(cand_set) < 100:
                    cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.10, max_cands=300))

        if nstr and nstr in cat_idx.num_idx:
            num_cands = cat_idx.num_idx[nstr]
            if len(num_cands) <= 500:
                cand_set.update(num_cands)

        if not cand_set:
            if ton > 0:
                cand_set.update(cat_idx.query_tonnage_window(ton, rel_tol=0.30, max_cands=200))
            if not cand_set:
                cand_set.add(0)

        # 2) 후보별 점수 계산 (Scoring)
        t_lifts = type_lift.get(kt, {})
        f_lifts = flag_lift.get(f"{kf}|{yb}", flag_lift.get(f"{kf}|ALL", {}))
        n_q_words = len(en_wtoks1)

        scored_cands: List[Tuple[float, int]] = []
        for c_i in cand_set:
            c_alpha = cat_idx.word_Alpha[c_i]
            c_skel = cat_idx.skeletons[c_i]
            c_nstr = cat_idx.num_strs[c_i]
            c_gt = cat_idx.gt[c_i]
            c_wtoks = cat_idx.word_tokens[c_i]
            c_pwords = cat_idx.phon_words[c_i]
            c_pfull = cat_idx.phon_full[c_i]

            # (a) 이름 유사도
            if kr_flag:
                sim_a1 = OSA.normalized_similarity(query_alpha1, c_alpha) if (query_alpha1 and c_alpha) else 0.0
                sim_a2 = OSA.normalized_similarity(query_alpha2, c_alpha) if (query_alpha2 and c_alpha) else 0.0
                sim_alpha = max(sim_a1, 0.97 * sim_a2)

                sim_s1 = OSA.normalized_similarity(query_skel1, c_skel) if (query_skel1 and c_skel) else 0.0
                sim_s2 = OSA.normalized_similarity(query_skel2, c_skel) if (query_skel2 and c_skel) else 0.0
                sim_skel = max(sim_s1, sim_s2)

                sim_phon = OSA.normalized_similarity(query_phon_full, c_pfull) if (query_phon_full and c_pfull) else 0.0

                # 단어 수가 같을 때 단어 1:1 정렬 유사도 (TEAM_RULES §2: 단어 순서 바뀐 선박 0/254)
                if n_q_words > 0 and n_q_words == len(c_wtoks):
                    w_sims = []
                    for qw1, qw2, qpw, cw, cpw in zip(en_wtoks1, en_wtoks2, query_phon_words, c_wtoks, c_pwords):
                        s_w = max(
                            OSA.normalized_similarity(qw1, cw),
                            0.97 * OSA.normalized_similarity(qw2, cw),
                            0.95 * OSA.normalized_similarity(qpw, cpw),
                        )
                        w_sims.append(s_w)
                    sim_tok = sum(w_sims) / n_q_words
                    word_cnt_bonus = 0.45
                else:
                    sim_tok = max(sim_alpha, sim_phon)
                    word_cnt_bonus = -0.35 * min(2, abs(n_q_words - len(c_wtoks)))

                # 첫 글자 발음 일치 보너스
                first_match = 0.35 if (query_phon_full and c_pfull and query_phon_full[0] == c_pfull[0]) else 0.0

                name_score = (
                    3.8 * sim_alpha
                    + 2.8 * sim_phon
                    + 2.2 * sim_skel
                    + 2.7 * sim_tok
                    + word_cnt_bonus
                    + first_match
                )
            else:
                if query_alpha1 and c_alpha:
                    sim_alpha = 1.0 if query_alpha1 == c_alpha else OSA.normalized_similarity(query_alpha1, c_alpha)
                else:
                    sim_alpha = 0.0
                sim_skel = OSA.normalized_similarity(query_skel1, c_skel) if (query_skel1 and c_skel) else 0.0

                if n_q_words > 1 and n_q_words == len(c_wtoks):
                    sim_tok = sum(OSA.normalized_similarity(a, b) for a, b in zip(en_wtoks1, c_wtoks)) / n_q_words
                else:
                    sim_tok = sim_alpha

                name_score = 8.0 * sim_alpha + 1.0 * sim_skel + 2.0 * sim_tok
                if comp and comp == cat_idx.compact_names[c_i]:
                    name_score += 2.5

            # (b) 숫자 토큰 점수 (TEAM_RULES §2: 숫자는 영문에 100% 남음)
            num_score = 0.0
            if nstr and c_nstr:
                if nstr == c_nstr:
                    num_score = 3.2
                else:
                    num_score = -4.0
            elif nstr and not c_nstr:
                num_score = -2.8
            elif not nstr and c_nstr:
                num_score = -1.6

            # (c) 톤수 점수 (TEAM_RULES §2: 정확 일치 54%, 3~5% 차이 37%, 최대 9.3%)
            ton_score = 0.0
            if ton > 0 and c_gt > 0:
                diff = abs(ton - c_gt)
                if alt_ton > 0:
                    diff = min(diff, abs(alt_ton - c_gt))
                rel = diff / max(1.0, c_gt)
                if diff <= 1.0:
                    ton_score = 3.6
                elif rel <= 0.005:
                    ton_score = 3.1
                elif rel <= 0.02:
                    ton_score = 2.5
                elif rel <= 0.052:
                    ton_score = 2.1 - 4.0 * (rel - 0.02)
                elif rel <= 0.095:
                    ton_score = 0.8 - 10.0 * (rel - 0.052)
                else:
                    ton_score = -3.5 * min(2.5, rel * 6.0)

            # (d) 선종 및 도착연도 적응형 국적 점수 (2015~2018년 국적 변경 46% 반영)
            ct = cat_idx.types[c_i]
            cf = cat_idx.flags[c_i]
            t_val = t_lifts.get(ct, -0.35)
            if kf == "UNK":
                f_val = 0.0
            else:
                default_f_pen = 0.0 if yb == "OLD" else -0.25
                f_val = f_lifts.get(cf, default_f_pen)
                if yb == "OLD" and f_val < 0:
                    f_val *= 0.25  # 과거 연도 국적 불일치 감점 완화

            meta_score = 0.40 * t_val + 0.32 * f_val

            total_score = name_score + num_score + ton_score + meta_score
            scored_cands.append((total_score, c_i))

        scored_cands.sort(key=lambda x: (-x[0], cat_idx.vessel_ids[x[1]]))
        best_vid = cat_idx.vessel_ids[scored_cands[0][1]]
        preds.append(best_vid)

        if gold_vid is not None:
            grank = 0
            for rank_idx, (_, c_i) in enumerate(scored_cands, start=1):
                if cat_idx.vessel_ids[c_i] == gold_vid:
                    grank = rank_idx
                    break
            gold_ranks.append(grank)
        else:
            gold_ranks.append(0)

    return preds, gold_ranks


def resolve_input_dir(base_dir: Path) -> Path:
    for cand in [base_dir / "data", base_dir / "open", base_dir.parent / "data", base_dir.parent / "open"]:
        if (cand / "test.csv").exists() and (cand / "vessel_catalog.csv").exists():
            return cand
    raise FileNotFoundError("test.csv 및 vessel_catalog.csv를 data/ 또는 open/에서 찾을 수 없습니다.")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = resolve_input_dir(base_dir)
    model_path = base_dir / "model" / "model.json"
    out_dir = base_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    test_df = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
    catalog_df = pd.read_csv(data_dir / "vessel_catalog.csv", encoding="utf-8-sig")

    cat_idx = CatalogIndex(catalog_df)
    use_lookup = bool(model.get("use_train_lookup_default", USE_TRAIN_LOOKUP)) and USE_TRAIN_LOOKUP

    preds, _ = predict_rows(test_df, model, cat_idx, use_train_lookup=use_lookup, gold_ids=None)

    sub = pd.DataFrame({
        "sample_id": test_df["sample_id"].astype(str),
        "target_vessel_id": preds,
    })
    sub.to_csv(out_dir / "submission.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
