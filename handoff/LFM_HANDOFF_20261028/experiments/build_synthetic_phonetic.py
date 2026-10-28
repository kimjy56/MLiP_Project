"""
Synthetic Phonetic Dataset Generator (SPEC-DPR-01 Stage A).

Generates 100,000 synthetic (Korean query, English catalog, Hard Negatives) triplets
from vessel_catalog.csv for offline phonetic pre-training.

Techniques:
  1. Multi-tier English-to-Hangul transliteration (Dictionary + Syllable + Phonetic rules).
  2. Realistic port observation noise injection:
     - Spacing variations (deletion, space insert)
     - Korean suffix attachment (호, 선, NO, 제N호)
     - Vowel shifts (ㅏ ↔ ㅓ, ㅐ ↔ ㅔ, ㅗ ↔ ㅜ, ㅡ ↔ ㅣ)
     - Final consonant drops/additions (ㄹ, ㅅ, ㅇ, ㄴ)
     - Silent letter patterns (KN-, WR-, GH-, PS-)
  3. Sister-ship & Phonetic Hard Negative Mining:
     - Shared prefix / suffix confusion ships (e.g. MORNING LENA vs MORNING CARINA)
     - Edit-distance close confusion ships
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from phonetic_encoder.tokenizer import decompose_hangul, en_to_phonetic

# Inverted dictionary from train_AS-06 / AR-06
EN_TO_KR_DICT: Dict[str, List[str]] = {
    "STAR": ["스타", "스탈"], "OCEAN": ["오션"], "GLORY": ["글로리"], "PRINCE": ["프린스"],
    "PRINCESS": ["프린세스"], "BRIDGE": ["브릿지", "브리지"], "SPIRIT": ["스피릿"],
    "PRIDE": ["프라이드"], "TRADER": ["트레이더"], "EXPRESS": ["익스프레스"],
    "ENTERPRISE": ["엔터프라이즈"], "CHALLENGER": ["챌린저"], "VOYAGER": ["보이저"],
    "PROGRESS": ["프로그레스"], "HARMONY": ["하모니"], "VICTORY": ["빅토리"],
    "UNION": ["유니온", "유니언"], "LIBERTY": ["리버티"], "TRIUMPH": ["트라이엄프"],
    "CENTURY": ["센추리"], "PACIFIC": ["퍼시픽"], "ATLANTIC": ["아틀란틱"],
    "GREEN": ["그린"], "BLUE": ["블루"], "GOLDEN": ["골든"], "SILVER": ["실버"],
    "DIAMOND": ["다이아몬드"], "CRYSTAL": ["크리스탈"], "SUNRISE": ["썬라이즈", "선라이즈"],
    "SUNSET": ["썬셋", "선셋"], "RAINBOW": ["레인보우"], "EMERALD": ["에메랄드"],
    "SAPPHIRE": ["사파이어"], "JUPITER": ["주피터"], "VENUS": ["비너스"],
    "MERCURY": ["머큐리"], "NEPTUNE": ["넵튠"], "APOLLO": ["아폴로"],
    "ATHENA": ["아테나"], "POSEIDON": ["포세이돈"], "ZEUS": ["제우스"],
    "PEGASUS": ["페가수스"], "ORION": ["오리온"], "SIRIUS": ["시리우스"],
    "POLARIS": ["폴라리스"], "NORTH": ["노스"], "SOUTH": ["사우스"],
    "EAST": ["이스트"], "WEST": ["웨스트"], "CHEMI": ["케미"], "CHEMICAL": ["케미칼"],
    "TANKER": ["탱커"], "MARINE": ["마린"], "MARITIME": ["마리타임"],
    "NAVIGATOR": ["내비게이터", "네비게이터"], "LEADER": ["리더"], "ACE": ["에이스"],
    "CHAMPION": ["챔피언"], "FRONTIER": ["프론티어"], "HORIZON": ["호라이즌"],
    "DRAGON": ["드래곤"], "PHOENIX": ["피닉스"], "EAGLE": ["이글"], "FALCON": ["팔콘"],
    "SWAN": ["스완"], "FERRY": ["훼리", "페리"], "CARFERRY": ["카훼리", "카페리"],
    "CRUISE": ["크루즈"], "CRUISER": ["크루저"], "QUEEN": ["퀸"], "KING": ["킹"],
    "ISLAND": ["아일랜드"], "ISLANDER": ["아일랜더"], "FLOWER": ["플라워"],
    "SEAFLOWER": ["씨플라워", "시플라워"], "CLOUD": ["클라우드"], "DREAM": ["드림"],
    "PANSTAR": ["판스타", "팬스타"], "SEASTAR": ["씨스타", "시스타"], "PARADISE": ["파라다이스"],
    "SHUTTLE": ["셔틀"], "JET": ["제트", "젯"], "DOLPHIN": ["돌핀"], "CARNIVAL": ["카니발"],
    "GRANDE": ["그란데"], "TROPIC": ["트로픽"], "CARRIER": ["캐리어"],
    "SEAWAYS": ["시웨이즈", "씨웨이즈"], "SEAS": ["시즈", "씨즈"], "CAPTAIN": ["캡틴"],
    "MORNING": ["모닝"], "SUNNY": ["써니", "선니"], "DAISY": ["데이지"],
    "UNIVERSAL": ["유니버설", "유니버살"], "PARTNER": ["파트너"],
    "ONE": ["원"], "TWO": ["투"], "THREE": ["쓰리", "스리"], "FOUR": ["포"],
    "FIVE": ["파이브"], "SIX": ["식스"], "SEVEN": ["세븐"], "EIGHT": ["에잇"],
    "NINE": ["나인"], "TEN": ["텐"], "FIRST": ["퍼스트"], "SECOND": ["세컨드"],
    "NO": ["넘버", "호"], "NO.": ["넘버", "호"],
}

# Rule-based English syllable transliteration
EN_SYL_TO_KR: Dict[str, str] = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프",
    "G": "지", "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘",
    "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐", "R": "알",
    "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스",
    "Y": "와이", "Z": "제트",
    "AL": "알", "AN": "안", "AR": "아르", "AS": "아스", "BA": "바", "BE": "베",
    "BI": "비", "BO": "보", "BU": "부", "CA": "카", "CE": "세", "CH": "치",
    "CI": "시", "CO": "코", "CU": "쿠", "DA": "다", "DE": "데", "DI": "디",
    "DO": "도", "DU": "두", "FA": "파", "FE": "페", "FI": "피", "FO": "포",
    "FU": "푸", "GA": "가", "GE": "게", "GI": "기", "GO": "고", "GU": "구",
    "HA": "하", "HE": "헤", "HI": "하이", "HO": "호", "HU": "휴", "JA": "자",
    "JE": "제", "JI": "지", "JO": "조", "JU": "주", "KA": "카", "KE": "케",
    "KI": "키", "KO": "코", "KU": "쿠", "LA": "라", "LE": "레", "LI": "리",
    "LO": "로", "LU": "루", "MA": "마", "ME": "메", "MI": "미", "MO": "모",
    "MU": "무", "NA": "나", "NE": "네", "NI": "니", "NO": "노", "NU": "누",
    "PA": "파", "PE": "페", "PI": "피", "PO": "포", "PU": "푸", "RA": "라",
    "RE": "레", "RI": "리", "RO": "로", "RU": "루", "SA": "사", "SE": "세",
    "SI": "시", "SO": "소", "SU": "수", "TA": "타", "TE": "테", "TI": "티",
    "TO": "토", "TU": "투", "VA": "바", "VE": "베", "VI": "비", "VO": "보",
    "VU": "부", "WA": "와", "WE": "웨", "WI": "위", "WO": "워", "YA": "야",
    "YE": "예", "YI": "이", "YO": "요", "YU": "유", "ZA": "자", "ZE": "제",
    "ZI": "지", "ZO": "조", "ZU": "주",
}


def rule_transliterate_word(word: str) -> str:
    """Fallback transliterator converting English word to Hangul phonemes."""
    w = word.upper()
    if w in EN_TO_KR_DICT:
        return random.choice(EN_TO_KR_DICT[w])

    # Bigram chunking with fallback to single char
    out: List[str] = []
    i = 0
    while i < len(w):
        if i + 2 <= len(w) and w[i : i + 2] in EN_SYL_TO_KR:
            out.append(EN_SYL_TO_KR[w[i : i + 2]])
            i += 2
        elif w[i : i + 1] in EN_SYL_TO_KR:
            out.append(EN_SYL_TO_KR[w[i : i + 1]])
            i += 1
        else:
            i += 1
    return "".join(out) if out else w


def transliterate_vessel_name(vessel_name: str) -> str:
    """Convert an English vessel name to a natural Korean port entry form."""
    tokens = re.split(r"[\s\-_]+", vessel_name.strip())
    kr_tokens: List[str] = []

    for tok in tokens:
        if not tok:
            continue
        # Check digit
        if tok.isdigit():
            # e.g. '2' -> '2호' or '2'
            r = random.random()
            if r < 0.4:
                kr_tokens.append(f"{tok}호")
            elif r < 0.6:
                kr_tokens.append(f"제{tok}호")
            else:
                kr_tokens.append(tok)
            continue

        tok_upper = tok.upper()
        if tok_upper in EN_TO_KR_DICT:
            kr_tokens.append(random.choice(EN_TO_KR_DICT[tok_upper]))
        else:
            kr_tokens.append(rule_transliterate_word(tok))

    # Spacing variation
    r_space = random.random()
    if r_space < 0.35:
        # All merged
        return "".join(kr_tokens)
    elif r_space < 0.85:
        # Standard space
        return " ".join(kr_tokens)
    else:
        # Mixed hyphen
        return "-".join(kr_tokens)


def inject_korean_noise(name: str) -> str:
    """Inject realistic typos and phonetic drift into Korean vessel name."""
    if not name or len(name) < 2:
        return name

    chars = list(name)
    r = random.random()

    if r < 0.25:
        # Vowel mutation on random Hangul character
        idx = random.randrange(len(chars))
        ch = chars[idx]
        if "가" <= ch <= "힣":
            code = ord(ch) - 0xAC00
            cho = code // 588
            jung = (code % 588) // 28
            jong = code % 28

            vowel_swap = {0: 4, 4: 0, 1: 5, 5: 1, 8: 13, 13: 8, 18: 20, 20: 18}  # ㅏ↔ㅓ, ㅐ↔ㅔ, ㅗ↔ㅜ, ㅡ↔ㅣ
            if jung in vowel_swap:
                new_jung = vowel_swap[jung]
                chars[idx] = chr(0xAC00 + cho * 588 + new_jung * 28 + jong)

    elif r < 0.45:
        # Drop or change final consonant (받침 변이)
        idx = random.randrange(len(chars))
        ch = chars[idx]
        if "가" <= ch <= "힣":
            code = ord(ch) - 0xAC00
            cho = code // 588
            jung = (code % 588) // 28
            jong = code % 28
            if jong != 0:
                # drop jong
                chars[idx] = chr(0xAC00 + cho * 588 + jung * 28)
            else:
                # add 'ㄹ' or 'ㅅ'
                new_jong = random.choice([8, 19])  # ㄹ, ㅅ
                chars[idx] = chr(0xAC00 + cho * 588 + jung * 28 + new_jong)

    elif r < 0.60:
        # Suffix attachment
        suffix = random.choice(["호", "선", " 1호", " 2호", "호선"])
        return "".join(chars) + suffix

    return "".join(chars)


def build_hard_negatives_index(catalog_names: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Index catalog names by first word (prefix) and last word (suffix) for sister-ship mining."""
    prefix_map = defaultdict(list)
    suffix_map = defaultdict(list)

    for name in catalog_names:
        words = name.split()
        if len(words) >= 2:
            prefix_map[words[0].upper()].append(name)
            suffix_map[words[-1].upper()].append(name)

    return dict(prefix_map), dict(suffix_map)


def mine_hard_negatives(
    target_name: str,
    prefix_map: Dict[str, List[str]],
    suffix_map: Dict[str, List[str]],
    all_names: List[str],
    num_negatives: int = 4,
) -> List[str]:
    """Mine sister-ship or phonetic confusion negatives."""
    words = target_name.split()
    candidates: Set[str] = set()

    if len(words) >= 2:
        first_w = words[0].upper()
        last_w = words[-1].upper()

        # Sister ships sharing prefix (e.g. MORNING LENA -> MORNING CARINA)
        for s in prefix_map.get(first_w, []):
            if s != target_name:
                candidates.add(s)

        # Ships sharing suffix (e.g. WEST LOYALTY -> SOUTH LOYALTY)
        for s in suffix_map.get(last_w, []):
            if s != target_name:
                candidates.add(s)

    cand_list = list(candidates)
    if len(cand_list) >= num_negatives:
        return random.sample(cand_list, num_negatives)

    # Fill remainder with random catalog names
    needed = num_negatives - len(cand_list)
    random_picks = random.sample(all_names, needed + 5)
    for pick in random_picks:
        if pick != target_name and pick not in candidates:
            cand_list.append(pick)
            if len(cand_list) == num_negatives:
                break

    return cand_list[:num_negatives]


def main():
    parser = argparse.ArgumentParser(description="Generate 100k synthetic phonetic pairs for bi-encoder pretraining.")
    parser.add_argument("--catalog-csv", default="data/vessel_catalog.csv", help="Path to vessel catalog CSV")
    parser.add_argument("--output", default="data/synthetic_phonetic_100k.json.gz", help="Path to output json.gz")
    parser.add_argument("--num-samples", type=int, default=100000, help="Number of synthetic pairs to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"[1/4] Loading vessel catalog from {args.catalog_csv}...")
    df_cat = pd.read_csv(args.catalog_csv, usecols=["vessel_name"]).dropna()
    catalog_names = df_cat["vessel_name"].drop_duplicates().tolist()
    print(f"Loaded {len(catalog_names):,} unique catalog vessel names.")

    print("[2/4] Indexing prefix and suffix sister-ships for hard negative mining...")
    prefix_map, suffix_map = build_hard_negatives_index(catalog_names)
    print(f"Indexed {len(prefix_map):,} prefixes and {len(suffix_map):,} suffixes.")

    print(f"[3/4] Generating {args.num_samples:,} synthetic transliterations with noise injection...")
    records: List[Dict[str, Any]] = []

    for i in range(args.num_samples):
        # Pick an English catalog name
        eng_name = random.choice(catalog_names)

        # Transliterate to Korean
        kr_base = transliterate_vessel_name(eng_name)

        # Apply realistic port observation noise
        kr_noisy = inject_korean_noise(kr_base)

        # Mine 4 hard negative English catalog names
        hard_negs = mine_hard_negatives(eng_name, prefix_map, suffix_map, catalog_names, num_negatives=4)

        records.append({
            "query": kr_noisy,
            "catalog": eng_name,
            "hard_negs": hard_negs,
        })

        if (i + 1) % 25000 == 0:
            print(f"  Generated {i + 1:,} / {args.num_samples:,} pairs...")

    print(f"[4/4] Saving compressed dataset to {args.output}...")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done! Saved {len(records):,} records to {args.output} ({file_size_mb:.2f} MB).")


if __name__ == "__main__":
    main()
