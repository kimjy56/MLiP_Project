"""
Multi-View Phonetic Tokenizer for Korean-English Vessel Matching (SPEC-DPR-01).

Combines:
  1. Korean query multi-view:
     - Clean Hangul: [KOR] ...
     - Jamo decomposition (초/중/종성): [JAMO] ...
     - Revised Romanization (RR): [ROM] ...
  2. English catalog multi-view:
     - Clean English name: [ENG] ...
     - Phonetic skeleton (consonant contraction / silent letter reduction): [PHON] ...
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

# Standard Hangul Unicode bounds
_HANGUL_START = 0xAC00
_HANGUL_END = 0xD7A3
_N_VOWEL = 21
_N_TAIL = 28

_CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]
_JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]
_JONGSEONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

# Romanization lookup tables
CHOSUNG_ROM = ["G", "KK", "N", "D", "TT", "R", "M", "B", "PP", "S", "SS", "", "J", "JJ", "CH", "K", "T", "P", "H"]
JUNGSUNG_ROM = ["A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WO", "WE", "WI", "YU", "EU", "UI", "I"]
JONGSUNG_ROM = ["", "K", "K", "K", "N", "N", "N", "T", "L", "K", "M", "P", "L", "L", "P", "L", "M", "P", "P", "T", "T", "NG", "T", "T", "K", "T", "P", "T"]

# Phonetic reduction maps
_EN_PHONETIC_REPLACEMENTS = [
    ("TCH", "C"), ("SCH", "S"), ("SH", "S"), ("CH", "C"),
    ("PH", "P"), ("GH", ""), ("CK", "K"), ("QU", "KW"),
    ("WR", "R"), ("KN", "N"), ("WH", "W"), ("TH", "S"),
    ("CE", "SE"), ("CI", "SI"), ("CY", "SY"),  # Soft C before E, I, Y (e.g. CECIL -> SESIL, PACIFIC -> PASIFIK)
    ("NG", "N"),
    ("WO", "U"), ("WU", "U"),  # SPEC-DPR-02: Semivowel W alignment (WOLF <-> 울프)
    ("OO", "U"), ("EE", "I"), ("EA", "I"), ("OA", "O"),
    ("OU", "U"), ("OW", "O"), ("AI", "E"), ("AY", "E"),
    ("AU", "O"), ("AW", "O"), ("EW", "U"), ("IE", "I"),
]

_EN_CONSONANT_TRANS = str.maketrans({
    "G": "K", "C": "K", "Q": "K",
    "B": "P", "V": "P", "F": "P",
    "D": "T", "L": "R", "Z": "S", "X": "KS",
})


def decompose_hangul(text: str) -> str:
    """Decompose Hangul syllables into standard Unicode Jamo characters."""
    out: List[str] = []
    for char in text:
        code = ord(char)
        if _HANGUL_START <= code <= _HANGUL_END:
            code -= _HANGUL_START
            out.append(_CHOSEONG[code // (_N_VOWEL * _N_TAIL)])
            out.append(_JUNGSEONG[(code % (_N_VOWEL * _N_TAIL)) // _N_TAIL])
            tail = code % _N_TAIL
            if tail:
                out.append(_JONGSEONG[tail])
        else:
            out.append(char)
    return "".join(out)


def syllable_to_rom(ch: str) -> str:
    """Convert a single Hangul syllable to revised romanization."""
    code = ord(ch) - _HANGUL_START
    if code < 0 or code > 11171:
        return ch
    cho = code // 588
    jung = (code % 588) // 28
    jong = code % 28
    return CHOSUNG_ROM[cho] + JUNGSUNG_ROM[jung] + JONGSUNG_ROM[jong]


# Multi-view non-phonetic exception mappings (SPEC-DPR-01 & SPEC-DPR-02)
KR_LETTER_MAP: List[Tuple[str, str]] = [
    ("에이치", "H"), ("애이치", "H"), ("더블유", "W"),
    ("에스티", "ST"), ("에스케이", "SK"), ("에이치엠엠", "HMM"),
    ("에스티엑스", "STX"), ("시앤시", "CNC"), ("씨앤씨", "CNC"),
    ("에이", "A"), ("애이", "A"), ("비", "B"), ("씨", "C"), ("시", "C"), ("디", "D"),
    ("이", "E"), ("에프", "F"), ("애프", "F"), ("지", "G"), ("쥐", "G"), ("아이", "I"),
    ("제이", "J"), ("재이", "J"), ("케이", "K"), ("엘", "L"), ("앨", "L"), ("엠", "M"),
    ("앰", "M"), ("엔", "N"), ("앤", "N"), ("오", "O"), ("피", "P"), ("큐", "Q"),
    ("알", "R"), ("아르", "R"), ("에스", "S"), ("애스", "S"), ("티", "T"), ("유", "U"),
    ("브이", "V"), ("엑스", "X"), ("액스", "X"), ("와이", "Y"), ("제트", "Z"), ("지트", "Z")
]
KR_LETTER_MAP.sort(key=lambda x: -len(x[0]))


def hangul_to_rom(text: str) -> str:
    """
    Romanize Hangul text syllable-by-syllable, with automatic conversion of
    consecutive letter-spellings (e.g. '씨엔씨 드림' -> 'CNC DEURIM', '에스케이 프론티어' -> 'SK PEURONTIEO')
    per SPEC-DPR-02.
    """
    if not text:
        return ""
    words = text.split()
    res_words: List[str] = []

    for w in words:
        curr = w
        acrs: List[str] = []
        while curr:
            matched = False
            for kr, en in KR_LETTER_MAP:
                if curr.startswith(kr):
                    acrs.append(en)
                    curr = curr[len(kr):]
                    matched = True
                    break
            if not matched:
                break

        full_acr = "".join(acrs)
        if len(full_acr) >= 2 and not curr:
            res_words.append(full_acr)
        elif len(full_acr) >= 2 and curr:
            rem_rom = "".join(syllable_to_rom(ch) if "가" <= ch <= "힣" else ch for ch in curr)
            res_words.append(full_acr + " " + rem_rom)
        else:
            rom_w = "".join(syllable_to_rom(ch) if "가" <= ch <= "힣" else ch for ch in w)
            res_words.append(rom_w)

    return " ".join(res_words)


def extract_tail_core_text(name: str) -> str:
    """
    Extract unique core/tail vessel name by stripping administrative numerals/suffixes
    and dropping the leading fleet prefix (e.g. GLOVIS, SEA, BOW, HAFNIA, MORNING)
    if the remaining name has 2 or more words (SPEC-DPR-02).
    Example:
      '글로비스 심포니' -> '심포니'
      'GLOVIS SYMPHONY' -> 'SYMPHONY'
      '씨 울프' -> '울프'
      'SEA WOLF' -> 'WOLF'
      '모닝 레나 2호' -> '레나'
      'MORNING LENA NO.2' -> 'LENA'
      '한나' -> '한나'
    """
    if not name:
        return ""
    s = clean_vessel_name(name)
    s = canonicalize_numerals(s)
    words = s.split()
    while words and (words[-1].isdigit() or words[-1] in ("호", "HO", "NO") or re.match(r"^[0-9]+호?$", words[-1])):
        words.pop()

    if len(words) >= 2:
        return " ".join(words[1:])
    elif len(words) == 1:
        return words[0]
    return clean_vessel_name(name)


def en_to_phonetic(word: str) -> str:
    """Create an English phonetic skeleton with consonant contractions."""
    if not word:
        return ""
    s = re.sub(r"[^A-Z0-9]", "", word.upper())
    if not s:
        return ""
    for pat, rep in _EN_PHONETIC_REPLACEMENTS:
        s = s.replace(pat, rep)
    if not s:
        return ""
    if len(s) >= 4 and s.endswith("E") and s[-2] not in "AEIOU":
        s = s[:-1]
    s = s.translate(_EN_CONSONANT_TRANS)
    if not s:
        return ""
    out = [s[0]]
    for ch in s[1:]:
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


# Multi-view non-phonetic exception mappings (SPEC-DPR-01 Multi-View Expansion)
KR_LETTER_MAP: List[Tuple[str, str]] = [
    ("에이치", "H"), ("애이치", "H"), ("더블유", "W"),
    ("에스티", "ST"), ("에스케이", "SK"), ("에이치엠엠", "HMM"),
    ("에이", "A"), ("애이", "A"), ("비", "B"), ("씨", "C"), ("시", "C"), ("디", "D"),
    ("이", "E"), ("에프", "F"), ("애프", "F"), ("지", "G"), ("쥐", "G"), ("아이", "I"),
    ("제이", "J"), ("재이", "J"), ("케이", "K"), ("엘", "L"), ("앨", "L"), ("엠", "M"),
    ("앰", "M"), ("엔", "N"), ("앤", "N"), ("오", "O"), ("피", "P"), ("큐", "Q"),
    ("알", "R"), ("아르", "R"), ("에스", "S"), ("애스", "S"), ("티", "T"), ("유", "U"),
    ("브이", "V"), ("엑스", "X"), ("액스", "X"), ("와이", "Y"), ("제트", "Z"), ("지트", "Z")
]
KR_LETTER_MAP.sort(key=lambda x: -len(x[0]))

EN_LETTER_NAME_MAP = {
    "A": "에이", "B": "비", "C": "씨", "D": "디", "E": "이", "F": "에프",
    "G": "지", "H": "에이치", "I": "아이", "J": "제이", "K": "케이", "L": "엘",
    "M": "엠", "N": "엔", "O": "오", "P": "피", "Q": "큐", "R": "알",
    "S": "에스", "T": "티", "U": "유", "V": "브이", "W": "더블유", "X": "엑스",
    "Y": "와이", "Z": "제트"
}

EN_SPECIAL_ACRONYMS = {
    "SK": "에스케이", "CNC": "시앤시", "STX": "에스티엑스", "HMM": "에이치엠엠",
    "SM": "에스엠", "KMT": "케이엠티", "JB": "제이비", "GL": "지엘",
    "JS": "제이에스", "CJ": "씨제이", "POS": "피오에스",
}

KNOWN_COMPANY_INITIALS = {
    "현대": "HD", "삼선": "SS", "동아": "DA", "대한": "DH", "한진": "HJ",
    "금호": "KH", "고려": "KR", "흥아": "HA", "팬오션": "POS", "남해": "NH",
    "동서": "DS", "대우": "DW", "삼성": "SHI", "포스코": "POSCO", "글로벌": "GLO"
}

EN_COMPANY_ACRONYMS = {v: k for k, v in KNOWN_COMPANY_INITIALS.items()}

SEMANTIC_MAP = {
    "신": "NEW", "태양": "SUN", "해양": "OCEAN", "바다": "SEA",
    "동": "EAST", "서": "WEST", "남": "SOUTH", "북": "NORTH",
    "황금": "GOLDEN", "청": "BLUE", "백": "WHITE", "흑": "BLACK",
    "승리": "VICTORY", "평화": "PEACE", "희망": "HOPE", "행복": "HAPPY",
    "하나": "ONE", "제일": "FIRST", "에이스": "ACE"
}

EN_SEMANTIC_MAP = {v: k for k, v in SEMANTIC_MAP.items()}

NUMERAL_PATTERNS = [
    (r"제\s*([0-9]+)\s*호?", r"\1"),
    (r"([0-9]+)\s*호", r"\1"),
    (r"\bNO\.?\s*([0-9]+)\b", r"\1"),
    (r"\b(ONE|FIRST|I|일호)\b", "1"),
    (r"\b(TWO|SECOND|II|이호)\b", "2"),
    (r"\b(THREE|THIRD|III|삼호)\b", "3"),
    (r"\b(FOUR|FOURTH|IV|사호)\b", "4"),
    (r"\b(FIVE|FIFTH|V|오호)\b", "5"),
    (r"\b(SIX|SIXTH|VI|육호)\b", "6"),
    (r"\b(SEVEN|SEVENTH|VII|칠호)\b", "7"),
    (r"\b(EIGHT|EIGHTH|VIII|팔호)\b", "8"),
    (r"\b(NINE|NINTH|IX|구호)\b", "9"),
    (r"\b(TEN|TENTH|X|십호)\b", "10"),
]


def canonicalize_numerals(text: str) -> str:
    s = text.upper()
    for pat, rep in NUMERAL_PATTERNS:
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s).strip()


def extract_acronym_view(text: str) -> Optional[str]:
    """Convert letter spellings (에스케이 -> SK, 시앤시 -> CNC) in Korean text."""
    words = text.split()
    res = []
    has_acr = False
    for w in words:
        curr = w
        acrs = []
        while curr:
            matched = False
            for kr, en in KR_LETTER_MAP:
                if curr.startswith(kr):
                    acrs.append(en)
                    curr = curr[len(kr):]
                    matched = True
                    break
            if not matched:
                break
        if len("".join(acrs)) >= 2 and not curr:
            res.append("".join(acrs))
            has_acr = True
        elif len("".join(acrs)) >= 2 and curr:
            res.append("".join(acrs) + " " + curr)
            has_acr = True
        else:
            res.append(w)
    return " ".join(res) if has_acr else None


def extract_initial_view(text: str) -> Optional[str]:
    """Extract Chosung initials for Korean company/words (현대 -> HD, 삼선 -> SS, 동아 -> DA, 대한 -> DH)."""
    words = text.split()
    res = []
    has_init = False
    for w in words:
        if w in KNOWN_COMPANY_INITIALS:
            res.append(KNOWN_COMPANY_INITIALS[w])
            has_init = True
        else:
            # Check prefix match for compound words (e.g. 현대포스 -> HD 포스)
            matched = False
            for kr, en in KNOWN_COMPANY_INITIALS.items():
                if w.startswith(kr) and len(w) > len(kr):
                    res.append(en + " " + w[len(kr):])
                    has_init = True
                    matched = True
                    break
            if not matched:
                res.append(w)
    return " ".join(res) if has_init else None


def extract_semantic_view(text: str) -> Optional[str]:
    """Extract semantic translations for common maritime terms (신 -> NEW, 태양 -> SUN)."""
    words = text.split()
    res = []
    has_sem = False
    for w in words:
        if w in SEMANTIC_MAP:
            res.append(SEMANTIC_MAP[w])
            has_sem = True
        else:
            matched = False
            for kr, en in SEMANTIC_MAP.items():
                if w.startswith(kr) and len(kr) >= 2:
                    res.append(en + " " + w[len(kr):])
                    has_sem = True
                    matched = True
                    break
            if not matched:
                res.append(w)
    return " ".join(res) if has_sem else None


def expand_latin_acronym(text: str) -> Optional[str]:
    words = text.upper().split()
    res = []
    has_exp = False
    for w in words:
        clean_w = re.sub(r"[^A-Z]", "", w)
        if clean_w in EN_SPECIAL_ACRONYMS:
            res.append(EN_SPECIAL_ACRONYMS[clean_w])
            has_exp = True
        elif 2 <= len(clean_w) <= 4 and all(c not in "AEIOU" for c in clean_w):
            expanded = "".join(EN_LETTER_NAME_MAP.get(c, c) for c in clean_w)
            res.append(expanded)
            has_exp = True
        else:
            res.append(w)
    return " ".join(res) if has_exp else None


def expand_latin_initials(text: str) -> Optional[str]:
    words = text.upper().split()
    res = []
    has_exp = False
    for w in words:
        clean_w = re.sub(r"[^A-Z]", "", w)
        if clean_w in EN_COMPANY_ACRONYMS:
            res.append(EN_COMPANY_ACRONYMS[clean_w])
            has_exp = True
        else:
            res.append(w)
    return " ".join(res) if has_exp else None


def expand_latin_semantics(text: str) -> Optional[str]:
    words = text.upper().split()
    res = []
    has_exp = False
    for w in words:
        clean_w = re.sub(r"[^A-Z]", "", w)
        if clean_w in EN_SEMANTIC_MAP:
            res.append(EN_SEMANTIC_MAP[clean_w])
            has_exp = True
        else:
            res.append(w)
    return " ".join(res) if has_exp else None


def clean_vessel_name(text: str) -> str:
    """Standardize vessel name by removing common administrative noise."""
    if not text:
        return ""
    s = str(text).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def format_hangul_query(raw_text: str) -> str:
    """
    Format Hangul vessel name into multi-view query string.
    Example: '할사이언 2호' -> '[KOR] 할사이언 2 [JAMO] ㅎㅏㄹㅅㅏㅇㅣㅇㅓㄴ 2 [ROM] HALSAI-EON 2'
    """
    clean = clean_vessel_name(raw_text)
    clean = canonicalize_numerals(clean)
    jamo = decompose_hangul(clean)
    rom = hangul_to_rom(clean)
    acr = extract_acronym_view(clean)
    init = extract_initial_view(clean)
    sem = extract_semantic_view(clean)

    parts = [f"[KOR] {clean}", f"[JAMO] {jamo}", f"[ROM] {rom}"]
    if acr:
        parts.append(f"[ACR] {acr}")
    if init:
        parts.append(f"[INIT] {init}")
    if sem:
        parts.append(f"[SEM] {sem}")
    return " ".join(parts)


def format_latin_catalog(raw_text: str) -> str:
    """
    Format Latin catalog vessel name into multi-view catalog string.
    Example: 'CNC DREAM' -> '[ENG] CNC DREAM [PHON] KNK TRM [ACR] 시앤시 DREAM'
    """
    clean = clean_vessel_name(raw_text).upper()
    clean = canonicalize_numerals(clean)
    phon = en_to_phonetic(clean)
    acr = expand_latin_acronym(clean)
    init = expand_latin_initials(clean)
    sem = expand_latin_semantics(clean)

    parts = [f"[ENG] {clean}", f"[PHON] {phon}"]
    if acr:
        parts.append(f"[ACR] {acr}")
    if init:
        parts.append(f"[INIT] {init}")
    if sem:
        parts.append(f"[SEM] {sem}")
    return " ".join(parts)


class PhoneticTokenizer:
    """
    Character/Jamo Tokenizer for Symphonym-v8 architecture.
    Provides compact vocabulary indexing (< 200 tokens) with 100% UTF-8 support.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SPACE_TOKEN = "<SPACE>"

    PAD_ID = 0
    UNK_ID = 1
    SPACE_ID = 2

    SCRIPT_MAP = {"OTHER": 0, "LATIN": 1, "HANGUL": 2}
    LANG_MAP = {"<UNK>": 0, "en": 1, "ko": 2}

    def __init__(self, vocab_file: Optional[Union[str, Path]] = None):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.script_to_id: Dict[str, int] = dict(self.SCRIPT_MAP)
        self.lang_to_id: Dict[str, int] = dict(self.LANG_MAP)

        if vocab_file is not None and Path(vocab_file).exists():
            self.load_vocab(vocab_file)
        else:
            self._build_default_vocab()

    def _build_default_vocab(self):
        """Construct the canonical character vocabulary for English and Hangul."""
        chars: List[str] = [self.PAD_TOKEN, self.UNK_TOKEN, self.SPACE_TOKEN]

        # Printable ASCII (33 to 126)
        for i in range(33, 127):
            chars.append(chr(i))

        # Standard Hangul Jamo (Choseong, Jungseong, Jongseong)
        jamo_all = (
            "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
            "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
            "ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ"
        )
        for c in jamo_all:
            if c not in chars:
                chars.append(c)

        # Unicode Hangul Compatibility Jamo block (U+3131 to U+3163)
        for i in range(0x3131, 0x3164):
            ch = chr(i)
            if ch not in chars:
                chars.append(ch)

        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    @property
    def num_scripts(self) -> int:
        return len(self.script_to_id)

    @property
    def num_langs(self) -> int:
        return len(self.lang_to_id)

    def save_vocab(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "char_to_id": self.char_to_id,
            "script_to_id": self.script_to_id,
            "lang_to_id": self.lang_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: Union[str, Path]):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char_to_id = data["char_to_id"]
        self.id_to_char = {int(i) if str(i).isdigit() else i: c for c, i in self.char_to_id.items()}
        self.script_to_id = data.get("script_to_id", dict(self.SCRIPT_MAP))
        self.lang_to_id = data.get("lang_to_id", dict(self.LANG_MAP))

    def encode(
        self,
        text: str,
        script: str = "LATIN",
        lang: str = "en",
        max_length: int = 96,
    ) -> Tuple[List[int], int, int, int]:
        """
        Encode raw string into (char_ids, script_id, lang_id, length).
        Hangul syllables are decomposed into Jamo before token id lookup.
        """
        # Decompose any raw Hangul into Jamo
        decomposed = decompose_hangul(text)
        decomposed = unicodedata.normalize("NFC", decomposed)

        char_ids: List[int] = []
        for ch in decomposed:
            if ch == " ":
                char_ids.append(self.SPACE_ID)
            elif not ch.strip():
                continue
            else:
                cid = self.char_to_id.get(ch, self.UNK_ID)
                char_ids.append(cid)

        if max_length is not None and len(char_ids) > max_length:
            char_ids = char_ids[:max_length]
        if not char_ids:
            char_ids = [self.UNK_ID]

        length = len(char_ids)
        s_id = self.script_to_id.get(script, self.SCRIPT_MAP["OTHER"])
        l_id = self.lang_to_id.get(lang, self.LANG_MAP["<UNK>"])
        return char_ids, s_id, l_id, length

    def batch_encode(
        self,
        items: List[Tuple[str, str, str]],  # (text, script, lang)
        max_length: int = 96,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encode a list of (text, script, lang) tuples.
        Returns a dictionary of PyTorch tensors padded to max sequence length in batch.
        """
        encoded = [self.encode(text, script, lang, max_length=max_length) for text, script, lang in items]
        batch_size = len(encoded)
        max_len = max(item[3] for item in encoded)

        char_tensor = torch.full((batch_size, max_len), self.PAD_ID, dtype=torch.long)
        script_tensor = torch.zeros(batch_size, dtype=torch.long)
        lang_tensor = torch.zeros(batch_size, dtype=torch.long)
        length_tensor = torch.zeros(batch_size, dtype=torch.long)

        for i, (c_ids, s_id, l_id, length) in enumerate(encoded):
            char_tensor[i, :length] = torch.tensor(c_ids, dtype=torch.long)
            script_tensor[i] = s_id
            lang_tensor[i] = l_id
            length_tensor[i] = length

        return {
            "char_ids": char_tensor.to(device),
            "script_ids": script_tensor.to(device),
            "lang_ids": lang_tensor.to(device),
            "lengths": length_tensor.to(device),
        }
