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
    ("NG", "N"),
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


def hangul_to_rom(text: str) -> str:
    """Romanize Hangul text syllable-by-syllable."""
    if not text:
        return ""
    out: List[str] = []
    for ch in text:
        if "가" <= ch <= "힣":
            out.append(syllable_to_rom(ch))
        else:
            out.append(ch)
    return "".join(out)


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
    Example: '할사이언 2호' -> '[KOR] 할사이언 2호 [JAMO] ㅎㅏㄹㅅㅏㅇㅣㅇㅓㄴ 2호 [ROM] HALSAI-EON 2HO'
    """
    clean = clean_vessel_name(raw_text)
    jamo = decompose_hangul(clean)
    rom = hangul_to_rom(clean)
    return f"[KOR] {clean} [JAMO] {jamo} [ROM] {rom}"


def format_latin_catalog(raw_text: str) -> str:
    """
    Format Latin catalog vessel name into multi-view catalog string.
    Example: 'HALCYON TWO' -> '[ENG] HALCYON TWO [PHON] HLSYN TU'
    """
    clean = clean_vessel_name(raw_text).upper()
    phon = en_to_phonetic(clean)
    return f"[ENG] {clean} [PHON] {phon}"


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
