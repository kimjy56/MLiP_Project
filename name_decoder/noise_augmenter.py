"""Deterministic Domain-Specific Noise Augmenter for Reverse Noisy Channel Training.

Generates realistic port record noise (typos, spacing shifts, prefix toggles)
strictly from training-fold catalog and observed name pairs.
Held-out validation rows are NEVER touched or augmented.
"""
from __future__ import annotations

import random
import re
from typing import List, Tuple

import unicodedata

from name_decoder.catalog import has_hangul


KEYBOARD_ADJACENT_EN = {
    "A": ["S", "Q", "Z"],
    "B": ["V", "G", "H", "N"],
    "C": ["X", "D", "F", "V"],
    "D": ["S", "E", "R", "F", "C", "X"],
    "E": ["W", "S", "D", "R"],
    "F": ["D", "R", "T", "G", "V", "C"],
    "G": ["F", "T", "Y", "H", "B", "V"],
    "H": ["G", "Y", "U", "J", "N", "B"],
    "I": ["U", "J", "K", "O"],
    "J": ["H", "U", "I", "K", "M", "N"],
    "K": ["J", "I", "O", "L", "M"],
    "L": ["K", "O", "P"],
    "M": ["N", "J", "K"],
    "N": ["B", "H", "J", "M"],
    "O": ["I", "K", "L", "P"],
    "P": ["O", "L"],
    "Q": ["W", "A"],
    "R": ["E", "D", "F", "T"],
    "S": ["A", "W", "E", "D", "X", "Z"],
    "T": ["R", "F", "G", "Y"],
    "U": ["Y", "H", "J", "I"],
    "V": ["C", "F", "G", "B"],
    "W": ["Q", "A", "S", "E"],
    "X": ["Z", "S", "D", "C"],
    "Y": ["T", "G", "H", "U"],
    "Z": ["A", "S", "X"],
}


def augment_english_name(name: str, rng: random.Random) -> str:
    """Generate realistic English shipping name noise."""
    name = unicodedata.normalize("NFC", name).strip()
    if len(name) < 4:
        return name

    op = rng.choice(["prefix", "transpose", "delete", "duplicate", "substitute"])

    if op == "prefix":
        # Toggle common maritime prefixes (M/V, MT, LPG, LNG)
        prefixes = ["M/V ", "MV ", "MT ", "M.V. "]
        for p in prefixes:
            if name.upper().startswith(p):
                return name[len(p):].strip()
        # If no prefix, randomly add one
        return rng.choice(["M/V ", "MT "]) + name

    words = name.split()
    # Pick a word with length >= 4
    cands = [i for i, w in enumerate(words) if len(w) >= 4 and re.match(r"^[A-Za-z]+$", w)]
    if not cands:
        return name

    w_idx = rng.choice(cands)
    word = words[w_idx]

    if op == "transpose":
        positions = [i for i in range(1, len(word) - 2) if word[i] != word[i + 1]]
        if positions:
            pos = rng.choice(positions)
            word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
    elif op == "delete":
        pos = rng.randint(1, len(word) - 2)
        word = word[:pos] + word[pos + 1:]
    elif op == "duplicate":
        pos = rng.randint(1, len(word) - 2)
        word = word[:pos] + word[pos] * 2 + word[pos + 1:]
    elif op == "substitute":
        pos = rng.randint(1, len(word) - 2)
        ch = word[pos].upper()
        if ch in KEYBOARD_ADJACENT_EN:
            rep = rng.choice(KEYBOARD_ADJACENT_EN[ch])
            if word[pos].islower():
                rep = rep.lower()
            word = word[:pos] + rep + word[pos + 1:]

    words[w_idx] = word
    return " ".join(words)


def augment_hangul_name(name: str, rng: random.Random) -> str:
    """Generate realistic Korean shipping name noise (spacing shifts, transliteration noise)."""
    name = unicodedata.normalize("NFC", name).strip()
    if len(name) < 3:
        return name

    op = rng.choice(["space_collapse", "space_split", "vowel_shift", "particle_drop"])

    if op == "space_collapse" and " " in name:
        return re.sub(r"\s+", "", name)

    if op == "space_split" and " " not in name and len(name) >= 4:
        split_idx = rng.randint(2, len(name) - 2)
        return name[:split_idx] + " " + name[split_idx:]

    if op == "vowel_shift":
        # Common vowel / transliteration shifts in Korean maritime records
        shifts = [
            ("웨", "예"), ("예", "웨"),
            ("위", "이"), ("이", "위"),
            ("오", "우"), ("우", "오"),
            ("아", "야"), ("야", "아"),
            ("퍼", "포"), ("포", "퍼"),
        ]
        rng.shuffle(shifts)
        for s_from, s_to in shifts:
            if s_from in name:
                return name.replace(s_from, s_to, 1)

    if op == "particle_drop":
        # Trailing syllable / particle variations
        suffixes = ["호", "호선", "1호", "2호"]
        for s in suffixes:
            if name.endswith(s) and len(name) > len(s) + 2:
                return name[:-len(s)].strip()

    return name


def generate_augmented_pairs(
    pairs_df,
    seed: int = 42,
    augment_factor: float = 0.5,
) -> List[Tuple[str, str]]:
    """Generate noise-augmented (catalog_name, noisy_observed_name) pairs.

    Args:
        pairs_df: DataFrame with 'catalog_name', 'observed_name', 'hangul'
        seed: Random seed for deterministic reproducibility
        augment_factor: Ratio of additional noise pairs to generate (e.g. 0.5 = 50% more)
    """
    rng = random.Random(seed)
    n_target = int(len(pairs_df) * augment_factor)

    records = pairs_df.to_dict("records")
    sampled = rng.sample(records, k=min(n_target, len(records)))

    augmented = []
    for r in sampled:
        cat_name = r["catalog_name"]
        obs_name = r["observed_name"]
        is_kr = r["hangul"]

        if is_kr:
            noisy_obs = augment_hangul_name(obs_name, rng)
        else:
            noisy_obs = augment_english_name(obs_name, rng)

        if noisy_obs and noisy_obs != obs_name:
            augmented.append((cat_name, noisy_obs))

    return augmented
