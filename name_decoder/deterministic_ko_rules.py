"""General spelling/numeral/romanization rules; no vessel/company dictionary."""
from __future__ import annotations
import itertools
import re
import unicodedata

RULE_VERSION = "general-ko-v1"
LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ["에이","비","씨","디","이","에프","지","에이치","아이","제이","케이","엘","엠",
     "엔","오","피","큐","알","에스","티","유","브이","더블유","엑스","와이","제트"]))
EN_NUMBERS = dict(zip(
    "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN".split(), range(11)))
ORDINALS = dict(zip("FIRST SECOND THIRD FOURTH FIFTH SIXTH SEVENTH EIGHTH NINTH TENTH".split(),range(1,11)))
ROMAN_NUMBERS = dict(zip("I II III IV V VI VII VIII IX X".split(),range(1,11)))
SINO = ["영","일","이","삼","사","오","육","칠","팔","구"]
EN_CARDINAL_READINGS = dict(zip(EN_NUMBERS,
    ["제로","원","투","쓰리","포","파이브","식스","세븐","에이트","나인","텐"]))
# Revised/older romanization graphemes, not named-vessel aliases.
ONSETS = {"G":0,"K":0,"KK":1,"N":2,"D":3,"T":3,"TT":4,"R":5,"L":5,
          "M":6,"B":7,"P":7,"PP":8,"S":9,"SS":10,"":11,"J":12,"JJ":13,
          "CH":14,"C":14,"KH":15,"TH":16,"PH":17,"H":18}
VOWELS = {"A":0,"AE":1,"YA":2,"YAE":3,"EO":4,"E":5,"YEO":6,"YE":7,
          "O":8,"WA":9,"WAE":10,"OE":11,"YO":12,"U":13,"WEO":14,"WE":15,
          "WI":16,"YU":17,"EU":18,"UI":19,"I":20}
TAILS = {"":0,"K":1,"G":1,"N":4,"D":7,"T":7,"L":8,"R":8,"M":16,
         "B":17,"P":17,"S":19,"NG":21,"J":22,"CH":23,"H":27}
ALLOWED_KOREAN = re.compile(r"[가-힣0-9 ]+\Z")


def canonical_name(value):
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC",value).upper().split())


def word_tokens(value):
    # Keep accents and names intact rather than silently replacing non-ASCII letters.
    return re.findall(r"[^\W\d_]+|[0-9]+", canonical_name(value), flags=re.UNICODE)


def clean_korean(value):
    if not isinstance(value,str):
        return ""
    value=" ".join(unicodedata.normalize("NFKC",value).split())
    return value if value and ALLOWED_KOREAN.fullmatch(value) else ""


def observed_korean(value):
    value=clean_korean(value)
    # Only unambiguous standalone designators and digit+호 are removed.
    value=re.sub(r"\s+(?:호|선)$","",value)
    return re.sub(r"(?<=\d)\s*호$","",value).strip()


def sino_number(number):
    n=int(number)
    if not 0 <= n <= 9999:
        return "".join(SINO[int(c)] for c in str(n)) if n>=0 else ""
    if n==0:return SINO[0]
    parts=[]
    for unit,label in [(1000,"천"),(100,"백"),(10,"십"),(1,"")]:
        digit,n=divmod(n,unit)
        if digit:parts.append(("" if digit==1 and unit>1 else SINO[digit])+label)
    return "".join(parts)


def number_variants(token):
    token=canonical_name(token)
    if token.isascii() and token.isdigit():
        n=int(token)
    elif token in EN_NUMBERS:n=EN_NUMBERS[token]
    elif token in ORDINALS:n=ORDINALS[token]
    elif token in ROMAN_NUMBERS:n=ROMAN_NUMBERS[token]
    else:return []
    values=[str(n)]
    if token in EN_CARDINAL_READINGS:values.append(EN_CARDINAL_READINGS[token])
    values.extend([sino_number(n),"제"+str(n)])
    return list(dict.fromkeys(values))


def romanized_hangul(token):
    """Conservative whole-token parse; ambiguous English words remain model tasks."""
    text=canonical_name(token)
    if not re.fullmatch("[A-Z]+",text):return ""
    text=re.sub(r"YOU(?=NG)","YEO",text)
    onsets=sorted(ONSETS,key=lambda k:(-len(k),k))
    vowels=sorted(VOWELS,key=lambda k:(-len(k),k))
    memo={}
    def parse(pos):
        if pos==len(text):return ""
        if pos in memo:return memo[pos]
        for onset in onsets:
            if not text.startswith(onset,pos):continue
            a=pos+len(onset)
            for vowel in vowels:
                if not text.startswith(vowel,a):continue
                b=a+len(vowel)
                # Prefer no final consonant if another full syllable follows.
                for tail in ["",*sorted((k for k in TAILS if k),key=lambda k:(-len(k),k))]:
                    if not text.startswith(tail,b):continue
                    rest=parse(b+len(tail))
                    if rest is not None:
                        syllable=chr(0xAC00+(ONSETS[onset]*21+VOWELS[vowel])*28+TAILS[tail])
                        memo[pos]=syllable+rest
                        return memo[pos]
        memo[pos]=None
        return None
    return parse(0) or ""


def deterministic_variants(token):
    token=canonical_name(token)
    candidates=number_variants(token)
    if re.fullmatch(r"[B-DF-HJ-NP-TV-Z]{2,4}",token):
        candidates.append("".join(LETTERS[c] for c in token))
    # Old romanization is only an extra candidate, never a replacement.
    if any(marker in token for marker in ("EO","YO","KW","CH","NG")):
        candidate=romanized_hangul(token)
        if candidate:candidates.append(candidate)
        for old,new in [("CHON","CHEON"),("CHUNG","CHEONG")]:
            if old in token:
                extra=romanized_hangul(token.replace(old,new))
                if extra:candidates.append(extra)
    return list(dict.fromkeys(candidates))


def rule_training_pairs():
    """Exhaustive 1/2-letter spelling rules and bounded generic numeric rules."""
    pairs=[]
    for length in (1,2):
        for chars in itertools.product(LETTERS,repeat=length):
            pairs.append(("".join(chars),"".join(LETTERS[c] for c in chars)))
    for n in range(100):
        for target in number_variants(str(n)):
            pairs.append((str(n),target))
    return pairs


def compact_korean(value):
    return re.sub(r"\s+","",observed_korean(value))


def hangul_to_jamo(value, phonetic_norm=True):
    # NFD decomposes every Hangul syllable into stable onset/vowel/final jamo.
    text=canonical_name(value)
    text=re.sub(r"\s+(?:호|선)$","",text)
    text=re.sub(r"(?<=\d)\s*호$","",text)
    text=unicodedata.normalize("NFD",re.sub(r"\s+","",text))
    if phonetic_norm:
        text=text.translate(str.maketrans({"ᅢ":"ᅦ","ᅤ":"ᅨ","ᄁ":"ᄀ","ᄄ":"ᄃ","ᄈ":"ᄇ","ᄊ":"ᄉ","ᄍ":"ᄌ"}))
    return text


def digit_sequence(value):
    return tuple(int(n) for n in re.findall(r"[0-9]+",canonical_name(value)))
