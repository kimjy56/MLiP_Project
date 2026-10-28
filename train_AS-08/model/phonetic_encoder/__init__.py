"""
Phonetic Encoder Package (SPEC-DPR-01)
Dense Phonetic Retrieval & Multi-View Phonetic Embedding for Korean-English Vessel Matching.
"""

from .tokenizer import PhoneticTokenizer
from .model import CharPhoneticBiEncoder

try:
    from .hybrid_matcher import HybridDenseLexicalMatcher
except ImportError:
    HybridDenseLexicalMatcher = None

try:
    from .loss import SymmetricInfoNCELoss, HybridContrastiveLoss
except ImportError:
    SymmetricInfoNCELoss = None
    HybridContrastiveLoss = None

__all__ = [
    "PhoneticTokenizer",
    "CharPhoneticBiEncoder",
    "HybridDenseLexicalMatcher",
    "SymmetricInfoNCELoss",
    "HybridContrastiveLoss",
]

