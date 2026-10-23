"""Lossless name keys, catalog identities, and a byte-token prefix tree."""
from dataclasses import dataclass
import re
import unicodedata


def canonical_name(value: str) -> str:
    """Keep digits/punctuation; never collapse ONE into 1 or remove suffixes."""
    if not isinstance(value, str):
        raise TypeError("A vessel name must be a string")
    return " ".join(unicodedata.normalize("NFC", value).upper().split())


def has_hangul(value: str) -> bool:
    return bool(re.search("[가-힣]", value))


@dataclass(frozen=True)
class NameCandidate:
    name: str
    score: float
    vessel_ids: tuple[str, ...]


class CatalogNames:
    def __init__(self, records):
        by_name = {}
        self.name_by_id = {}
        for record in records:
            vessel_id = str(record["vessel_id"])
            name = canonical_name(record["vessel_name"])
            if not vessel_id or vessel_id in self.name_by_id:
                raise ValueError("Catalog IDs must be nonempty and unique")
            if not name:
                raise ValueError("Catalog names must be nonempty")
            self.name_by_id[vessel_id] = name
            by_name.setdefault(name, []).append(vessel_id)
        if not by_name:
            raise ValueError("Catalog must not be empty")
        # Sorting gives stable traversal and tie order without train/test frequency.
        self.ids_by_name = {name: tuple(sorted(ids)) for name, ids in sorted(by_name.items())}
        self.names = tuple(self.ids_by_name)


class ByteNameTrie:
    """ByT5: pad/start=0, EOS=1, UTF-8 bytes shifted by 3.

    EOS is a real branch: both 'ACE' and 'ACE ONE' can be completed.
    The entire official catalog is permitted; held-out answers never add branches.
    """
    def __init__(self, names):
        self.root = {}
        self.max_new_tokens = 0
        for name in names:
            tokens = [byte + 3 for byte in name.encode("utf-8")] + [1]
            self.max_new_tokens = max(self.max_new_tokens, len(tokens))
            node = self.root
            for token in tokens:
                node = node.setdefault(token, {})
        if not self.root:
            raise ValueError("Cannot build an empty catalog trie")

    def allowed(self, prefix):
        if not prefix or prefix[0] != 0:
            raise ValueError("Expected ByT5 decoder start token 0")
        node = self.root
        for token in prefix[1:]:
            if token not in node:
                raise ValueError("Generated prefix left the catalog trie")
            node = node[token]
            if token == 1:
                return [0]  # Finished beams may be padded by generate().
        return sorted(node)

    def __call__(self, batch_id, input_ids):
        return self.allowed(input_ids.tolist())
