"""Catalog-only lookup for generated names; stable ties and selectable scoring."""
from rapidfuzz import fuzz, process

SCORERS = {"ratio": fuzz.ratio, "WRatio": fuzz.WRatio}

class CatalogNameSearch:
    def __init__(self, catalog_names, scorer="ratio"):
        if scorer not in SCORERS:
            raise ValueError(f"Unknown catalog scorer: {scorer}")
        self.names = tuple(sorted(set(catalog_names)))
        if not self.names:
            raise ValueError("Catalog must not be empty")
        self.scorer = scorer

    def top_names(self, generated_text, limit=10):
        if not 1 <= limit <= 10:
            raise ValueError("Candidate limit must be between 1 and 10")
        # Input is already cleaned by generation; default ratio uses full strings.
        return [name for name, _, _ in process.extract(
            generated_text, self.names, scorer=SCORERS[self.scorer], limit=limit)]

    def top_matches(self, generated_text, limit=10):
        if not 1 <= limit <= 10:
            raise ValueError("Candidate limit must be between 1 and 10")
        return [(name, float(score)) for name, score, _ in process.extract(
            generated_text, self.names, scorer=SCORERS[self.scorer], limit=limit)]

