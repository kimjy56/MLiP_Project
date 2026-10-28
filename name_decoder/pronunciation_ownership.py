"""Label-free reverse retrieval guard against ambiguous optional pronunciations."""
from collections import defaultdict
from rapidfuzz import process
from rapidfuzz.distance import OSA


class PronunciationOwnershipGuard:
    def __init__(self, baseline, *, to_jamo, number_signature):
        self.to_jamo = to_jamo
        self.number_signature = number_signature
        self.groups = defaultdict(lambda: {'choices': [], 'owners': []})
        self.anchors = defaultdict(list)
        max_count = 0
        for owner, variants in baseline.items():
            seen = set()
            counts = defaultdict(int)
            for text in variants:
                signature = tuple(number_signature(text))
                jamo = to_jamo(text)
                key = (signature, jamo)
                if not jamo or key in seen:
                    continue
                seen.add(key)
                group = self.groups[signature]
                group['choices'].append(jamo)
                group['owners'].append(owner)
                self.anchors[(owner, signature)].append(jamo)
                counts[signature] += 1
            max_count = max(max_count, max(counts.values(), default=0))
        # At most max_count hits can belong to this query's original owner.
        # Taking one more hit proves whether ANY competing owner beats/ties it.
        self.hit_limit = max_count + 1

    def check(self, owner, text):
        signature = tuple(self.number_signature(text))
        jamo = self.to_jamo(text)
        own = self.anchors.get((owner, signature), [])
        if not jamo or not own:
            return False, 'no_original_anchor'
        own_best = max(OSA.normalized_similarity(jamo, anchor) for anchor in own)
        group = self.groups[signature]
        hits = process.extract(jamo, group['choices'], scorer=OSA.normalized_similarity,
                               score_cutoff=max(0.0, own_best - 1e-12), limit=self.hit_limit)
        competing = [(score, group['owners'][position]) for _, score, position in hits
                     if group['owners'][position] != owner]
        if any(score >= own_best - 1e-12 for score, _ in competing):
            return False, 'competing_owner_beats_or_ties'
        return True, 'unique_closest_original_owner'