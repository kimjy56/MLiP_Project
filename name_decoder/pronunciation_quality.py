"""Fixed, label-free quality and diversity policy for offline catalog pronunciations."""
import math
import re
from collections import Counter
from rapidfuzz.distance import OSA

POLICY_VERSION = 'retain-base-diverse-samples-v1'
DEFAULT_POLICY = {
    'seed':20261026,'batch_size':64,'samples_per_name':4,'max_new_tokens':32,
    'temperature':0.7,'top_p':0.9,'top_k':0,'max_variants':4,
    'anchor_min_raw_jamo_similarity':0.55,'relative_sample_logprob_margin':0.75,
    'preserve_original_variants':True,'require_eos':True,
    'require_matching_normalized_number_signature':True,
}


def clean_sample(text):
    text = text.strip().strip('\"\'`').strip()
    if not text or '\n' in text or '\r' in text:
        return None
    if not re.fullmatch(r'[\uac00-\ud7a30-9\s.\-#]+', text):
        return None
    text = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[\uac00-\ud7a3]', text):
        return None
    if re.search(r'([\uac00-\ud7a3])\1{5,}', text):
        return None
    return text


def select_variants(catalog_name, originals, samples, *, to_jamo, number_signature,
                    max_variants=4, min_anchor_similarity=0.55, logprob_margin=0.75):
    """Preserve original order; filter samples without labels, then add farthest-first.

    to_jamo returns a RAW (not phonetic-collapsed) Jamo signature with numbers
    separated by the caller. Generated log-probabilities are sampling-policy
    scores, not calibrated correctness probabilities or teacher-forced LM NLL.
    """
    if max_variants < len(originals):
        raise ValueError('Cannot truncate baseline variants')
    selected = list(originals)
    selected_jamo = [to_jamo(text) for text in selected]
    original_jamo = [value for value in selected_jamo if value]
    expected_numbers = number_signature(catalog_name)
    rejected = Counter()
    eligible = {}
    for item in samples:
        if not item.get('finished', False):
            rejected['no_eos'] += 1
            continue
        text = clean_sample(item.get('text', ''))
        if text is None:
            rejected['invalid_text'] += 1
            continue
        score = float(item.get('mean_logprob', float('-inf')))
        if not math.isfinite(score):
            rejected['nonfinite_logprob'] += 1
            continue
        if number_signature(text) != expected_numbers:
            rejected['number_signature_mismatch'] += 1
            continue
        jamo = to_jamo(text)
        if not jamo or jamo in selected_jamo:
            rejected['duplicate_or_empty_jamo'] += 1
            continue
        if original_jamo and max(OSA.normalized_similarity(jamo, anchor) for anchor in original_jamo) < min_anchor_similarity:
            rejected['anchor_dissimilar'] += 1
            continue
        prior = eligible.get(jamo)
        if prior is None or (-score, text) < (-prior['score'], prior['text']):
            eligible[jamo] = {'text': text, 'jamo': jamo, 'score': score}
    if eligible:
        threshold = max(item['score'] for item in eligible.values()) - logprob_margin
        pool = [item for item in eligible.values() if item['score'] >= threshold]
        rejected['relative_logprob'] += len(eligible) - len(pool)
        while pool and len(selected) < max_variants:
            def priority(item):
                distance = min((1.0 - OSA.normalized_similarity(item['jamo'], anchor)
                                for anchor in selected_jamo if anchor), default=1.0)
                return (-distance, -item['score'], item['text'])
            best = min(pool, key=priority)
            selected.append(best['text'])
            selected_jamo.append(best['jamo'])
            pool.remove(best)
    return selected, dict(rejected)

def schedule_names(catalog_names, baseline, scope):
    """Schedule by existing asset coverage only; never accepts validation labels."""
    names = sorted(set(catalog_names))
    if scope == 'all':
        return names
    if scope == 'deficient':
        return [name for name in names if len(baseline.get(name, [])) < 2]
    raise ValueError('Unknown catalog generation scope')
