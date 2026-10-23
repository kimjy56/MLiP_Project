"""Training-only non-Hangul noise. Only official train/catalog inputs."""
import hashlib
import random
import re

PROBABILITIES = {'name_typo': .365, 'ton_missing': .07, 'ton_shift': .37, 'flag_missing': .17}


def _rng(sample_id, kind, seed):
    raw = f'{seed}|{sample_id}|{kind}'.encode('utf-8')
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest()[:16], 'big'))


def noise_copy(row, catalog_row, seed=42):
    """Return (copied dict, audit flags). Caller passes target's official catalog row.

    Random streams are separate per field and stable by sample_id, independent
    of row order or fold. Missing tonnage overrides shift when both are drawn.
    Preserve original field noise when no new event is requested.
    """
    out = dict(row)
    if str(row['target_vessel_id']) != str(catalog_row['vessel_id']):
        raise ValueError('Catalog row must match the training target')
    sid = str(row['sample_id'])
    flags = {key: False for key in PROBABILITIES}
    flags.update(eligible=not bool(re.search('[가-힣]', str(row['vessel_name']))),
                 name_applied=False, ton_shift_applied=False)
    if not flags['eligible']:
        return out, flags
    for key, probability in PROBABILITIES.items():
        flags[key] = _rng(sid, key, seed).random() < probability
    out['sample_id'] = 'NOISE:' + sid
    if flags['name_typo']:
        rng = _rng(sid, 'name_operations', seed)
        original = str(out['vessel_name'])
        name = original
        for _ in range(rng.randint(1, 2)):
            matches = list(re.finditer(r'[A-Za-z]{6,}', name))
            if not matches:
                break
            match = rng.choice(matches)
            word = match.group()
            op = rng.choice(['delete', 'duplicate', 'transpose'])
            if op == 'transpose':
                positions = [i for i in range(2, len(word)-3) if word[i] != word[i+1]]
                if not positions:
                    continue
                pos = rng.choice(positions)
                changed = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            else:
                pos = rng.randrange(2, len(word)-2)
                changed = word[:pos] + (word[pos]*2 if op == 'duplicate' else '') + word[pos+1:]
            name = name[:match.start()] + changed + name[match.end():]
        out['vessel_name'] = name
        flags['name_applied'] = name != original
    if flags['ton_shift'] and not flags['ton_missing']:
        rng = _rng(sid, 'ton_magnitude', seed)
        factor = 1 + rng.choice([-1, 1]) * rng.uniform(.03, .05)
        # AR catalog has one canonical gross tonnage. Use it for both query fields.
        try:
            ton = float(catalog_row['gross_tonnage'])
            if ton > 0:
                out['gross_tonnage'] = out['international_gross_tonnage'] = ton * factor
                flags['ton_shift_applied'] = True
        except (ValueError, TypeError):
            pass
    if flags['ton_missing']:
        out['gross_tonnage'] = out['international_gross_tonnage'] = ''
    if flags['flag_missing']:
        out['flag_name'] = ''
    return out, flags
