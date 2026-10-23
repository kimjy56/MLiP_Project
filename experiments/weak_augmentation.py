"""Fold-local, label-preserving-intent perturbations; no test inputs or external data."""
from collections import Counter, defaultdict
import hashlib
import re
import numpy as np
import pandas as pd


def normalized(value):
    return ''.join(c for c in str(value).upper() if c.isalnum())


def augment(training, catalog, seed=42, fraction=0.5):
    """One possible mutation per selected unique real row; numeric characters preserved.

    Half of attempts delete one letter from a name with >=5 alphabetic letters.
    Other attempts mask one nonempty metadata field. Known cross-ID name collisions
    are rejected. All probabilities are fixed hypotheses, not fitted to validation.
    """
    columns = ['vessel_name', 'gross_tonnage', 'international_gross_tonnage',
               'flag_name', 'vessel_type', 'target_vessel_id']
    unique = training.drop_duplicates(columns)
    owners = defaultdict(set)
    for name, vid in zip(catalog.vessel_name, catalog.vessel_id):
        owners[normalized(name)].add(vid)
    for name, vid in zip(training.vessel_name, training.target_vessel_id):
        owners[normalized(name)].add(vid)
    rng = np.random.default_rng(seed)
    result, counts = [], Counter()
    for source in unique.to_dict('records'):
        if rng.random() >= fraction:
            continue
        row = dict(source)
        if rng.random() < 0.5:
            name = str(row['vessel_name'])
            positions = [i for i, c in enumerate(name) if c.isalpha()]
            if len(positions) < 5:
                counts['rejected_short_name'] += 1
                continue
            p = int(rng.choice(positions))
            changed = name[:p] + name[p + 1:]
            if re.findall(r'\d+', name) != re.findall(r'\d+', changed):
                counts['rejected_number_boundary'] += 1
                continue
            if owners[normalized(changed)] - {row['target_vessel_id']}:
                counts['rejected_known_collision'] += 1
                continue
            row['vessel_name'] = changed
            operation = 'delete_one_letter'
        else:
            choices = [c for c in columns[1:5] if str(row[c]).strip()]
            if not choices:
                counts['rejected_no_metadata'] += 1
                continue
            column = str(rng.choice(choices))
            row[column] = ''
            operation = 'mask_' + column
        row['sample_id'] = 'AUG:' + str(row['sample_id'])
        row['call_sign'] = ''
        counts[operation] += 1
        result.append(row)
    frame = pd.DataFrame(result, columns=training.columns)
    # No synthetic rows enter the lookup dictionaries, which are fit on real rows.
    assert set(frame.target_vessel_id).issubset(set(training.target_vessel_id))
    digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    return frame, {'seed': seed, 'selection_fraction': fraction, 'rows': len(frame),
                   'unique_source_rows': len(unique), 'operations': dict(counts),
                   'synthetic_sha256': digest}
