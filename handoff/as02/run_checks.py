"""Reproducible AS-02 group OOF and independent-noise aggregate audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone, timedelta
import itertools
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_noise import noise_copy, PROBABILITIES


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat()


def main():
    out = ROOT / 'artifacts/as02_handoff/results'
    out.mkdir(exist_ok=False)
    record_path = ROOT / 'project_log/experiments/AS02-HANDOFF.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    record.update(status='running', started_at_kst=now(), commands=[sys.executable, *sys.argv],
        environment={'python': platform.python_version(), 'platform': platform.platform()},
        code={'file_sha256': {str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*.py')}})
    def save():
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    save()
    try:
        train = pd.read_csv(ROOT/'data/train.csv', dtype={'sample_id': str, 'target_vessel_id': str})
        cat = pd.read_csv(ROOT/'data/vessel_catalog.csv', dtype={'vessel_id': str})
        folds = pd.read_csv(ROOT/'eval_kit/folds_group5_seed42.csv', dtype=str)
        assert folds.sample_id.is_unique and set(folds.sample_id) == set(train.sample_id)
        joined = train.merge(folds[['sample_id','target_vessel_id','fold']], on=['sample_id','target_vessel_id'], validate='one_to_one')
        assert len(joined) == len(train) and joined.groupby('target_vessel_id').fold.nunique().max() == 1
        record['data']['sha256'] = {str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'data/train.csv',ROOT/'data/vessel_catalog.csv',ROOT/'eval_kit/folds_group5_seed42.csv']}
        by_id = cat.set_index('vessel_id', drop=False).to_dict('index')
        flags = []
        for row in train.to_dict('records'):
            copied, audit = noise_copy(row, by_id[row['target_vessel_id']])
            if audit['eligible']:
                flags.append(audit)
                repeat, repeated = noise_copy(row, by_id[row['target_vessel_id']])
                assert repeated == audit and str(repeat) == str(copied)
                assert re_digits(copied['vessel_name']) == re_digits(row['vessel_name'])
        f = pd.DataFrame(flags)
        noise = {'rows':len(f), 'requested_probabilities':PROBABILITIES,
                 'observed_rates':{k:float(f[k].mean()) for k in PROBABILITIES},
                 'applied_rates':{k:float(f[k].mean()) for k in ['name_applied','ton_shift_applied']},
                 'conditional_rates':{}}
        for a,b in itertools.combinations(PROBABILITIES, 2):
            noise['conditional_rates'][a+' | '+b] = {str(v):float(f.loc[f[b]==v,a].mean()) for v in [False,True]}
        noise['limitations'] = ['Independent draws, not proof that resulting fields or names are statistically independent.',
            'Original train noise retained; final missingness differs from event probabilities.',
            'Short names may be ineligible for typo; missing tonnage overrides shift.']
        (out/'noise_audit.json').write_text(json.dumps(noise,indent=2),encoding='utf-8')
        base = load(ROOT/'artifacts/as02_handoff/baseline/script.py','ar02')
        variant = load(ROOT/'handoff/as02/script.py','as02')
        idx = base.CatalogIndex(cat)
        parts = []
        record['fold_results'] = []
        for k in range(5):
            tick = time.perf_counter()
            real, val = joined[joined.fold != str(k)], joined[joined.fold == str(k)]
            assert not set(real.target_vessel_id) & set(val.target_vessel_id)
            model = base.fit_matching_model(real.drop(columns='fold'),cat)
            query = val.drop(columns=['target_vessel_id','fold'])
            gold = val.target_vessel_id.tolist()
            before,ranks = base.predict_rows(query,model,idx,gold_ids=gold)
            default,_ = variant.predict_rows(query,model,idx,gold_ids=gold)
            assert before == default, 'Default behavior changed'
            after, ranks2 = variant.predict_rows(query,model,idx,gold_ids=gold,disable_korean_metadata=True)
            ko = query.vessel_name.map(base.is_korean_name).to_numpy()
            before,after,gold = np.array(before),np.array(after),np.array(gold)
            assert np.array_equal(before[~ko],after[~ko]), 'Latin predictions changed'
            change = {'fold':k, 'seconds':time.perf_counter()-tick, 'seen_vessel_rows':0}
            for label,mask in [('hangul',ko),('non_hangul',~ko)]:
                change[label] = {'rows':int(mask.sum()),'before_accuracy':float((before[mask]==gold[mask]).mean()),
                    'after_accuracy':float((after[mask]==gold[mask]).mean()),
                    'changed':int(((before!=after)&mask).sum()),
                    'fixed':int(((before!=gold)&(after==gold)&mask).sum()),
                    'broken':int(((before==gold)&(after!=gold)&mask).sum()),
                    'candidate_coverage':float((np.asarray(ranks)[mask]>0).mean())}
            frame = pd.DataFrame({'sample_id':val.sample_id,'fold':k,'before':before,'after':after,'gold':gold,'hangul':ko})
            frame.to_csv(out/f'fold_{k}.csv',index=False)
            (out/f'model_fold_{k}.json').write_text(json.dumps(model,ensure_ascii=False),encoding='utf-8')
            parts.append(frame)
            record['fold_results'].append(change)
            save()
            print(json.dumps(change),flush=True)
        all_rows = pd.concat(parts)
        for key,name in [('before','AR02-reproduced'),('after','AS-02')]:
            all_rows[['sample_id',key,'fold']].rename(columns={key:'pred_vessel_id'}).to_csv(out/f'oof_{name}.csv',index=False)
        record.update(status='completed',finished_at_kst=now(),artifacts=[{'path':str(p),'sha256':sha(p)} for p in out.iterdir() if p.is_file()])
        save()
    except BaseException as exc:
        record.update(status='failed',failure={'type':type(exc).__name__,'message':str(exc)},finished_at_kst=now())
        save()
        raise


def re_digits(value):
    import re
    return re.findall(r'\d+',str(value))


if __name__ == '__main__':
    main()
