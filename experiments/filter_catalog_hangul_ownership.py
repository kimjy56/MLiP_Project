"""Filter optional new KO variants by label-free nearest original pronunciation."""
import argparse
import re
from collections import Counter
from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.noisy_channel_common import configure_cpu,reserve_experiment,write_json,record_failure,sha256,now_kst
from experiments.evaluate_jamo_scoring import load_engine
from name_decoder.pronunciation_ownership import PronunciationOwnershipGuard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id',default='KO-OWNERSHIP-01')
    parser.add_argument('--source-experiment-id',default='KO-DIVERSE-01')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',args.source_experiment_id):
        parser.error('Invalid source experiment ID')
    configure_cpu(2)
    out,path,record=reserve_experiment(args.experiment_id,ROOT/'artifacts',
        'Apply nearest-original-pronunciation ownership guard to optional additions',' '.join(sys.argv))
    started=time.perf_counter()
    try:
        record['source_experiment_id']=args.source_experiment_id
        record['policy']=record.get('policy') or {'owner_similarity':'normalized Jamo OSA','number_group':'same engine-normalized number signature','acceptance':'own best similarity strictly greater than every other baseline name; reject ties','scope':'new additions only','training_labels_used':False,'fitted_thresholds':False}
        source=ROOT/'artifacts'/args.source_experiment_id/'catalog_full_ko_map.json'
        base=ROOT/'AR-06/catalog_full_ko_map.json'
        generation=json.loads((ROOT/'project_log/experiments'/(args.source_experiment_id+'.json')).read_text(encoding='utf-8'))
        if sha256(source)!=generation['artifact_sha256']['catalog_full_ko_map.json']:
            raise ValueError('Input map hash mismatch')
        baseline=json.loads(base.read_text(encoding='utf-8-sig'))
        expanded=json.loads(source.read_text(encoding='utf-8'))
        expected=json.loads((ROOT/'artifacts'/args.source_experiment_id/'generation_config.json').read_text(encoding='utf-8'))['base_map_sha256']
        if sha256(base)!=expected:raise ValueError('Original map changed')
        engine=load_engine(ROOT/'artifacts/AS-06/script.py','_pronunciation_ownership_normalizer')
        def number_signature(text):return tuple(engine.normalize_vessel_name(text,kr_mode='AR06')[2])
        def to_jamo(text):
            words=engine.normalize_vessel_name(text,kr_mode='AR06')[1]
            return ''.join(engine.hangul_to_jamo(word,phonetic_norm=True) for word in words)
        guard=PronunciationOwnershipGuard(baseline,to_jamo=to_jamo,number_signature=number_signature)
        mapping={name:list(values) for name,values in baseline.items()}
        counts=Counter();decisions=[]
        candidates=[(name,text) for name,values in expanded.items() for text in values[len(baseline.get(name,[])):]]
        print(f'Checking {len(candidates)} optional variants; original-owner hit limit={guard.hit_limit}',flush=True)
        for i,(name,text) in enumerate(candidates):
            accepted,reason=guard.check(name,text)
            counts[reason]+=1
            if accepted:mapping.setdefault(name,[]).append(text)
            decisions.append({'catalog_name':name,'new_variant':text,'accepted':accepted,'reason':reason})
            if (i+1)%500==0 or i+1==len(candidates):
                print(f'  {i+1}/{len(candidates)} checked ({time.perf_counter()-started:.1f}s)',flush=True)
        if any(mapping.get(n,[])[:len(v)]!=v for n,v in baseline.items()):raise AssertionError('Baseline variants changed')
        write_json(out/'catalog_full_ko_map.json',mapping)
        write_json(out/'decisions.json',decisions)
        manifest={'source_map_sha256':sha256(source),'base_map_sha256':sha256(base),
            'source_generation_config_sha256':sha256(ROOT/'artifacts'/args.source_experiment_id/'generation_config.json'),
            'policy':record['policy'],'source_sha256':{p:sha256(ROOT/p) for p in [
                'name_decoder/pronunciation_ownership.py','experiments/filter_catalog_hangul_ownership.py','artifacts/AS-06/script.py']}}
        write_json(out/'generation_config.json',manifest)
        stats={'processed_scope_complete':True,'all_original_variants_preserved':True,
            'checked_additions':len(candidates),'accepted_additions':sum(d['accepted'] for d in decisions),
            'removed_additions':sum(not d['accepted'] for d in decisions),'reasons':dict(counts),
            'new_map_entries':len(mapping),'names_with_added_variants':sum(len(v)>len(baseline.get(n,[])) for n,v in mapping.items()),
            'variant_count_histogram':dict(Counter(map(len,mapping.values())))}
        write_json(out/'generation_summary.json',stats)
        record.update(status='generated_unvalidated',finished_at_kst=now_kst(),diagnostics=stats,artifacts=[str(out)],
            artifact_sha256={'catalog_full_ko_map.json':sha256(out/'catalog_full_ko_map.json'),'generation_config.json':sha256(out/'generation_config.json')},
            decision='Postprocessing only; compare fixed-engine accuracy before considering use.',
            limitations=['Checks against baseline pronunciations, which may themselves be wrong or ambiguous.',
                'The guard is label-free but was proposed after aggregate development results; not independent confirmation.'],
            cost={'filtering_seconds':time.perf_counter()-started,'timing_scope':'load/index/global OSA ownership checks/save; no neural training'})
        record['code']['file_sha256'].update(manifest['source_sha256'])
        write_json(path,record)
        print(json.dumps(stats),flush=True)
    except BaseException as exc:
        record_failure(path,record,exc);raise


if __name__=='__main__':main()