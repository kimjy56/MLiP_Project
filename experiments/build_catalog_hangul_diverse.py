"""Generate quality-filtered diverse pronunciations for the full official catalog.

Preserves the original AS-06 map and frozen generator; never trains or builds ZIPs.
Batch artifacts bind seeds, source files, model weights, policy and output hashes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.noisy_channel_common import (
    configure_cpu, reserve_experiment, write_json, record_failure, sha256, now_kst,
)
from experiments.evaluate_jamo_scoring import load_engine
from name_decoder.pronunciation_quality import POLICY_VERSION, DEFAULT_POLICY, select_variants, schedule_names


def batch_seed(names, seed):
    content = json.dumps([seed, names], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(content).digest()[:8], 'big') % (2**31)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--scope', choices=['all','deficient'], default='all')
    parser.add_argument('--limit-batches', type=int)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error('limit-batches must be positive')
    configure_cpu(2)
    out, path, record = reserve_experiment(args.experiment_id, ROOT / 'artifacts',
        'Quality-controlled offline diverse catalog pronunciation generation', ' '.join(sys.argv))
    started = time.perf_counter()
    try:
        import torch
        import pandas as pd
        from transformers import AutoModelForCausalLM, LogitsProcessorList
        from name_decoder.catalog import canonical_name
        from name_decoder.noisy_channel import load_channel_tokenizer
        from experiments.build_catalog_full_hangul import HangulOnlyLogitsProcessor, encode_ko_generation_prompt
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for this full-catalog generation experiment')
        audit = json.loads((ROOT / 'artifacts/AUDIT-GENERATOR-LINEAGE-01/audit.json').read_text(encoding='utf-8'))
        if not audit['all_checks_passed']:
            raise ValueError('Recorded generator lineage audit failed')
        policy = dict(DEFAULT_POLICY)
        policy.update(record.get('policy') or {})
        record['policy'] = policy
        engine_path = ROOT / 'artifacts/AS-06/script.py'
        engine = load_engine(engine_path, '_pronunciation_quality_normalizer')
        def to_jamo(text):
            _, words, _ = engine.normalize_vessel_name(text, kr_mode='AR06')
            return ''.join(engine.hangul_to_jamo(word, phonetic_norm=False) for word in words)
        def number_signature(text):
            return tuple(engine.normalize_vessel_name(text, kr_mode='AR06')[2])
        model_dir = ROOT / 'artifacts/LM-LFM25-FULL-R24-01/model'
        model_manifest = json.loads((model_dir / 'noisy_channel_manifest.json').read_text(encoding='utf-8'))
        for filename, digest in model_manifest['checkpoint_sha256'].items():
            if sha256(model_dir / filename) != digest:
                raise ValueError('Model hash mismatch: ' + filename)
        baseline_path = ROOT / 'AR-06/catalog_full_ko_map.json'
        baseline_hash = sha256(baseline_path)
        expected = json.loads((ROOT / 'project_log/experiments/AS-06.json').read_text(encoding='utf-8-sig'))['code']['file_sha256']['AR-06/catalog_full_ko_map.json']
        if baseline_hash != expected.lower():
            raise ValueError('Original map hash mismatch')
        mapping = json.loads(baseline_path.read_text(encoding='utf-8-sig'))
        baseline = {name: list(values) for name, values in mapping.items()}
        frame = pd.read_csv(ROOT / 'data/vessel_catalog.csv', keep_default_na=False)
        all_names = sorted({canonical_name(str(name)) for name in frame.vessel_name if canonical_name(str(name))})
        names = schedule_names(all_names,baseline,args.scope)
        code_paths = ['experiments/build_catalog_hangul_diverse.py','name_decoder/pronunciation_quality.py',
            'experiments/build_catalog_full_hangul.py','name_decoder/noisy_channel.py','artifacts/AS-06/script.py']
        source_hashes = {p: sha256(ROOT / p) for p in code_paths}
        config = {'scope':args.scope,'policy': policy, 'policy_version': POLICY_VERSION, 'model_sha256': model_manifest['checkpoint_sha256'],
            'model_manifest_sha256':sha256(model_dir / 'noisy_channel_manifest.json'), 'base_map_sha256':baseline_hash,
            'catalog_sha256':sha256(ROOT / 'data/vessel_catalog.csv'), 'ordered_names_sha256':hashlib.sha256(json.dumps(names,ensure_ascii=False,separators=(',', ':')).encode('utf-8')).hexdigest(),
            'source_sha256':source_hashes, 'prompt_version':model_manifest['prompt_version']}
        write_json(out / 'generation_config.json', config)
        batches = out / 'batches'; batches.mkdir()
        if args.resume_from:
            old_config = json.loads((args.resume_from / 'generation_config.json').read_text(encoding='utf-8'))
            if old_config != config:
                raise ValueError('Resume config differs: do not mix models/policies/maps')
            for old in sorted((args.resume_from / 'batches').glob('*.json')):
                shutil.copy2(old, batches / old.name)
        record['code']['file_sha256'].update(source_hashes)
        record['data'].update(split_id='generator excludes fold0; full public catalog inference',
            source_files_sha256={'catalog':config['catalog_sha256'],'base_map':baseline_hash}, seed=policy['seed'])
        record['model'] = {'type':'frozen LFM2.5-350M; offline inference only','parameters':policy,'loss':None,
            'training_experiment':'LM-LFM25-FULL-R24-01','heldout_fold':0}
        record['environment'].update(device='cuda',gpu_name=torch.cuda.get_device_name(0),cuda=torch.version.cuda,threads=2)
        record['generation_config_sha256'] = sha256(out / 'generation_config.json')
        record['resume_from'] = str(args.resume_from) if args.resume_from else None
        write_json(path, record)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        tokenizer = load_channel_tokenizer(model_dir)
        eos = tokenizer.eos_token_id
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        model = AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16).to('cuda').eval()
        processors = LogitsProcessorList([HangulOnlyLogitsProcessor(tokenizer,eos)])
        totals = Counter(); rejection_counts = Counter(); completed = 0
        n_batches = (len(names) + policy['batch_size'] - 1) // policy['batch_size']
        planned_batches = min(n_batches,args.limit_batches) if args.limit_batches else n_batches
        inference_start = time.perf_counter()
        print(f'CUDA {record["environment"]["gpu_name"]}; names={len(names)}; batches={planned_batches}; source={model_manifest["experiment_id"]}',flush=True)
        for batch_i in range(planned_batches):
            start = batch_i * policy['batch_size']
            batch_names = names[start:start + policy['batch_size']]
            result_path = batches / f'{batch_i:05d}.json'
            if result_path.exists():
                batch = json.loads(result_path.read_text(encoding='utf-8'))
                if batch['names'] != batch_names or batch['seed'] != batch_seed(batch_names,policy['seed']):
                    raise ValueError('Resumed batch schedule differs')
                totals['resumed_batches'] += 1
            else:
                seed = batch_seed(batch_names,policy['seed'])
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                prompts = [encode_ko_generation_prompt(tokenizer,name,model_manifest['prompt_version']) for name in batch_names]
                width = max(map(len,prompts))
                ids = torch.full((len(prompts),width),pad,dtype=torch.long,device='cuda')
                mask = torch.zeros_like(ids)
                for i,prompt in enumerate(prompts):
                    ids[i,-len(prompt):] = torch.tensor(prompt,dtype=torch.long,device='cuda')
                    mask[i,-len(prompt):] = 1
                with torch.inference_mode():
                    generated = model.generate(input_ids=ids,attention_mask=mask,
                        do_sample=True,num_beams=1,num_return_sequences=policy['samples_per_name'],
                        temperature=policy['temperature'],top_p=policy['top_p'],top_k=policy['top_k'],
                        max_new_tokens=policy['max_new_tokens'],min_new_tokens=1,use_cache=True,
                        pad_token_id=pad,eos_token_id=eos,logits_processor=processors,
                        return_dict_in_generate=True,output_scores=True)
                    tokens = generated.sequences[:,width:]
                    score_sum = torch.zeros(len(tokens),device='cuda')
                    count = torch.zeros(len(tokens),device='cuda')
                    alive = torch.ones(len(tokens),dtype=torch.bool,device='cuda')
                    for step,scores in enumerate(generated.scores):
                        logp = scores.float().log_softmax(-1).gather(1,tokens[:,step,None]).squeeze(1)
                        score_sum += torch.where(alive,logp,0.0);count += alive.float()
                        alive &= tokens[:,step].ne(eos)
                    averages = (score_sum/count.clamp_min(1)).cpu().tolist()
                    finished = tokens.eq(eos).any(-1).cpu().tolist()
                    decoded = tokenizer.batch_decode(tokens.cpu(),skip_special_tokens=True)
                batch = {'names':batch_names,'seed':seed,'outputs':{},'rejections':{},'generated_samples':len(decoded)}
                rejected = Counter()
                for i,name in enumerate(batch_names):
                    k = policy['samples_per_name']
                    samples = [{'text':decoded[j], 'finished':finished[j], 'mean_logprob':averages[j]}
                        for j in range(i*k,(i+1)*k)]
                    variants,reasons = select_variants(name,baseline.get(name,[]),samples,to_jamo=to_jamo,
                        number_signature=number_signature,max_variants=policy['max_variants'],
                        min_anchor_similarity=policy['anchor_min_raw_jamo_similarity'],
                        logprob_margin=policy['relative_sample_logprob_margin'])
                    batch['outputs'][name] = variants
                    rejected.update(reasons)
                batch['rejections'] = dict(rejected)
                write_json(result_path,batch)
                del generated,tokens,ids,mask,scores,score_sum,count,alive
            for name,variants in batch['outputs'].items():
                if variants:mapping[name] = variants
            rejection_counts.update(batch['rejections'])
            totals['generated_samples'] += batch['generated_samples']
            completed += len(batch_names)
            if batch_i % 10 == 0 or batch_i + 1 == planned_batches:
                elapsed = time.perf_counter()-inference_start
                print(f'  {batch_i+1}/{planned_batches} batches, {completed} names, {elapsed:.1f}s',flush=True)
        if any(mapping[name][:len(values)] != values for name,values in baseline.items()):
            raise AssertionError('Baseline variants were changed')
        if sha256(baseline_path) != baseline_hash:
            raise AssertionError('Original map was modified')
        write_json(out / 'catalog_full_ko_map.json',mapping)
        counts = Counter(map(len,mapping.values()))
        diagnostics = {'catalog_unique_names':len(all_names),'scheduled_names':len(names),'scope':args.scope,
            'processed_names':completed,'processed_scope_complete':completed==len(names),
            'processed_all':args.scope=='all' and completed==len(all_names),
            'original_map_entries':len(baseline),'new_map_entries':len(mapping),
            'names_with_added_variants':sum(len(mapping.get(n,[]))>len(baseline.get(n,[])) for n in names),
            'added_variants':sum(len(mapping.get(n,[]))-len(baseline.get(n,[])) for n in names),
            'missing_names':len(set(all_names)-set(mapping)), 'variant_count_histogram':dict(counts),
            'rejection_counts':dict(rejection_counts),'all_original_variants_preserved':True, **dict(totals)}
        write_json(out / 'generation_summary.json',diagnostics)
        record.update(status='generated_unvalidated' if diagnostics['processed_scope_complete'] else 'pilot_generated_unvalidated',
            finished_at_kst=now_kst(),diagnostics=diagnostics,artifacts=[str(out)],
            artifact_sha256={'catalog_full_ko_map.json':sha256(out / 'catalog_full_ko_map.json'),
                'generation_config.json':sha256(out / 'generation_config.json')},
            cost={'total_seconds':time.perf_counter()-started,'inference_seconds':time.perf_counter()-inference_start,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                'timing_scope':'model/hash/load plus sampling/quality/batch serialization; no neural training or submission inference'},
            limitations=['Sampling-policy logprob is not a correctness probability or teacher-forced NLL.',
                'Fixed seed/config/hardware is recorded; cross-hardware bitwise identity is not guaranteed.',
                'Frozen model excludes fold0 but includes folds1-4; no full5fold validation claim.',
                'No accuracy result until separate matching comparison.'],
            decision='Generation only; preserve AS-06 until fixed-engine matching comparison.',next_actions=['Evaluate original AS-06 scoring with this map under pre-registered conditions.'])
        write_json(path,record)
        print(json.dumps(diagnostics),flush=True)
    except BaseException as exc:
        record_failure(path,record,exc)
        raise


if __name__ == '__main__':
    main()