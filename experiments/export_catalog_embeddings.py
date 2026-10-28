"""Export FP16 catalog embeddings with checkpoint, catalog-order and fold provenance."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phonetic_encoder.model import CharPhoneticBiEncoder
from experiments.dpr03_common import start_run, finish_run, write_json, sha256, object_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-id', default='EXPORT-DPR-SYMPHON-03')
    p.add_argument('--checkpoint', default='artifacts/DPR-SYMPHON-03/fold_0_encoder.pt')
    p.add_argument('--catalog-csv', default='data/vessel_catalog.csv')
    p.add_argument('--output-npy', default='artifacts/DPR-SYMPHON-03/catalog_embeds_fp16.npy')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    output = Path(args.output_npy)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError('Never overwrite existing embeddings/metadata')
    out, record_path, record = start_run(args.experiment_id)
    started = time.perf_counter()
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=6)
        torch.set_num_threads(6)
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
        model = CharPhoneticBiEncoder.load_checkpoint(args.checkpoint, device=device).float().eval()
        catalog = pd.read_csv(args.catalog_csv, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        ids, names = catalog.vessel_id.tolist(), catalog.vessel_name.tolist()
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate catalog IDs')
        if device == 'cuda': torch.cuda.synchronize()
        tick = time.perf_counter()
        embeds = model.encode_catalog(names, batch_size=args.batch_size).astype(np.float16)
        if device == 'cuda': torch.cuda.synchronize()
        encoding_seconds = time.perf_counter()-tick
        if embeds.shape != (len(ids), 128) or not np.isfinite(embeds).all():
            raise ValueError('Invalid embeddings')
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name+'.tmp')
        with temporary.open('wb') as stream: np.save(stream, embeds)
        temporary.replace(output)
        import json
        checkpoint_manifest = Path(args.checkpoint).with_suffix('.manifest.json')
        lineage = json.loads(checkpoint_manifest.read_text(encoding='utf-8'))
        if lineage['checkpoint_sha256'] != sha256(args.checkpoint):
            raise ValueError('Checkpoint does not match training manifest')
        data = lineage['data']
        metadata = {'num_vessels':len(ids), 'embed_dim':128, 'dtype':'float16',
                    'file_size_bytes':output.stat().st_size, 'file_sha256':sha256(output),
                    'embeddings_sha256':sha256(output), 'checkpoint_sha256':sha256(args.checkpoint),
                    'catalog_sha256':sha256(args.catalog_csv), 'catalog_ids_sha256':object_hash(ids),
                    'folds_sha256':data['split_file_sha256'],
                    'fit_vessel_ids_sha256':data['fit_vessel_ids_sha256'],
                    'heldout_fold':lineage['config']['fold'], 'checkpoint_selection':lineage['selection'],
                    'code':record['code'], 'encoding_seconds':encoding_seconds,
                    'device':device, 'batch_size':args.batch_size}
        write_json(output.with_suffix('.json'), metadata)
        record.update(data=data, artifacts=[str(output),str(output.with_suffix('.json'))], export=metadata)
        record['metrics']['metric_source']='catalog embedding export only; no accuracy evaluation'
        record['cost'].update(inference_seconds=time.perf_counter()-started, assets_bytes=output.stat().st_size,
                              timing_scope='model/catalog load, encoding, FP16 conversion, hashing and save')
        finish_run(record_path, record, 'exported_unvalidated')
        print('Exported:',output,embeds.shape,encoding_seconds,flush=True)
    except BaseException as exc:
        finish_run(record_path, record, 'failed', exc)
        raise


if __name__ == '__main__':
    main()
