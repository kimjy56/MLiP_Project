"""Official subset evaluation for clean fold0 rank-LM diagnostics."""
import sys,json,subprocess,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.noisy_channel_common import write_json,sha256
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-id',default='EVAL-LM-RANK-01')
parser.add_argument('--compare-lm',type=Path)
args=parser.parse_args()
OUT=ROOT/'reports/noisy_channel'/args.experiment_id
for mode in ['baseline','lm','fusion','gated']:
    command=[sys.executable,'-X','utf8',str(ROOT/'eval_kit/evaluate.py'),'--oof',str(OUT/f'predictions_{mode}.csv'),'--train',str(ROOT/'data/train.csv'),'--subset','--id',f'{args.experiment_id}-{mode}']
    if mode!='baseline':command+=['--compare',str(OUT/'predictions_baseline.csv')]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,encoding='utf-8')
    (OUT/f'official_{mode}.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(f'Official evaluation failed: {mode}; see saved output')
    print(f'official {mode} passed',flush=True)
record_path=ROOT/'project_log/experiments'/f'{args.experiment_id}.json'
record=json.loads(record_path.read_text(encoding='utf-8'))
record['official_evaluator']={'script_sha256':sha256(ROOT/'eval_kit/evaluate.py'),'scope':'--subset clean full fold0; not full five-fold OOF','bootstrap':2000,'seed':0,'outputs':{f'official_{mode}.txt':sha256(OUT/f'official_{mode}.txt') for mode in ['baseline','lm','fusion','gated']}}
if args.compare_lm:
    command=[sys.executable,'-X','utf8',str(ROOT/'eval_kit/evaluate.py'),'--oof',str(OUT/'predictions_lm.csv'),'--train',str(ROOT/'data/train.csv'),'--subset','--compare',str(args.compare_lm)]
    result=subprocess.run(command,cwd=ROOT,capture_output=True,encoding='utf-8')
    target=OUT/'official_lm_vs_previous.txt'
    target.write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Matched LM comparison failed')
    record['official_evaluator']['matched_lm_comparison']={'source':str(args.compare_lm),'source_sha256':sha256(args.compare_lm),'output_sha256':sha256(target)}
    print('official matched LM comparison passed',flush=True)
write_json(record_path,record);write_json(OUT/'run_record.json',record)
