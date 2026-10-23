"""Summarize completed checks and prepare an isolated AS handoff archive."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import zipfile
import pandas as pd
from run_checks import ROOT, sha, now


def main():
    output = ROOT/'artifacts/as02_handoff/results'
    record_path = ROOT/'project_log/experiments/AS02-HANDOFF.json'
    record = json.loads(record_path.read_text(encoding='utf-8'))
    assert record['status']=='completed'
    rows = pd.concat([pd.read_csv(output/f'fold_{k}.csv',dtype={'sample_id':str,'before':str,'after':str,'gold':str}) for k in range(5)])
    summary = {}
    for language, mask in [('hangul',rows.hangul),('non_hangul',~rows.hangul)]:
        x = rows[mask]
        summary[language] = {'rows':len(x),'before_accuracy':float((x.before==x.gold).mean()),
            'after_accuracy':float((x.after==x.gold).mean()),'changed':int((x.before!=x.after).sum()),
            'fixed':int(((x.before!=x.gold)&(x.after==x.gold)).sum()),
            'broken':int(((x.before==x.gold)&(x.after!=x.gold)).sum())}
    summary['before_weighted'] = .7*summary['hangul']['before_accuracy']+.3*summary['non_hangul']['before_accuracy']
    summary['after_weighted'] = .7*summary['hangul']['after_accuracy']+.3*summary['non_hangul']['after_accuracy']
    summary['regression_gate_passed'] = summary['hangul']['after_accuracy']-summary['hangul']['before_accuracy'] >= -.01
    assert summary['non_hangul']['changed']==0
    supplied = pd.read_csv(ROOT/'artifacts/as02_handoff/received/AR-02/oof_AR-02.csv',dtype=str)
    if {'sample_id','pred_vessel_id'}.issubset(supplied.columns):
        check = rows.merge(supplied[['sample_id','pred_vessel_id']],on='sample_id',validate='one_to_one')
        summary['received_oof_changed_rows'] = int((check.before!=check.pred_vessel_id).sum())
        summary['received_oof_matched_rows'] = len(check)
    for name in ['AR02-reproduced','AS-02']:
        command = [sys.executable,'-X','utf8',str(ROOT/'eval_kit/evaluate.py'),'--train',str(ROOT/'data/train.csv'),
            '--oof',str(output/f'oof_{name}.csv'),'--id',name,'--leaderboard',str(output/'leaderboard.csv')]
        if name == 'AS-02':
            command += ['--compare',str(output/'oof_AR02-reproduced.csv')]
        result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        (output/f'eval_{name}.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        assert result.returncode==0, result.stderr
    package = ROOT/'artifacts/as02_handoff/package'
    package.mkdir(exist_ok=False)
    (package/'model').mkdir()
    shutil.copy2(ROOT/'handoff/as02/script.py',package/'script.py')
    shutil.copy2(ROOT/'artifacts/as02_handoff/baseline/requirements.txt',package/'requirements.txt')
    model = json.loads((ROOT/'artifacts/as02_handoff/baseline/model/model.json').read_text(encoding='utf-8'))
    model['disable_korean_metadata'] = True
    # Preserve AR-02 lookup behavior for the one-factor AS-02 experiment.
    (package/'model/model.json').write_text(json.dumps(model,ensure_ascii=False),encoding='utf-8')
    if summary['regression_gate_passed']:
        archive = ROOT/'artifacts/as02_handoff/submit_AS-02.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(package.rglob('*')):
                if p.is_file():
                    z.write(p,p.relative_to(package).as_posix())
        summary['zip'] = {'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}
    record['summary'] = summary
    record['finished_at_kst'] = now()
    record['decision'] = 'G5 regression gate passed; local AS-02 package prepared, not uploaded' if summary['regression_gate_passed'] else 'G5 regression gate failed; no submission ZIP generated'
    record['artifacts'] += [{'path':str(p),'sha256':sha(p)} for p in output.glob('eval_*.txt')]
    record_path.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
