"""Fetch official LFM2.5 checkpoint at a resolved immutable revision."""
import os,json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_OFFLINE']='0'
from huggingface_hub import model_info,snapshot_download
ROOT=Path(__file__).resolve().parents[1]
def main():
    repo='LiquidAI/LFM2.5-350M'
    dest=ROOT/'artifacts/pretrained/lfm2.5-350m'
    manifest=dest/'download_manifest.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf-8'))
        revision=old['revision']
    else:
        revision=model_info(repo,token=False).sha
    snapshot_download(repo,revision=revision,token=False,local_dir=dest,
        cache_dir=ROOT/'.cache/huggingface',allow_patterns=['*.json','*.safetensors','*.jinja','LICENSE*','README.md'],max_workers=2)
    files={}
    for p in sorted(dest.iterdir()):
        if p.is_file() and p.name!='download_manifest.json':
            with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files[p.name]={'sha256':digest,'bytes':p.stat().st_size}
    if 'model.safetensors' not in files:raise FileNotFoundError('Weights missing')
    manifest.write_text(json.dumps({'repo_id':repo,'revision':revision,'files':files,
        'recorded_at_kst':datetime.now(timezone(timedelta(hours=9))).isoformat()},indent=2)+'\n',encoding='utf-8')
    print('Downloaded and hashed',repo,revision)
if __name__=='__main__':main()
