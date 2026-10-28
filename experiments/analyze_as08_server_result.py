"""Read-only AS06/07/08 archive audit; import-only probes, no fitting/inference/ZIP creation."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def value_summary(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return {"type": type(value).__name__, "length": len(value) if isinstance(value, (dict,list,str)) else None,
            "sha256": sha_bytes(data)}


def code_summary(content):
    tree = ast.parse(content)
    functions, constants = {}, {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            functions[node.name] = sha_bytes(ast.dump(node, include_attributes=False).encode())
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, (str,int,float,bool,type(None))):
                        constants[target.id] = value
                    elif isinstance(value, (dict,list,tuple)):
                        constants[target.id] = {"length":len(value),"ast_sha256":sha_bytes(ast.dump(node.value).encode())}
    return {"definitions": functions, "constants":constants}


PROBE = r"""
import importlib, json, runpy, socket, sys
from pathlib import Path
bundle = Path(sys.argv[1])
socket.socket.connect = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Network disabled"))
socket.create_connection = socket.socket.connect
import torch
loads=[]
def blocked_load(*a, **kw):
    loads.append(1)
    raise AssertionError("Model loading not permitted in import-only probe")
torch.load = blocked_load
result={}
for p in (bundle,bundle/"model"):
    sys.path.insert(0,str(p))
try:
    importlib.import_module("model.phonetic_encoder.hybrid_matcher")
    result["direct_package_import"]={"passed":True}
except Exception as exc:
    result["direct_package_import"]={"passed":False,"type":type(exc).__name__,"message":str(exc)}
loaded = runpy.run_path(str(bundle/"script.py"),run_name="_archive_import_audit_")
result["entrypoint_dense_class_available"] = loaded.get("HybridDenseLexicalMatcher") is not None
result["loaded_module_path"] = None
if result["entrypoint_dense_class_available"]:
    result["loaded_module_path"] = loaded["HybridDenseLexicalMatcher"].__module__
result["model_load_calls"] = len(loads)
result["repository_path_available"] = any(Path(p).resolve() == Path(sys.argv[2]).resolve() for p in sys.path)
print(json.dumps(result))
"""


def isolated_probe(archive, source_overrides=None):
    temp_root = (ROOT / "scratch/ANALYSIS-AS08-SERVER-01/import_probes").resolve()
    assert temp_root.is_relative_to(ROOT.resolve())
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vessel-archive-import-", dir=temp_root) as temp:
        base = Path(temp).resolve()
        assert base.is_relative_to(temp_root)
        bundle = base / "bundle"
        # Only audited .py sources are materialized, never weights/data/ZIP.
        for name in archive.namelist():
            if name.endswith(".py"):
                parts = Path(name).parts
                if Path(name).is_absolute() or ".." in parts:
                    raise ValueError("Unsafe archive member")
                target = bundle / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        for name, content in (source_overrides or {}).items():
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        result = subprocess.run([sys.executable,"-I","-B","-X","utf8","-c",PROBE,str(bundle),str(ROOT)],
                                cwd=base, capture_output=True,text=True,timeout=60,check=True)
        return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/as08_server_analysis/archive_audit.json")
    args=parser.parse_args()
    models, sources, archives = {}, {}, {}
    report={"recorded_at_kst":datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
            "python":sys.version,"executable":sys.executable,"packages":{}}
    for number in ("06","07","08"):
        filename=f"submit_AS-{number}.zip"
        path=ROOT/filename
        before=sha_file(path)
        with zipfile.ZipFile(path) as archive:
            names=archive.namelist()
            source=archive.read("script.py").decode("utf-8-sig")
            model=json.loads(archive.read("model/model.json"))
            models[number]=model
            sources[number]=code_summary(source)
            item={"filename":filename,"zip_sha256":before,"zip_bytes":path.stat().st_size,
                  "python_members":[n for n in names if n.endswith(".py")],
                  "member_sha256":{n:sha_bytes(archive.read(n)) for n in names if not n.endswith("/")},
                  "model_summary":{k:value_summary(v) for k,v in model.items()},
                  "source_summary":sources[number]}
            if number in ("07","08"):
                item["isolated_import"]=isolated_probe(archive)
                # Verify two concrete repairs in temporary code-only sandboxes, without editing any source/ZIP.
                item["temporary_add_loss_probe"]=isolated_probe(archive,{
                    "model/phonetic_encoder/loss.py":(ROOT/"phonetic_encoder/loss.py").read_bytes()})
                init=b'from .tokenizer import PhoneticTokenizer\nfrom .model import CharPhoneticBiEncoder\n'
                item["temporary_inference_only_init_probe"]=isolated_probe(archive,{
                    "model/phonetic_encoder/__init__.py":init})
            report["packages"][number]=item
        assert before==sha_file(path)
    report["comparisons"]={}
    for a,b in [("06","07"),("07","08"),("06","08")]:
        common=set(models[a])&set(models[b])
        report["comparisons"][a+"_"+b]={
            "same_model_keys":sorted(k for k in common if models[a][k]==models[b][k]),
            "changed_model_keys":sorted(k for k in common if models[a][k]!=models[b][k]),
            "model_keys_added":sorted(set(models[b])-set(models[a])),
            "model_keys_removed":sorted(set(models[a])-set(models[b])),
            "changed_definitions":sorted(k for k in set(sources[a]["definitions"])&set(sources[b]["definitions"])
                if sources[a]["definitions"][k]!=sources[b]["definitions"][k]),
            "definitions_added":sorted(set(sources[b]["definitions"])-set(sources[a]["definitions"])),
            "definitions_removed":sorted(set(sources[a]["definitions"])-set(sources[b]["definitions"])),
            "changed_constants":{k:{"before":sources[a]["constants"].get(k),"after":sources[b]["constants"].get(k)}
                  for k in set(sources[a]["constants"])|set(sources[b]["constants"])
                  if sources[a]["constants"].get(k)!=sources[b]["constants"].get(k)}}
    difference=Decimal("0.9003732004")-Decimal("0.9122153056")
    report["server_score_comparison"]={"score_delta":str(difference),"percentage_point_delta":str(difference*100),
                                      "duration_delta_vs_AS07_seconds":236-372}
    report["scope"]={"models_loaded":False,"predictions_computed":False,"network_allowed":False,
                     "original_sources_or_archives_modified":False,
                     "probes":"Exact archived sources and two temporary import-only repairs; same installed interpreter/packages"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"AS07":report["packages"]["07"]["isolated_import"],
                     "AS08":report["packages"]["08"]["isolated_import"],
                     "temporary_repairs":{n:[report["packages"][n][p]["entrypoint_dense_class_available"]
                       for p in ["temporary_add_loss_probe","temporary_inference_only_init_probe"]] for n in ("07","08")},
                     "changed_model_keys_07_08":report["comparisons"]["07_08"]["changed_model_keys"]},ensure_ascii=False))


if __name__=="__main__":
    main()
