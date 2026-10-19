#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_submission.py — 제출 zip 점검 (DACON 236771 항만 선박 매칭)

버전 1.0 (2026-10-19) · 공유 폴더에 올리기 전에 각자 돌리고, 리더도 제출 전에 다시 돌린다.
실행(서버와 같은 Python 3.11 환경에서):
  python check_submission.py --zip submit_AS-01.zip --train <train.csv> [--catalog <vessel_catalog.csv>]

하는 일
  1. zip 구조   맨 위에 script.py, requirements.txt, model/만. 역슬래시·절대 경로·'..' 금지. 크기(한도 10GB)
  2. 패키지     requirements.txt에 서버에 이미 있는 패키지를 적었는지(버전 충돌로 설치 오류 위험)
  3. 코드 훑기  네트워크·외부 AI API·train.csv 읽기·개인 경로·pickle·시간 읽기 흔적(경고만)
  4. 실행       train에서 뽑은 n행(정답 열 제거, sample_id는 TEST_로 바꿈)을 data/test.csv로,
                카탈로그를 data/vessel_catalog.csv로 두고 python script.py → output/submission.csv 확인.
                서버처럼 output/은 미리 만들어 두고, data/에 파일을 쓰면 실패로 본다(서버에서는 읽기 전용).
  5. 행 독립    전체 / 순서 섞기(다른 작업 폴더에서) / 절반 / 한 행씩 k번 — 같은 행의 예측이 모두 같아야 한다.
                두 번 이상 돌리므로 난수 때문에 결과가 바뀌는 코드도 여기서 잡힌다.
  6. 시간       행 수가 다른 실행 시간으로 3,000행·30,000행 시간을 어림한다(서버 6 vCPU, 추론 한도 20분).
주의: zip 안의 코드를 실제로 실행한다. 받은 zip은 script.py를 먼저 훑어보고 돌린다.
점수는 참고용이다. 뽑은 행은 모델이 학습에 쓴 행이라 실제보다 높게 나온다.
종료 코드: 0 = 실패 없음(주의는 있을 수 있음) · 1 = 실패 있음 · 2 = 입력 오류
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

VERSION = "1.0"
ALLOWED_TOP = {"script.py", "requirements.txt", "model"}
SERVER_PKGS = {"torch", "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "joblib", "lightgbm", "catboost",
               "rapidfuzz", "unidecode", "transformers", "accelerate", "safetensors", "sentencepiece", "tqdm"}
SERVER_VERSIONS = {"python": "3.11", "pandas": "2.0.3", "numpy": "1.26.4", "sklearn": "1.5.2", "lightgbm": "4.6.0"}
SUSPECT = [
    (r"^\s*(import|from)\s+(requests|urllib|http\.client|socket|httpx|aiohttp)\b", "네트워크 모듈(서버에 인터넷 없음)"),
    (r"\b(openai|anthropic|google\.generativeai|google\.genai)\b", "외부 AI API(서버에서 호출 불가)"),
    (r"train\.csv", "train.csv를 읽음(서버에 있다고 가정하지 않는다. 필요한 값은 model/에)"),
    (r"\b[A-Za-z]:[\\/]|/Users/|/home/|Desktop", "개인 절대 경로"),
    (r"\bpickle\b|\.pkl\b|joblib\.load", "pickle 계열(서버 버전과 다르면 깨질 수 있음, json·npz·LightGBM 텍스트 권장)"),
    (r"time\.time\(\)|perf_counter\(|datetime\.now\(", "시간 읽기(로그용이면 괜찮다. 남은 시간에 따라 계산을 바꾸면 행 독립 위반)"),
]


class Report:
    def __init__(self) -> None:
        self.n_fail = 0
        self.n_warn = 0
        self.seen = set()  # 실행마다 반복되는 같은 문제는 한 번만 센다

    def once(self, key: str) -> bool:
        if key in self.seen:
            return False
        self.seen.add(key)
        return True

    def ok(self, msg: str) -> None:
        print("통과  " + msg)

    def info(self, msg: str) -> None:
        print("정보  " + msg)

    def warn(self, msg: str) -> None:
        self.n_warn += 1
        print("주의  " + msg)

    def fail(self, msg: str) -> None:
        self.n_fail += 1
        print("실패  " + msg)


def is_hangul_name(value) -> bool:  # 주최 baseline_lib.is_hangul_name과 같은 정의
    return isinstance(value, str) and any("가" <= char <= "힣" for char in value)


# ---------------------------------------------------------------- 1~3. 정적 점검
def check_structure(zpath: Path, rep: Report) -> bool:
    """zip 구조를 보고, 실행해 볼 수 있으면 True."""
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        rep.fail("zip 파일이 아니거나 깨졌다")
        return False
    with zf:
        names = [n for n in zf.namelist() if n]
        runnable = True
        bs = [n for n in names if "\\" in n]
        if bs:
            rep.fail(f"경로에 역슬래시가 든 항목 {len(bs)}개(예: {bs[0]}). 리눅스 서버에서 폴더가 깨진다. "
                     "PowerShell Compress-Archive 대신 Python zipfile로 만들 것")
        unsafe = [n for n in names if n.startswith("/") or re.match(r"^[A-Za-z]:", n) or ".." in PurePosixPath(n.replace("\\", "/")).parts]
        if unsafe:
            rep.fail(f"절대 경로나 '..'가 든 항목 {len(unsafe)}개(예: {unsafe[0]})")
            runnable = False
        tops = sorted({n.replace("\\", "/").split("/")[0] for n in names})
        if "script.py" not in tops:
            wrapped = sorted({n.split("/")[0] for n in names if n.replace("\\", "/").endswith("/script.py") and n.replace("\\", "/").count("/") == 1})
            if wrapped:
                rep.fail(f"script.py가 맨 위가 아니라 '{wrapped[0]}/' 폴더 안에 있다. 폴더째가 아니라 그 안의 내용물을 압축할 것")
            else:
                rep.fail("맨 위에 script.py가 없다")
            runnable = False
        if "requirements.txt" not in tops:
            rep.fail("맨 위에 requirements.txt가 없다(비어 있어도 있어야 한다)")
        if "model" not in tops:
            rep.fail("맨 위에 model/ 폴더가 없다")
        extra = [t for t in tops if t not in ALLOWED_TOP]
        if extra:
            rep.fail(f"맨 위에 허용되지 않은 항목 {extra} (script.py, requirements.txt, model/만 둔다)")
        junk = [n for n in names if "__MACOSX" in n or n.endswith(".DS_Store") or "__pycache__" in n or ".ipynb_checkpoints" in n]
        if junk:
            rep.warn(f"필요 없는 파일 {len(junk)}개(__pycache__·.DS_Store 등). 빼면 ZIP이 작아진다(동점 기준 2순위)")
        raw = sum(i.file_size for i in zf.infolist())
        size = zpath.stat().st_size
        rep.info(f"ZIP {size / 1e6:.2f} MB (풀면 {raw / 1e6:.2f} MB, 한도 10 GB) · 항목 {len(names)}개")
        if size > 10 * 1024 ** 3:
            rep.fail("ZIP이 10 GB를 넘는다")
        if not (bs or unsafe or extra or "script.py" not in tops or "requirements.txt" not in tops or "model" not in tops):
            rep.ok("zip 구조: 맨 위에 script.py · requirements.txt · model/만 있음")

        if "requirements.txt" in tops:
            txt = zf.read("requirements.txt").decode("utf-8-sig", errors="replace")
            pk = []
            for line in txt.splitlines():
                s = line.split("#")[0].strip()
                if not s or s.startswith("-"):
                    continue
                pk.append((re.split(r"[<>=!~\[;\s]", s, maxsplit=1)[0].strip().lower().replace("_", "-"), s))
            dup = [s for n, s in pk if n in SERVER_PKGS]
            other = [s for n, s in pk if n not in SERVER_PKGS]
            if dup:
                rep.warn(f"서버에 이미 있는 패키지를 적었다 {dup} → 빼는 것이 안전하다(버전 충돌 시 설치 오류)")
            if other:
                rep.warn(f"추가로 설치할 패키지 {other} → 설치 10분 안·설치 때만 인터넷. 정말 필요한지 확인")
            if not pk:
                rep.ok("requirements.txt 비어 있음(서버 기본 패키지만 사용)")

        hits = {}
        for n in names:
            if n.endswith(".py"):
                src = zf.read(n).decode("utf-8", errors="replace")
                for pat, why in SUSPECT:
                    if re.search(pat, src, flags=re.MULTILINE):
                        hits.setdefault(why, set()).add(n)
        for why, files in hits.items():
            rep.warn(f"코드 훑기 — {why}: {', '.join(sorted(files)[:3])}")
        if not hits:
            rep.ok("코드 훑기: 의심 패턴 없음")
    return runnable


# ---------------------------------------------------------------- 4~6. 실행 점검
def safe_extract(zpath: Path, dest: Path) -> None:
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.infolist():
            p = PurePosixPath(m.filename.replace("\\", "/"))
            if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", m.filename):
                raise RuntimeError(f"위험한 경로: {m.filename}")
        zf.extractall(dest)


def snapshot(d: Path) -> set:
    out = set()
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in files:
            out.add(str(Path(root, f).relative_to(d)))
    return out


def run_once(zpath: Path, work: Path, test: pd.DataFrame, catalog: Path, bom: bool, label: str,
             other_cwd: bool, timeout: int) -> dict:
    d = Path(tempfile.mkdtemp(prefix=f"run_{label}_", dir=work))
    safe_extract(zpath, d)
    (d / "data").mkdir(exist_ok=True)
    test.to_csv(d / "data" / "test.csv", index=False, encoding="utf-8-sig" if bom else "utf-8")
    shutil.copyfile(catalog, d / "data" / "vessel_catalog.csv")
    (d / "output").mkdir(exist_ok=True)  # 서버는 output/을 붙여 준다
    before = snapshot(d)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "6",
                "NUMEXPR_NUM_THREADS": "6", "PYTHONIOENCODING": "utf-8"})
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, str(d / "script.py")], cwd=str(work if other_cwd else d), env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        rc, err = p.returncode, p.stderr.decode("utf-8", errors="replace")
        (d / "_stdout.log").write_bytes(p.stdout)
        (d / "_stderr.log").write_bytes(p.stderr)
    except subprocess.TimeoutExpired:
        rc, err = None, f"시간 초과({timeout}초)"
    sec = time.perf_counter() - t0
    after = snapshot(d) - {"_stdout.log", "_stderr.log"}
    new = sorted(after - before)
    res = {"label": label, "rc": rc, "sec": sec, "dir": d, "rows": len(test),
           "err_tail": "\n".join(err.strip().splitlines()[-5:]),
           "data_new": [x for x in new if x.startswith("data")],
           "other_new": [x for x in new if not x.startswith("data") and not x.startswith("output")], "sub": None}
    sp = d / "output" / "submission.csv"
    if rc == 0 and sp.is_file():
        res["sub"] = pd.read_csv(sp, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return res


def check_output(r: dict, test: pd.DataFrame, cat_ids: set, rep: Report) -> bool:
    lab = r["label"]
    if r["rc"] is None:
        rep.fail(f"[{lab}] {r['err_tail']}")
        return False
    if r["rc"] != 0:
        rep.fail(f"[{lab}] script.py가 오류로 끝남(종료 코드 {r['rc']}). 마지막 줄:\n      " + r["err_tail"].replace("\n", "\n      "))
        return False
    if r["data_new"] and rep.once("data_new"):
        rep.fail(f"[{lab}] data/에 파일을 썼다 {r['data_new'][:3]} → 서버의 data/는 읽기 전용이라 오류가 난다")
    if r["other_new"] and rep.once("other_new"):
        rep.warn(f"[{lab}] output/ 밖에 새 파일 {r['other_new'][:3]} → 서버에서 쓰기가 막혀 있으면 오류. output/만 쓰는 것이 안전")
    s = r["sub"]
    if s is None:
        rep.fail(f"[{lab}] output/submission.csv가 없다")
        return False
    ok = True
    if list(s.columns) != ["sample_id", "target_vessel_id"]:
        rep.fail(f"[{lab}] submission.csv 컬럼이 {list(s.columns)} — sample_id, target_vessel_id 두 개여야 한다")
        return False
    if len(s) != len(test) or s["sample_id"].duplicated().any() or set(s["sample_id"]) != set(test["sample_id"]):
        rep.fail(f"[{lab}] submission.csv 행이 test와 1:1이 아니다(행 {len(s)} vs {len(test)}, 중복 {int(s['sample_id'].duplicated().sum())})")
        ok = False
    empty = int((s["target_vessel_id"].str.strip() == "").sum())
    if empty:
        rep.fail(f"[{lab}] 빈 예측 {empty}행")
        ok = False
    unknown = int((~s["target_vessel_id"].isin(cat_ids)).sum())
    if unknown:
        rep.warn(f"[{lab}] 카탈로그에 없는 vessel_id {unknown}행(무조건 오답)")
    if ok and list(s["sample_id"]) != list(test["sample_id"]):
        rep.warn(f"[{lab}] 출력 행 순서가 test.csv 순서와 다르다(채점은 sample_id로 맞출 가능성이 크지만 test 순서로 쓰는 것이 안전)")
    return ok


def same_preds(base: pd.DataFrame, other: pd.DataFrame) -> tuple:
    m = other.merge(base, on="sample_id", how="inner", suffixes=("", "_a"))
    return int((m["target_vessel_id"] != m["target_vessel_id_a"]).sum()), len(m)


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(prog="check_submission.py", description="제출 zip 점검 (DACON 236771)")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--train", required=True, help="train.csv 경로")
    ap.add_argument("--catalog", default=None, help="vessel_catalog.csv 경로(생략하면 train.csv와 같은 폴더)")
    ap.add_argument("--n", type=int, default=300, help="시험할 행 수(기본 300, 한글 30%%)")
    ap.add_argument("--single", type=int, default=3, help="한 행씩 따로 돌려 볼 행 수(기본 3)")
    ap.add_argument("--timeout", type=int, default=1200, help="실행 한 번의 제한 시간(초)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="임시 실행 폴더(로그 포함)를 지우지 않는다")
    a = ap.parse_args(argv)

    zpath, train = Path(a.zip), Path(a.train)
    catalog = Path(a.catalog) if a.catalog else train.parent / "vessel_catalog.csv"
    for p, lab in ((zpath, "zip"), (train, "train.csv"), (catalog, "vessel_catalog.csv")):
        if not p.is_file():
            print(f"입력 오류: {lab} 파일이 없다: {p}", file=sys.stderr)
            return 2
    rep = Report()

    def _ver(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:
            return "없음"

    vers = {"python": ".".join(map(str, sys.version_info[:3])), "pandas": pd.__version__, "numpy": np.__version__,
            "sklearn": _ver("sklearn"), "lightgbm": _ver("lightgbm")}
    print(f"check_submission.py {VERSION} · {zpath.name} · " + " · ".join(f"{k} {v}" for k, v in vers.items()))
    diff = [f"{k} {vers[k]}(서버 {v})" for k, v in SERVER_VERSIONS.items() if not vers[k].startswith(v)]
    if diff:
        rep.warn("서버와 다른 버전에서 실행 중: " + ", ".join(diff) + " → 서버와 같은 Python 3.11 환경에서 다시 확인할 것")

    print("-" * 72)
    runnable = check_structure(zpath, rep)
    if not runnable:
        print("-" * 72)
        print(f"결과: 실패 {rep.n_fail} · 주의 {rep.n_warn} → 구조를 고친 뒤 다시 점검")
        return 1

    # 시험 데이터: train에서 n행(한글 30%), 정답 열 제거, sample_id → TEST_
    raw = train.read_bytes()[:3]
    bom = raw == b"\xef\xbb\xbf"
    tr = pd.read_csv(train, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    is_h = tr["vessel_name"].map(is_hangul_name).to_numpy(bool)
    n = min(a.n, len(tr))
    n_h = min(int(round(n * 0.3)), int(is_h.sum()))
    rng = np.random.RandomState(a.seed)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(is_h), n_h, replace=False),
                                  rng.choice(np.flatnonzero(~is_h), n - n_h, replace=False)]))
    sub = tr.iloc[idx].reset_index(drop=True)
    sub["sample_id"] = [f"TEST_{i:04d}" for i in range(len(sub))]
    truth = sub[["sample_id", "target_vessel_id"]].copy()
    test = sub.drop(columns=["target_vessel_id"])
    cat_ids = set(pd.read_csv(catalog, encoding="utf-8-sig", dtype=str, keep_default_na=False, usecols=["vessel_id"])["vessel_id"])
    print("-" * 72)
    rep.info(f"시험 데이터: train에서 {n}행(한글 {n_h}행), 정답 열 제거 → data/test.csv" + (" (UTF-8 BOM, 원본과 같게)" if bom else ""))

    work = Path(tempfile.mkdtemp(prefix="subcheck_"))
    try:
        A = run_once(zpath, work, test, catalog, bom, "전체", False, a.timeout)
        if not check_output(A, test, cat_ids, rep):
            print("-" * 72)
            print(f"결과: 실패 {rep.n_fail} · 주의 {rep.n_warn} → 실행 오류를 고친 뒤 다시 점검" +
                  (f" (로그: {A['dir']})" if a.keep else " (--keep을 붙이면 로그가 남는다)"))
            return 1
        rep.ok(f"[전체] {n}행 실행 {A['sec']:.1f}초 · output/submission.csv 형식 정상")
        pa = A["sub"]
        m = pa.merge(truth, on="sample_id", suffixes=("", "_t"))
        c = (m["target_vessel_id"] == m["target_vessel_id_t"]).to_numpy()
        hh = test.set_index("sample_id").loc[m["sample_id"], "vessel_name"].map(is_hangul_name).to_numpy(bool)
        acc_h = c[hh].mean() if hh.any() else float("nan")
        acc_n = c[~hh].mean() if (~hh).any() else float("nan")
        rep.info(f"참고 점수(학습에 쓴 행이라 높게 나옴): 한글 {acc_h:.1%} · 비한글 {acc_n:.1%} · 가중 {0.7 * acc_h + 0.3 * acc_n:.4f}")

        # 순서 섞기 + 다른 작업 폴더
        shuf = test.sample(frac=1.0, random_state=a.seed + 1).reset_index(drop=True)
        B = run_once(zpath, work, shuf, catalog, bom, "섞기", True, a.timeout)
        if B["rc"] != 0:  # 다른 작업 폴더에서 실패 → zip 폴더에서 다시 돌려 원인을 가른다
            B2 = run_once(zpath, work, shuf, catalog, bom, "섞기", False, a.timeout)
            if B2["rc"] == 0:
                rep.warn("[섞기] 작업 폴더가 zip 폴더가 아니면 실패한다 → 경로를 Path(__file__).parent 기준으로 잡을 것")
            B = B2
        if check_output(B, shuf, cat_ids, rep):
            k, tot = same_preds(pa, B["sub"])
            (rep.fail if k else rep.ok)(f"[섞기] 행 순서를 섞어 다시 실행: 예측이 다른 행 {k}/{tot}" +
                                        (" → 행 독립 위반이거나 실행마다 결과가 바뀐다(난수·set 순서 등)" if k else ""))
        # 절반
        half = test.iloc[: max(1, n // 2)].reset_index(drop=True)
        C = run_once(zpath, work, half, catalog, bom, "절반", False, a.timeout)
        if check_output(C, half, cat_ids, rep):
            k, tot = same_preds(pa, C["sub"])
            (rep.fail if k else rep.ok)(f"[절반] 앞 {len(half)}행만 넣어 실행: 예측이 다른 행 {k}/{tot}" + (" → 행 독립 위반" if k else ""))
        # 한 행씩
        singles = []
        pick = [int(np.flatnonzero(test["vessel_name"].map(is_hangul_name).to_numpy())[0])] if n_h else []
        pick += [i for i in rng.choice(n, size=min(n, a.single + 1), replace=False).tolist() if i not in pick]
        for j, i in enumerate(pick[: a.single]):
            one = test.iloc[[i]].reset_index(drop=True)
            D = run_once(zpath, work, one, catalog, bom, f"한행{j + 1}", False, a.timeout)
            if check_output(D, one, cat_ids, rep):
                k, _ = same_preds(pa, D["sub"])
                singles.append((k, D["sec"]))
        if singles:
            bad = sum(k for k, _ in singles)
            (rep.fail if bad else rep.ok)(f"[한 행씩] {len(singles)}번 따로 실행: 예측이 다른 경우 {bad}" + (" → 행 독립 위반" if bad else ""))

        # 시간 어림: t = 고정 + 행당 × 행 수 (전체·절반·한 행 실행으로 직선 맞춤)
        pts = [(A["rows"], A["sec"])] + ([(C["rows"], C["sec"])] if C["rc"] == 0 else []) + [(1, s_) for _, s_ in singles]
        if len(pts) >= 2 and len({p[0] for p in pts}) >= 2:
            x = np.array([p[0] for p in pts], dtype=float)
            y = np.array([p[1] for p in pts], dtype=float)
            b, a0 = np.polyfit(x, y, 1)
            b, a0 = max(b, 0.0), max(a0, 0.0)
            t3k, t30k = a0 + b * 3000, a0 + b * 30000
            msg = (f"시간 어림(이 PC, 6스레드 제한): 시작 {a0:.1f}초 + 행당 {b * 1000:.1f}ms → 3,000행 {t3k:.0f}초 · "
                   f"30,000행 {t30k:.0f}초 (서버 추론 한도 1,200초)")
            if t3k > 1200:
                rep.fail(msg + " → 3,000행도 한도를 넘는다")
            elif t30k > 600:
                rep.warn(msg + " → test가 10배면 위험(TEAM_RULES §5 목표: 10배여도 10분 안)")
            else:
                rep.ok(msg)
        rep.info("인터넷 차단은 이 도구가 시험하지 못한다. 네트워크 코드가 없는지 코드 훑기 결과를 확인할 것")
    finally:
        if a.keep:
            print(f"(실행 폴더를 남겼다: {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print("-" * 72)
    verdict = "올려도 된다" + (" (주의 항목 확인)" if rep.n_warn else "") if rep.n_fail == 0 else "고친 뒤 다시 점검"
    print(f"결과: 실패 {rep.n_fail} · 주의 {rep.n_warn} → {verdict}")
    return 0 if rep.n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
