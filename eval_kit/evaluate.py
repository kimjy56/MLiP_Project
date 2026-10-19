#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""evaluate.py — 팀 공통 평가 함수 (DACON 236771 항만 선박 매칭)

버전 1.0 (2026-10-19) · 리더 배포. 고치지 말고, 버그가 의심되면 리더에게 알린다. 사용법은 README.md.

점수 = 0.7 × 한글 이름 행 정확도 + 0.3 × 비한글 이름 행 정확도   (대회 공식 산식)
  한글 행  : vessel_name에 '가'~'힣' 글자가 하나라도 있는 행 (주최 baseline_lib.is_hangul_name과 같은 정의)
  정확도   : pred_vessel_id가 target_vessel_id와 글자 그대로 같은 행의 비율 (공백·대소문자를 고쳐 주지 않는다)
  신뢰구간 : 선박 단위 부트스트랩 95% (한글 행·비한글 행을 따로 선박 단위로 복원추출, B=2000, seed 0)
  비교     : --compare 파일과 같은 행끼리 짝지은 부트스트랩. Δ = (--oof) - (--compare)
             개선 = Δ가중 CI 하한 > 0 · 악화 = CI 상한 < 0 · 그 밖 = 동등 (TEAM_RULES §4)

OOF 파일 (CSV, UTF-8)
  필수 sample_id, pred_vessel_id   train 행 전부(12,222행), sample_id 중복 없이
  권장 fold                        그 행을 예측한 모델의 검증 fold 번호. fold 파일과 다르면 입력 오류
  선택 gold_rank                   정답 선박의 후보 안 순위(1 = 1위, 0 = 후보에 없음) → 검색 recall·오류 분해

종료 코드: 0 정상 · 2 입력 오류(점수를 내지 않는다)
필요 패키지: numpy, pandas (pandas 2.0.3 ~ 3.x, Python 3.9 이상)
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_VERSION = "1.0"
W_H = 0.7
HERE = Path(__file__).resolve().parent
DEFAULT_FOLDS = "folds_group5_seed42.csv"
TRAIN_SHA256 = "61bfd199c827c9f7486fc8c9febb419c87ff0611a62d60cfbd7c6d177fa0648c"  # DACON 원본 train.csv
KNOWN_FOLDS = {  # SHA-256 → (분할 이름, 설명)
    "9d09ede0f9254949aacf8cf80638a6c24de0ca5516ca3e9db329f55b9476ca51": ("G5-s42", "기준: 선박 그룹 5-fold seed 42"),
    "c94ffbf0552694a5e21401db46ae4aad0f4727fbaad0faf3fddf99025557901a": ("G5-s0", "보조: 선박 그룹 5-fold seed 0"),
    "1aa3025df3845a5f3e5e7ff72cf1f38141c70652fe89d8dc12997f5dfcf88588": ("G5-s1", "보조: 선박 그룹 5-fold seed 1"),
    "354efbb2488596f38798de98788ca3db363497fa4ec64a3863ef02f914dc51ad": ("G5-s2", "보조: 선박 그룹 5-fold seed 2"),
    "b0b3b9b2f5c31dee9cba8dc4c309ff9ec748c5628c67536ae21fba454cad9526": ("R5-s42", "보조: 행 분할 5-fold seed 42 (아는 선박 시험)"),
}
LB_COLUMNS = [
    "date", "id", "scheme", "oof", "oof_sha8", "rows",
    "hangul", "hangul_lo", "hangul_hi", "non_hangul", "non_hangul_lo", "non_hangul_hi",
    "weighted", "weighted_lo", "weighted_hi", "recall_h", "recall_nh",
    "compare_to", "d_weighted", "d_lo", "d_hi", "verdict",
    "train_sec", "infer_sec", "zip_mb", "note", "checks", "eval_ver",
]


class InputError(Exception):
    """입력 파일 문제. 점수를 내지 않고 종료 코드 2로 끝낸다."""


def is_hangul_name(value: object) -> bool:
    # 주최 baseline_lib.is_hangul_name과 같은 식. 정규화(NFC 등)를 하지 않는다.
    return isinstance(value, str) and any("가" <= char <= "힣" for char in value)


# ---------------------------------------------------------------- 파일 읽기
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path, label: str, required: list) -> pd.DataFrame:
    if not path.is_file():
        raise InputError(f"{label} 파일이 없다: {path}")
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise InputError(f"{label} 파일이 UTF-8이 아니다: {path} (엑셀로 저장하면 CP949로 바뀐다. 코드의 to_csv로 다시 만들 것)")
    except Exception as e:  # 빈 파일, 깨진 CSV 등
        raise InputError(f"{label} 파일을 읽지 못했다: {path} ({type(e).__name__}: {e})")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise InputError(f"{label} 파일에 필수 컬럼 {missing}가 없다. 있는 컬럼: {list(df.columns)}")
    return df


def _positions(sample_ids, truth: dict) -> pd.Series:
    """sample_id → train 행 번호(train에 없으면 NaN)."""
    return truth["pos"].reindex(np.asarray(sample_ids, dtype=object))


def _examples(values, k: int = 3) -> str:
    return ", ".join(list(dict.fromkeys(str(v) for v in values))[:k])


def find_train(arg) -> Path:
    if arg:
        return Path(arg)
    for p in (Path.cwd() / "train.csv", Path.cwd() / "open" / "train.csv", Path.cwd() / "data" / "train.csv"):
        if p.is_file():
            return p
    raise InputError("train.csv를 찾지 못했다. --train 으로 경로를 줄 것")


def load_truth(path: Path) -> dict:
    df = _read(path, "train", ["sample_id", "vessel_name", "target_vessel_id"])
    if df["sample_id"].duplicated().any():
        raise InputError("train.csv의 sample_id가 중복된다. 원본 파일인지 확인할 것")
    sid = df["sample_id"].to_numpy(dtype=object)
    sha = _sha256(path)
    return {
        "path": path, "df": df, "sha": sha, "sha_ok": sha == TRAIN_SHA256, "sid": sid,
        "pos": pd.Series(np.arange(len(sid), dtype=np.int64), index=sid),
        "target": df["target_vessel_id"].to_numpy(dtype=object),
        "is_h": np.array([is_hangul_name(v) for v in df["vessel_name"].tolist()], dtype=bool),
    }


def load_oof(path: Path, truth: dict, allow_subset: bool, label: str = "OOF") -> dict:
    df = _read(path, label, ["sample_id", "pred_vessel_id"])
    sid = df["sample_id"]
    dup = sid[sid.duplicated(keep=False)]
    if len(dup):
        raise InputError(f"{label}: sample_id 중복 {dup.nunique()}종 {len(dup)}행 (예: {_examples(dup)})")
    pos = _positions(sid, truth)
    unknown = pos.isna().to_numpy()
    if unknown.any():
        raise InputError(f"{label}: train에 없는 sample_id {int(unknown.sum())}행 (예: {_examples(sid[unknown])})")
    rows = pos.to_numpy(dtype=np.int64)
    n_truth = len(truth["sid"])
    if len(rows) < n_truth and not allow_subset:
        missing = np.setdiff1d(np.arange(n_truth), rows)
        raise InputError(f"{label}: train {n_truth:,}행 중 {len(missing):,}행이 없다 (예: {_examples(truth['sid'][missing])}). "
                         "일부 행만 평가하려면 --subset")
    order = np.argsort(rows, kind="stable")  # train 행 순서로 맞춘다(파일의 행 순서와 무관)
    rows = rows[order]
    pred = df["pred_vessel_id"].to_numpy(dtype=object)[order]
    stripped = np.array([p.strip() for p in pred], dtype=object)
    out = {"path": path, "rows": rows, "pred": pred, "sha": _sha256(path),
           "n_empty": int((stripped == "").sum()), "n_space": int(((stripped != pred) & (stripped != "")).sum())}
    if "fold" in df.columns:
        fv = pd.to_numeric(df["fold"], errors="coerce").to_numpy(dtype=np.float64)[order]
        if np.isnan(fv).any() or (fv != np.round(fv)).any():
            raise InputError(f"{label}: fold 컬럼에 정수가 아닌 값이 있다")
        out["fold"] = fv.astype(np.int64)
    if "gold_rank" in df.columns:
        gr = pd.to_numeric(df["gold_rank"], errors="coerce").to_numpy(dtype=np.float64)[order]
        bad = np.isnan(gr) | (gr < 0) | (gr != np.round(gr))
        if bad.any():
            raise InputError(f"{label}: gold_rank에 0 이상의 정수가 아닌 값이 {int(bad.sum())}행 있다 (후보에 없으면 0)")
        out["gold_rank"] = gr.astype(np.int64)
    return out


def load_folds(path: Path, truth: dict) -> dict:
    df = _read(path, "fold", ["sample_id", "target_vessel_id", "fold"])
    n = len(truth["sid"])
    if len(df) != n or df["sample_id"].duplicated().any():
        raise InputError(f"fold 파일의 행({len(df):,})이 train({n:,})과 1:1이 아니다: {path}")
    pos = _positions(df["sample_id"], truth)
    if pos.isna().any():
        raise InputError(f"fold 파일에 train에 없는 sample_id가 있다: {path}")
    rows = pos.to_numpy(dtype=np.int64)
    fv = pd.to_numeric(df["fold"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(fv).any() or (fv != np.round(fv)).any():
        raise InputError(f"fold 파일의 fold 값이 정수가 아니다: {path}")
    fold = np.empty(n, dtype=np.int64)
    fold[rows] = fv.astype(np.int64)
    tgt = np.empty(n, dtype=object)
    tgt[rows] = df["target_vessel_id"].to_numpy(dtype=object)
    bad = tgt != truth["target"]
    if bad.any():
        raise InputError(f"fold 파일과 train.csv의 정답이 {int(bad.sum())}행 다르다. 둘 중 하나가 원본이 아니다")
    sha = _sha256(path)
    name, desc = KNOWN_FOLDS.get(sha, ("?", "등록되지 않은 fold 파일(리더 배포본이 아님)"))
    per_vessel = pd.Series(fold).groupby(truth["target"]).nunique()
    kind = "group" if int((per_vessel > 1).sum()) == 0 else "row"
    known = None
    if kind == "row":  # 행 분할: 그 행의 선박이 다른 fold(학습 부분)에 있으면 '아는 선박'
        known = np.zeros(n, dtype=bool)
        tser = pd.Series(truth["target"])
        for k in np.unique(fold):
            m = fold == k
            known[m] = tser[m].isin(set(tser[~m])).to_numpy()
    return {"path": path, "fold": fold, "sha": sha, "name": name, "desc": desc, "kind": kind, "known": known}


# ---------------------------------------------------------------- 점수와 부트스트랩
def score(correct, is_h) -> dict:
    c = np.asarray(correct, dtype=np.float64)
    h = np.asarray(is_h, dtype=bool)
    n_h, n_nh = int(h.sum()), int((~h).sum())
    sh = float(c[h].mean()) if n_h else float("nan")
    snh = float(c[~h].mean()) if n_nh else float("nan")
    return {"h": sh, "nh": snh, "w": W_H * sh + (1 - W_H) * snh, "n_h": n_h, "n_nh": n_nh}


def _strata(values, is_h, vessel) -> dict:
    """층(한글 행 / 비한글 행)마다 선박별 행 수와 값의 합."""
    out = {}
    for s in (True, False):
        m = is_h == s
        if not m.any():
            out[s] = None
            continue
        codes, _ = pd.factorize(vessel[m])
        n_v = np.bincount(codes).astype(np.float64)
        sums = np.bincount(codes, weights=np.asarray(values, dtype=np.float64)[m], minlength=len(n_v))
        out[s] = (n_v, sums)
    return out


def _reps(rng, n_v, sums, B: int, chunk: int = 200) -> np.ndarray:
    """선박을 복원추출한 B개 재표본의 (값 합 / 행 수)."""
    V = len(n_v)
    vals = []
    for start in range(0, B, chunk):
        k = min(chunk, B - start)
        idx = rng.integers(0, V, size=(k, V))
        idx += (np.arange(k, dtype=idx.dtype) * V)[:, None]
        cnt = np.bincount(idx.ravel(), minlength=k * V).reshape(k, V).astype(np.float64)
        vals.append((cnt @ sums) / (cnt @ n_v))
    return np.concatenate(vals)


def _summ(est: float, reps) -> dict:
    if reps is None:
        return {"est": est, "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    return {"est": est, "lo": float(np.quantile(reps, 0.025)), "hi": float(np.quantile(reps, 0.975)),
            "se": float(np.std(reps, ddof=1))}


def bootstrap(values, is_h, vessel, est: dict, B: int, seed: int) -> dict:
    """선박 단위 층화 부트스트랩. values = 정답 여부(0/1) 또는 짝지은 차이(-1/0/1)."""
    if B <= 0:
        return {k: _summ(est[k], None) for k in ("h", "nh", "w")}
    st = _strata(values, is_h, vessel)
    rng = np.random.default_rng(seed)
    reps = {s: (_reps(rng, st[s][0], st[s][1], B) if st[s] is not None else None) for s in (True, False)}
    rw = W_H * reps[True] + (1 - W_H) * reps[False] if reps[True] is not None and reps[False] is not None else None
    return {"h": _summ(est["h"], reps[True]), "nh": _summ(est["nh"], reps[False]), "w": _summ(est["w"], rw)}


def verdict(ci: dict) -> str:
    if not np.isfinite(ci["lo"]):
        return "판정 불가"
    if ci["lo"] > 0:
        return "개선"
    if ci["hi"] < 0:
        return "악화"
    return "동등"


# ---------------------------------------------------------------- 출력 도우미
def _p(x: float, d: int = 2) -> str:
    return "-" if not np.isfinite(x) else f"{100 * x:.{d}f}%"


def _ci(c: dict, pct: bool = True, d: int = 2) -> str:
    if not np.isfinite(c["lo"]):
        return "[-]"
    if pct:
        return f"[{100 * c['lo']:.{d}f}, {100 * c['hi']:.{d}f}]"
    return f"[{c['lo']:.{d}f}, {c['hi']:.{d}f}]"


def _f(x, d: int = 5) -> str:
    return "" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _signed(x: float, pct: bool, d: int) -> str:
    if not np.isfinite(x):
        return "-"
    return f"{100 * x:+.{d}f}%p" if pct else f"{x:+.{d}f}"


def _slices(truth: dict, rows, correct, is_h, out) -> None:
    df = truth["df"].iloc[rows]
    tests = []
    if "international_gross_tonnage" in df.columns:
        t_na = pd.to_numeric(df["international_gross_tonnage"], errors="coerce").isna().to_numpy()
        tests.append(("국제총톤수 결측", t_na))
    if "flag_name" in df.columns:
        f_na = df["flag_name"].str.strip().isin(["", "nan", "NaN", "NA", "None", "null", "NULL"]).to_numpy()
        tests.append(("국적 결측", f_na))
    cnt = pd.Series(truth["target"]).map(pd.Series(truth["target"]).value_counts()).to_numpy()[rows]
    tests += [("선박당 1행", cnt == 1), ("선박당 2~3행", (cnt >= 2) & (cnt <= 3)), ("선박당 4행 이상", cnt >= 4)]
    if "arrival_year" in df.columns:
        y = pd.to_numeric(df["arrival_year"], errors="coerce").to_numpy()
        tests += [("도착 2015~2018", (y >= 2015) & (y <= 2018)), ("도착 2019~2022", (y >= 2019) & (y <= 2022)),
                  ("도착 2023~2026", (y >= 2023) & (y <= 2026))]
    out("구간별 정확도 (행 수가 적은 구간은 흔들린다)")
    for name, m in tests:
        mh, mn = m & is_h, m & ~is_h
        out(f"  {name:<14} 한글 {_p(correct[mh].mean() if mh.any() else float('nan')):>8} ({int(mh.sum()):>5}행)"
            f" · 비한글 {_p(correct[mn].mean() if mn.any() else float('nan')):>8} ({int(mn.sum()):>6}행)")


def _append_leaderboard(path: Path, line: dict) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    if exists:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
        if header != LB_COLUMNS:
            raise InputError(f"{path}의 머리줄이 evaluate.py {EVAL_VERSION} 형식과 다르다. 새 파일 이름을 쓸 것")
        with open(path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([line[c] for c in LB_COLUMNS])
    else:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: 엑셀에서 한글이 깨지지 않게
            w = csv.writer(f)
            w.writerow(LB_COLUMNS)
            w.writerow([line[c] for c in LB_COLUMNS])


# ---------------------------------------------------------------- 실행
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evaluate.py", description="팀 공통 평가 함수 (DACON 236771). 사용법은 README.md")
    p.add_argument("--oof", required=True, help="평가할 OOF CSV")
    p.add_argument("--train", default=None, help="train.csv 경로 (생략하면 ./train.csv, ./open/, ./data/ 순으로 찾는다)")
    p.add_argument("--folds", default=None, help=f"fold 파일 (생략하면 evaluate.py 옆의 {DEFAULT_FOLDS})")
    p.add_argument("--compare", default=None, help="짝지어 비교할 기준 OOF (Δ = --oof - --compare)")
    p.add_argument("--subset", action="store_true", help="train 일부 행만 든 OOF 허용(변형 시험용). 전체 점수와 섞지 않는다")
    p.add_argument("--slices", action="store_true", help="톤수·국적 결측, 선박당 행 수, 도착연도별 정확도도 출력")
    p.add_argument("--boot", type=int, default=2000, help="부트스트랩 반복 수 (0이면 신뢰구간 생략)")
    p.add_argument("--seed", type=int, default=0, help="부트스트랩 시드 (팀 기록은 기본값 0으로)")
    p.add_argument("--id", default="", help="실험 ID (예: O-03)")
    p.add_argument("--scheme", default="", help="검증 이름 (생략하면 fold 파일에서 자동, 예: G5-s42)")
    p.add_argument("--note", default="", help="무엇을 바꿨는지 한 줄")
    p.add_argument("--train-sec", default="", help="학습 시간(초)")
    p.add_argument("--infer-sec", default="", help="추론 시간(초, test 3,000행 기준)")
    p.add_argument("--zip-mb", default="", help="제출 ZIP 크기(MB)")
    p.add_argument("--leaderboard", default=None, help="요약 줄을 덧붙일 CSV (없으면 머리줄과 함께 만든다)")
    return p


def run(argv=None, out=None) -> dict:
    out = out or print
    a = build_parser().parse_args(argv)
    warns = []
    truth = load_truth(find_train(a.train))
    oof = load_oof(Path(a.oof), truth, a.subset)

    fold_path = Path(a.folds) if a.folds else HERE / DEFAULT_FOLDS
    folds = load_folds(fold_path, truth) if (a.folds or fold_path.is_file()) else None
    checks = ["train=" + ("ok" if truth["sha_ok"] else "DIFF")]
    if not truth["sha_ok"]:
        warns.append("train.csv가 DACON 원본과 바이트가 다르다(엑셀 저장 등). 점수는 내지만 팀원 점수와 비교할 수 없다")
    if folds is None:
        warns.append(f"fold 파일이 없어 fold 검사를 건너뛴다 ({DEFAULT_FOLDS}를 evaluate.py 옆에 둘 것)")
        checks.append("folds=none")
    else:
        checks.append("folds=" + folds["name"])
        if folds["name"] == "?":
            warns.append("등록되지 않은 fold 파일이다. 리더가 배포한 파일만 팀 점수로 인정한다")
    oofs = [("OOF", oof)]
    base = None
    if a.compare:
        base = load_oof(Path(a.compare), truth, a.subset, label="비교 OOF")
        oofs.append(("비교 OOF", base))
    for label, o in oofs:
        if folds is not None and "fold" in o:
            bad = o["fold"] != folds["fold"][o["rows"]]
            if bad.any():
                raise InputError(f"{label}: fold 컬럼이 fold 파일({fold_path.name})과 {int(bad.sum()):,}행 다르다. "
                                 "다른 분할로 만든 OOF는 이 기준으로 비교할 수 없다")
        if o["n_empty"]:
            warns.append(f"{label}: 빈 예측 {o['n_empty']}행 (오답으로 셈)")
        if o["n_space"]:
            warns.append(f"{label}: 앞뒤 공백이 붙은 예측 {o['n_space']}행 (글자 그대로 비교해 오답으로 셈. 제출에서도 오답이 된다)")
    if "fold" in oof:
        checks.append("foldcol=ok")
    else:
        checks.append("foldcol=none")
        warns.append("OOF에 fold 컬럼이 없어 어떤 분할로 만든 OOF인지 확인하지 못했다 (fold 컬럼을 넣을 것)")

    rows = oof["rows"]
    is_h = truth["is_h"][rows]
    vessel = truth["target"][rows]
    correct = (oof["pred"] == truth["target"][rows]).astype(np.int64)
    est = score(correct, is_h)
    boot = bootstrap(correct, is_h, vessel, est, a.boot, a.seed)
    scheme = a.scheme or (folds["name"] if folds is not None else "?")
    if a.subset and len(rows) < len(truth["sid"]):
        scheme += f"/부분{len(rows)}"

    # ---- 출력: 머리
    out(f"evaluate.py {EVAL_VERSION} · OOF {oof['path'].name} ({len(rows):,}행, sha {oof['sha'][:8]})")
    out(f"train {truth['path']} · {len(truth['sid']):,}행 · " + ("원본 해시 일치" if truth["sha_ok"] else "원본과 다름(!)"))
    if folds is not None:
        out(f"fold  {folds['path'].name} · {folds['name']} {folds['desc']} · " +
            ("OOF fold 컬럼 일치" if "fold" in oof else "OOF에 fold 컬럼 없음"))
        if folds["kind"] == "row":
            out("      행 분할(아는 선박 포함) 점수다. 기준(선박 그룹 분할) 점수와 합치거나 맞비교하지 않는다")
    if a.subset and len(rows) < len(truth["sid"]):
        out(f"      부분 평가: train {len(truth['sid']):,}행 중 {len(rows):,}행 (변형 시험 등). 전체 점수와 섞지 않는다")
    out("-" * 72)
    v_h = len(pd.unique(vessel[is_h])) if est["n_h"] else 0
    v_nh = len(pd.unique(vessel[~is_h])) if est["n_nh"] else 0
    out(f"한글    {_p(est['h'], 3):>9}  {_ci(boot['h']):<18} {est['n_h']:>7,}행 {v_h:>6,}척")
    out(f"비한글  {_p(est['nh'], 3):>9}  {_ci(boot['nh']):<18} {est['n_nh']:>7,}행 {v_nh:>6,}척")
    out(f"가중    {_f(est['w']) or '-':>9}  {_ci(boot['w'], pct=False, d=5)}   (0.7·한글 + 0.3·비한글, CI = 선박 단위 부트스트랩 B={a.boot})")
    n_wrong_h, n_wrong_nh = int(est["n_h"] - correct[is_h].sum()), int(est["n_nh"] - correct[~is_h].sum())
    out(f"오답    한글 {n_wrong_h:,}행 · 비한글 {n_wrong_nh:,}행")

    # ---- fold별
    if folds is not None:
        out("-" * 72)
        f_of = folds["fold"][rows]
        parts = []
        for k in np.unique(f_of):
            m = f_of == k
            s = score(correct[m], is_h[m])
            parts.append(f"  fold {k}: 한글 {_p(s['h'], 1):>7} ({s['n_h']:>4}행) · 비한글 {_p(s['nh'], 2):>7} · 가중 {_f(s['w'], 4) or '-'}")
        out("fold별 (한 fold 숫자로 방법을 고르지 않는다)")
        for line in parts:
            out(line)
        if folds["kind"] == "row":
            kn = folds["known"][rows]
            out("아는 선박 / 새 선박 행 (행 분할에서 학습 부분에 같은 선박이 있으면 '아는 선박')")
            for name, m in (("아는 선박", kn), ("새 선박", ~kn)):
                s = score(correct[m], is_h[m])
                out(f"  {name}: 한글 {_p(s['h'], 2):>7} ({s['n_h']:,}행) · 비한글 {_p(s['nh'], 2):>7} ({s['n_nh']:,}행)")

    # ---- 검색 recall
    rec_h = rec_nh = float("nan")
    if "gold_rank" in oof:
        gr = oof["gold_rank"]
        out("-" * 72)
        inc = gr >= 1
        rec_h = float(inc[is_h].mean()) if est["n_h"] else float("nan")
        rec_nh = float(inc[~is_h].mean()) if est["n_nh"] else float("nan")
        miss_h, miss_nh = int(((gr == 0) & is_h).sum()), int(((gr == 0) & ~is_h).sum())
        out(f"검색 recall  한글 {_p(rec_h)} · 비한글 {_p(rec_nh)} (gold_rank ≥ 1)")
        out(f"오답 분해    한글: 검색 누락 {miss_h:,} · 순위 오류 {n_wrong_h - miss_h:,}  |  "
            f"비한글: 검색 누락 {miss_nh:,} · 순위 오류 {n_wrong_nh - miss_nh:,}")
        t5h = float(((gr >= 1) & (gr <= 5))[is_h].mean()) if est["n_h"] else float("nan")
        t5n = float(((gr >= 1) & (gr <= 5))[~is_h].mean()) if est["n_nh"] else float("nan")
        out(f"정답이 5위 안  한글 {_p(t5h)} · 비한글 {_p(t5n)}")
        r1_wrong = int(((gr == 1) & (correct == 0)).sum())
        ok_not1 = int(((gr != 1) & (correct == 1)).sum())
        if r1_wrong:
            warns.append(f"gold_rank=1인데 오답 {r1_wrong}행 (동점을 다르게 깬 경우면 정상, 아니면 gold_rank 계산 확인)")
        if ok_not1:
            warns.append(f"정답인데 gold_rank≠1인 행 {ok_not1}행 → gold_rank 정의 확인 (사전 조회로 맞힌 행도 1로 적는다)")

    if a.slices:
        out("-" * 72)
        _slices(truth, rows, correct, is_h, out)

    # ---- 짝지은 비교
    cmp_res = None
    if base is not None:
        out("-" * 72)
        common = np.intersect1d(rows, base["rows"])
        if len(common) < max(len(rows), len(base["rows"])):
            out(f"(두 OOF의 행이 달라 공통 {len(common):,}행만 비교한다)")
        ia = np.searchsorted(rows, common)
        ib = np.searchsorted(base["rows"], common)
        ca = correct[ia]
        cb = (base["pred"][ib] == truth["target"][common]).astype(np.int64)
        hh = truth["is_h"][common]
        ea, eb = score(ca, hh), score(cb, hh)
        d_est = {k: ea[k] - eb[k] for k in ("h", "nh", "w")}
        dboot = bootstrap(ca - cb, hh, truth["target"][common], d_est, a.boot, a.seed)
        key = "w" if np.isfinite(d_est["w"]) else ("h" if np.isfinite(d_est["h"]) else "nh")
        vd = verdict(dboot[key])
        out(f"짝지은 비교: Δ = 이 OOF - {base['path'].name} (sha {base['sha'][:8]}), 같은 행 {len(common):,}")
        for name, k, m, pct, d in (("한글", "h", hh, True, 2), ("비한글", "nh", ~hh, True, 2), ("가중", "w", None, False, 5)):
            if m is not None:
                fixed, broke = int(((ca == 1) & (cb == 0) & m).sum()), int(((ca == 0) & (cb == 1) & m).sum())
                tail = f"  갈린 행 {fixed + broke:,} (고침 {fixed:,} / 망침 {broke:,})"
            else:
                tail = f"  → 판정: {vd}" + ("" if key == "w" else f" ({'한글' if key == 'h' else '비한글'} 기준)")
            out(f"  {name:<6} {_signed(d_est[k], pct, d):>12}  {_ci(dboot[k], pct=pct, d=d if not pct else 2):<22}{tail}")
        cmp_res = {"d": d_est, "boot": dboot, "verdict": vd, "n_common": int(len(common))}

    # ---- 경고와 요약 줄
    if warns:
        out("-" * 72)
        for w_ in warns:
            out("주의: " + w_)
    line = {
        "date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "id": a.id, "scheme": scheme,
        "oof": oof["path"].name, "oof_sha8": oof["sha"][:8], "rows": str(len(rows)),
        "hangul": _f(est["h"]), "hangul_lo": _f(boot["h"]["lo"]), "hangul_hi": _f(boot["h"]["hi"]),
        "non_hangul": _f(est["nh"]), "non_hangul_lo": _f(boot["nh"]["lo"]), "non_hangul_hi": _f(boot["nh"]["hi"]),
        "weighted": _f(est["w"]), "weighted_lo": _f(boot["w"]["lo"]), "weighted_hi": _f(boot["w"]["hi"]),
        "recall_h": _f(rec_h, 4), "recall_nh": _f(rec_nh, 4),
        "compare_to": base["path"].name if base is not None else "",
        "d_weighted": _f(cmp_res["d"]["w"]) if cmp_res else "", "d_lo": _f(cmp_res["boot"]["w"]["lo"]) if cmp_res else "",
        "d_hi": _f(cmp_res["boot"]["w"]["hi"]) if cmp_res else "", "verdict": cmp_res["verdict"] if cmp_res else "",
        "train_sec": a.train_sec, "infer_sec": a.infer_sec, "zip_mb": a.zip_mb, "note": a.note,
        "checks": ";".join(checks), "eval_ver": EVAL_VERSION,
    }
    out("-" * 72)
    out("요약 줄 (leaderboard.csv 형식):")
    out(",".join(LB_COLUMNS))
    out(",".join(_csv_cell(line[c]) for c in LB_COLUMNS))
    if a.leaderboard:
        _append_leaderboard(Path(a.leaderboard), line)
        out(f"→ {a.leaderboard}에 한 줄 덧붙였다")
    return {"score": est, "boot": boot, "compare": cmp_res, "line": line, "warnings": warns,
            "folds": None if folds is None else folds["name"], "rows": rows, "correct": correct}


def _csv_cell(v: str) -> str:
    v = str(v)
    return '"' + v.replace('"', '""') + '"' if any(ch in v for ch in ',"\n') else v


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):  # 콘솔 인코딩이 달라도 글자 때문에 죽지 않게
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    try:
        run(argv)
    except InputError as e:
        print(f"입력 오류: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
