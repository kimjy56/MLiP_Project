#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""selftest_evaluate.py — evaluate.py 자체 시험 (리더가 배포 전에, 팀원은 환경 확인용으로 한 번 돌린다)

실행: python selftest_evaluate.py --train <train.csv 경로> [--baseline-lib <baseline_submit/model/baseline_lib.py>]
  - 임시 폴더에 train에서 만든 OOF를 써서 시험하고 끝나면 지운다. 화면에는 PASS/FAIL과 집계만 나온다.
  - 마지막 줄이 "전부 통과"가 아니면 evaluate.py를 쓰지 말고 리더에게 알린다.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True  # 묶음 폴더에 __pycache__를 만들지 않는다
_spec = importlib.util.spec_from_file_location("evaluate_under_test", HERE / "evaluate.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

RESULTS = []


def check(name: str, cond, detail: str = "") -> None:
    RESULTS.append((name, bool(cond)))
    print(("PASS  " if cond else "FAIL  ") + name + (f"  ({detail})" if detail else ""))


def run_ev(args):
    """(종료 코드, 결과, 출력 줄)"""
    buf = []
    try:
        return 0, ev.run(args, out=buf.append), buf
    except ev.InputError as e:
        return 2, None, [str(e)]
    except SystemExit as e:
        return int(e.code or 0), None, buf


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--baseline-lib", default=None)
    a = ap.parse_args()
    print(f"Python {sys.version.split()[0]} · numpy {np.__version__} · pandas {pd.__version__} · evaluate.py {ev.EVAL_VERSION}")

    train = Path(a.train)
    tr = pd.read_csv(train, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    N = len(tr)
    is_h = tr["vessel_name"].map(ev.is_hangul_name).to_numpy(bool)
    folds = pd.read_csv(HERE / ev.DEFAULT_FOLDS, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    fold_of = folds.set_index("sample_id").loc[tr["sample_id"], "fold"].to_numpy()
    tmp = Path(tempfile.mkdtemp(prefix="evaltest_"))
    T = ["--train", str(train), "--boot", "300"]

    def write(name: str, df: pd.DataFrame, **kw) -> str:
        p = tmp / name
        df.to_csv(p, index=False, **kw)
        return str(p)

    try:
        base = pd.DataFrame({"sample_id": tr["sample_id"], "fold": fold_of})
        perfect = base.assign(pred_vessel_id=tr["target_vessel_id"], gold_rank=1)
        allwrong = base.assign(pred_vessel_id="VESSEL_NONE", gold_rank=0)
        p_perfect, p_wrong = write("perfect.csv", perfect), write("wrong.csv", allwrong)

        # 1~3. 산식
        c, r, _ = run_ev(["--oof", p_perfect] + T)
        check("정답 그대로 → 한글·비한글·가중 모두 1.0", c == 0 and r["score"]["h"] == 1 and r["score"]["nh"] == 1 and r["score"]["w"] == 1)
        c, r, _ = run_ev(["--oof", p_wrong] + T)
        check("전부 오답 → 0", c == 0 and r["score"]["w"] == 0)
        only_h = base.assign(pred_vessel_id=np.where(is_h, tr["target_vessel_id"], "X"))
        c, r, _ = run_ev(["--oof", write("only_h.csv", only_h)] + T)
        check("한글만 정답 → 가중 0.7", c == 0 and abs(r["score"]["w"] - 0.7) < 1e-12, f"{r['score']['w']:.12f}" if r else "")
        only_nh = base.assign(pred_vessel_id=np.where(~is_h, tr["target_vessel_id"], "X"))
        c, r, _ = run_ev(["--oof", write("only_nh.csv", only_nh)] + T)
        check("비한글만 정답 → 가중 0.3", c == 0 and abs(r["score"]["w"] - 0.3) < 1e-12, f"{r['score']['w']:.12f}" if r else "")
        check("한글 행 1,191 · 비한글 11,031 (공식 정의)", int(is_h.sum()) == 1191 and int((~is_h).sum()) == 11031,
              f"{int(is_h.sum())} / {int((~is_h).sum())}")

        # 4. 기준점 E00 (docs/03: 26.2% · 98.04% · 0.4775 [0.4456, 0.5108], recall 68.6% · 98.54%)
        e00 = HERE / "oof_E00_baseline.csv"
        c, r, out = run_ev(["--oof", str(e00), "--train", str(train)])
        exp_w = 0.7 * 312 / 1191 + 0.3 * 10815 / 11031
        ok = c == 0 and abs(r["score"]["h"] - 312 / 1191) < 1e-12 and abs(r["score"]["nh"] - 10815 / 11031) < 1e-12 \
            and abs(r["score"]["w"] - exp_w) < 1e-12
        check("E00 재현: 한글 312/1191 · 비한글 10815/11031 · 가중 0.47750", ok, f"가중 {r['score']['w']:.6f}" if r else "")
        if r:
            b = r["boot"]["w"]
            check("E00 신뢰구간이 docs/03 [0.4456, 0.5108]와 ±0.003 안", abs(b["lo"] - 0.4456) < 0.003 and abs(b["hi"] - 0.5108) < 0.003,
                  f"[{b['lo']:.4f}, {b['hi']:.4f}]")
            check("E00 검색 recall 68.60% · 98.54%", r["line"]["recall_h"] == "0.6860" and r["line"]["recall_nh"] == "0.9854",
                  f"{r['line']['recall_h']} · {r['line']['recall_nh']}")
            check("기준 fold 파일을 해시로 인식(G5-s42)", r["folds"] == "G5-s42", str(r["folds"]))
        e00df = pd.read_csv(e00, dtype=str, keep_default_na=False)

        # 5. 행 순서 무관 (부트스트랩까지 같아야 함)
        shuf = e00df.sample(frac=1.0, random_state=7)
        c2, r2, _ = run_ev(["--oof", write("e00_shuf.csv", shuf), "--train", str(train)])
        same = c2 == 0 and r and r2["score"] == r["score"] and all(r2["boot"][k] == r["boot"][k] for k in ("h", "nh", "w"))
        check("행 순서를 섞어도 점수·신뢰구간이 같음", same)

        # 6~10. 입력 오류는 점수를 내지 않음(종료 코드 2)
        c, _, msg = run_ev(["--oof", write("dup.csv", pd.concat([e00df, e00df.iloc[:1]]))] + T)
        check("sample_id 중복 → 입력 오류", c == 2, msg[0][:60] if msg else "")
        c, _, msg = run_ev(["--oof", write("miss.csv", e00df.iloc[1:])] + T)
        check("행 누락 → 입력 오류", c == 2, msg[0][:60] if msg else "")
        c, r3, _ = run_ev(["--oof", str(tmp / "miss.csv"), "--subset"] + T)
        check("행 누락 + --subset → 남은 행만 평가", c == 0 and len(r3["rows"]) == N - 1)
        bad_id = e00df.copy()
        bad_id.loc[bad_id.index[0], "sample_id"] = "TRAIN_XXXXX"
        c, _, _ = run_ev(["--oof", write("badid.csv", bad_id)] + T)
        check("train에 없는 sample_id → 입력 오류", c == 2)
        c, _, _ = run_ev(["--oof", write("nocol.csv", e00df.rename(columns={"pred_vessel_id": "prediction"}))] + T)
        check("필수 컬럼 없음 → 입력 오류", c == 2)
        wrongfold = e00df.assign(fold=((e00df["fold"].astype(int) + 1) % 5).astype(str))
        c, _, msg = run_ev(["--oof", write("wrongfold.csv", wrongfold)] + T)
        check("다른 분할의 fold 컬럼 → 입력 오류", c == 2, msg[0][:60] if msg else "")
        badrank = e00df.copy()
        badrank.loc[badrank.index[0], "gold_rank"] = "-1"
        c, _, _ = run_ev(["--oof", write("badrank.csv", badrank)] + T)
        check("gold_rank 음수 → 입력 오류", c == 2)
        check("main()의 종료 코드 2", ev.main(["--oof", str(tmp / "nocol.csv"), "--train", str(train)]) == 2)

        # 11. 공백이 붙은 예측은 오답 + 경고
        sp = perfect.copy()
        sp.loc[sp.index[:5], "pred_vessel_id"] = " " + sp.loc[sp.index[:5], "pred_vessel_id"]
        c, r4, _ = run_ev(["--oof", write("space.csv", sp)] + T)
        check("앞뒤 공백 예측 → 오답으로 세고 경고", c == 0 and int(N - r4["correct"].sum()) == 5 and any("공백" in w for w in r4["warnings"]))

        # 12. BOM·CRLF 파일도 같은 점수
        c, r5, _ = run_ev(["--oof", write("bom.csv", e00df, encoding="utf-8-sig", lineterminator="\r\n")] + T)
        check("UTF-8 BOM + CRLF → 같은 점수", c == 0 and r5["score"] == r["score"])

        # 13. 한글 판정: 공식 함수와 같음
        names = tr["vessel_name"].tolist()
        rx = [bool(re.search("[가-힣]", v)) for v in names]
        check("한글 판정 = 정규식 [가-힣] (train 전 행)", rx == [ev.is_hangul_name(v) for v in names])
        edge = ["가", "힣", "ㄱ", "ㅏ", "가", "ABC", "", None, float("nan"), 123, "A가", "Ⅱ", "가나다 1호", "가́"]
        expect = [True, True, False, False, False, False, False, False, False, False, True, False, True, True]
        check("한글 판정 경계 사례(자모만·NFD 분해형·숫자·None은 비한글)", [ev.is_hangul_name(v) for v in edge] == expect)
        if a.baseline_lib:
            sp2 = importlib.util.spec_from_file_location("baseline_lib_ref", a.baseline_lib)
            bl = importlib.util.module_from_spec(sp2)
            sp2.loader.exec_module(bl)
            check("한글 판정 = 주최 baseline_lib.is_hangul_name (train 전 행 + 경계 사례)",
                  [bl.is_hangul_name(v) for v in names + edge] == [ev.is_hangul_name(v) for v in names + edge])
        else:
            print("SKIP  주최 baseline_lib 대조 (--baseline-lib 경로를 주면 시험)")

        # 14. 짝지은 비교
        c, r6, _ = run_ev(["--oof", str(e00), "--compare", str(e00)] + T)
        d = r6["compare"] if r6 else None
        check("같은 파일끼리 비교 → Δ 0, CI [0, 0], 동등",
              c == 0 and d["d"]["w"] == 0 and d["boot"]["w"]["lo"] == 0 and d["boot"]["w"]["hi"] == 0 and d["verdict"] == "동등")
        c, r7, _ = run_ev(["--oof", p_perfect, "--compare", str(e00)] + T)
        check("정답 vs E00 → 개선", c == 0 and r7["compare"]["verdict"] == "개선")
        c, r8, _ = run_ev(["--oof", str(e00), "--compare", p_perfect] + T)
        check("E00 vs 정답 → 악화", c == 0 and r8["compare"]["verdict"] == "악화")
        check("Δ의 부호: (정답 - E00) = -(E00 - 정답)",abs(r7["compare"]["d"]["w"] + r8["compare"]["d"]["w"]) < 1e-12)

        # 15. leaderboard.csv
        lb = tmp / "leaderboard.csv"
        c1, _, _ = run_ev(["--oof", str(e00), "--leaderboard", str(lb), "--id", "E00", "--note", "주최 baseline, 쉼표, \"따옴표\""] + T)
        c2, _, _ = run_ev(["--oof", p_perfect, "--leaderboard", str(lb), "--id", "T-01"] + T)
        rows = pd.read_csv(lb, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        check("leaderboard.csv 생성 + 덧붙이기(머리줄 1개, 2줄, 쉼표·따옴표 메모 보존)",
              c1 == 0 and c2 == 0 and list(rows.columns) == ev.LB_COLUMNS and len(rows) == 2
              and rows["note"].iloc[0] == "주최 baseline, 쉼표, \"따옴표\"" and rows["weighted"].iloc[1] == "1.00000")
        other = tmp / "other.csv"
        other.write_text("a,b\n1,2\n", encoding="utf-8")
        c, _, _ = run_ev(["--oof", str(e00), "--leaderboard", str(other)] + T)
        check("머리줄이 다른 파일에는 덧붙이지 않음 → 입력 오류", c == 2)

        # 16. 배포 fold 파일 인식
        aux = HERE / "aux"
        names_ok = []
        for fn, nm in (("folds_group5_seed0.csv", "G5-s0"), ("folds_group5_seed1.csv", "G5-s1"),
                       ("folds_group5_seed2.csv", "G5-s2"), ("folds_row5_seed42.csv", "R5-s42")):
            fpath = aux / fn
            if not fpath.is_file():
                names_ok.append(False)
                continue
            fd = ev.load_folds(fpath, ev.load_truth(train))
            names_ok.append(fd["name"] == nm and fd["kind"] == ("row" if nm.startswith("R") else "group"))
            if nm == "R5-s42":
                kn = fd["known"].mean()
                check("행 분할 R5: 아는 선박 행 71.2% (docs/06 A7과 같음)", abs(kn - 0.712) < 0.0005, f"{kn:.4f}")
        check("보조 fold 4개를 해시·종류로 인식", all(names_ok))
        c, _, _ = run_ev(["--oof", p_perfect, "--folds", str(aux / "folds_row5_seed42.csv")] + T)
        check("기준 fold 컬럼이 든 OOF를 행 분할 fold로 평가 → 입력 오류(분할 불일치)", c == 2)

        # 17. 부트스트랩 생략
        c, r9, _ = run_ev(["--oof", str(e00), "--train", str(train), "--boot", "0"])
        check("--boot 0 → 점수만, CI 생략", c == 0 and r9["score"] == r["score"] and r9["line"]["weighted_lo"] == "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print("-" * 60)
    print(f"{len(RESULTS)}개 중 {len(RESULTS) - n_fail}개 통과" + (" — 전부 통과" if n_fail == 0 else f" — 실패 {n_fail}개, 리더에게 알릴 것"))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
