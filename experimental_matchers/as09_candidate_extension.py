"""Experimental AS-09 candidate union, leaving original retrieval/scoring intact.

The isolated function is compiled from the local AS-09 source. It adds a union
after the normal fallback in each language branch. Source and asset parity with
the historical submitted ZIP must be separately verified before evaluation.
"""
from __future__ import annotations

import ast
import hashlib
import inspect

import pandas as pd

from experimental_matchers.ko_fusion.as09_adapter import as09

QUERY_COLUMNS = ("sample_id", "vessel_name", "call_sign", "gross_tonnage",
                 "international_gross_tonnage", "flag_name", "vessel_type", "arrival_year")


class _InjectAfterFallback(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_If(self, node):
        test = node.test
        if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and
            isinstance(test.operand, ast.Name) and test.operand.id == "cand_set"):
            # Do not descend into this fallback: nested checks must remain unchanged.
            self.count += 1
            union = ast.parse("cand_set.update(forced_candidate_indices or ())").body[0]
            return [node, ast.copy_location(union, node)]
        return self.generic_visit(node)


def _build_extended_predictor():
    source = inspect.getsource(as09.predict_rows)
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("Unexpected AS-09 predictor source")
    fn = tree.body[0]
    if fn.name != "predict_rows" or any(a.arg == "forced_candidate_indices" for a in fn.args.args):
        raise RuntimeError("Unsupported AS-09 predictor signature")
    fn.args.kwonlyargs.append(ast.arg(arg="forced_candidate_indices"))
    fn.args.kw_defaults.append(ast.Constant(value=None))
    injector = _InjectAfterFallback()
    tree = injector.visit(tree)
    if injector.count != 2:
        raise RuntimeError("AS-09 candidate fallback structure changed; manual audit required")
    ast.fix_missing_locations(tree)
    namespace = dict(as09.__dict__)
    exec(compile(tree, as09.__file__, "exec"), namespace)
    return namespace["predict_rows"], hashlib.sha256(source.encode("utf-8")).hexdigest()


_extended_predict_rows, SOURCE_FUNCTION_SHA256 = _build_extended_predictor()


def predict_row(query: pd.DataFrame, model, index, *, candidates=(), use_train_lookup=None,
                dense_matcher=None, dense_mode=None, dense_weight=2.5, diagnostic=False):
    """Call the baseline scorer with a fixed-catalog candidate union for one row."""
    if len(query) != 1 or "vessel_name" not in query:
        raise ValueError("Expected exactly one input row")
    query = query.loc[:, [c for c in QUERY_COLUMNS if c in query]].copy()
    mode = dense_mode or model.get("dense_mode", "none")
    is_korean = as09.is_korean_name(query.iloc[0]["vessel_name"])
    if is_korean and mode == "hybrid" and dense_matcher is None:
        raise RuntimeError("Dense matcher missing for an AS-09 hybrid Korean query")
    if is_korean and (not model.get("type_trans_prob") or not model.get("flag_trans_prob")):
        raise RuntimeError("AS-09 Korean fallback does not initialize meta_score; transition tables are required")
    ids = tuple(dict.fromkeys(str(v) for v in candidates))
    if any(vid not in index.vid_to_idx for vid in ids):
        raise ValueError("Candidate is absent from the fixed catalog")
    result = _extended_predict_rows(
        query, model, index, use_train_lookup=use_train_lookup, gold_ids=None,
        top_candidates=index.n if diagnostic else None, dense_matcher=dense_matcher,
        dense_mode=mode, dense_weight=dense_weight,
        forced_candidate_indices=tuple(index.vid_to_idx[vid] for vid in ids),
    )
    if diagnostic:
        return result[0][0], tuple(result[2][0])
    return result[0][0]
