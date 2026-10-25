"""Prepare observed-name/canonical-name pairs using only the training fold."""
import pandas as pd

from .catalog import CatalogNames, canonical_name, has_hangul


def prepare_final_frames(train, catalog):
    """All official training rows, without manufacturing a validation split."""
    if train.empty or train.sample_id.duplicated().any():
        raise ValueError("Training data must be nonempty with unique sample IDs")
    index = CatalogNames(catalog.to_dict("records"))
    if not set(train.target_vessel_id).issubset(index.name_by_id):
        raise ValueError("Training target is missing from the catalog")
    rows = train.copy()
    rows["source"] = rows.vessel_name.map(canonical_name)
    rows["target"] = rows.target_vessel_id.map(index.name_by_id)
    rows["hangul"] = rows.vessel_name.map(has_hangul)
    if rows.source.eq("").any():
        raise ValueError("Empty training name")
    pairs = rows[["source", "target", "hangul"]].drop_duplicates(["source", "target"])
    return pairs.reset_index(drop=True), rows, index


def prepare_frames(train, catalog, folds, heldout_fold):
    if train.sample_id.duplicated().any() or folds.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id in data or folds")
    if set(train.sample_id) != set(folds.sample_id):
        raise ValueError("Fold IDs must exactly cover training data")
    if not set(folds.fold).issubset(set(range(5))) or heldout_fold not in set(folds.fold):
        raise ValueError("Invalid fold assignment")
    aligned = train.merge(folds, on="sample_id", validate="one_to_one", suffixes=("", "_fold"))
    if not aligned.target_vessel_id.eq(aligned.target_vessel_id_fold).all():
        raise ValueError("Fold target IDs disagree with training data")
    if aligned.groupby("target_vessel_id").fold.nunique().max() != 1:
        raise ValueError("A vessel crosses folds: not a vessel-group split")
    index = CatalogNames(catalog.to_dict("records"))
    if not set(train.target_vessel_id).issubset(index.name_by_id):
        raise ValueError("Training target is missing from the catalog")
    aligned["source"] = aligned.vessel_name.map(canonical_name)
    aligned["target"] = aligned.target_vessel_id.map(index.name_by_id)
    if aligned.source.eq("").any():
        raise ValueError("Empty training name")
    aligned["hangul"] = aligned.vessel_name.map(has_hangul)
    fit_rows = aligned.loc[aligned.fold.ne(heldout_fold)].copy()
    validation = aligned.loc[aligned.fold.eq(heldout_fold)].copy()
    # Repeated port records should not teach a name pair hundreds of times.
    # Ambiguous sources with different canonical targets remain separate pairs.
    pairs = fit_rows[["source", "target", "hangul"]].drop_duplicates(["source", "target"])
    return pairs.reset_index(drop=True), validation, fit_rows, index


def read_frames(data_dir, folds_path, heldout_fold, final_train=False):
    train = pd.read_csv(data_dir / "train.csv", keep_default_na=False,
                        dtype={"sample_id": str, "target_vessel_id": str, "vessel_name": str})
    catalog = pd.read_csv(data_dir / "vessel_catalog.csv", keep_default_na=False,
                          dtype={"vessel_id": str, "vessel_name": str})
    if final_train:
        pairs, fit_rows, index = prepare_final_frames(train, catalog)
        return pairs, fit_rows.iloc[:0].copy(), fit_rows, index
    if not folds_path.is_file():
        raise FileNotFoundError(
            f"Common fold file missing: {folds_path}. Supply the distributed file; "
            "do not regenerate it. --final-train trains all rows without evaluation.")
    folds = pd.read_csv(folds_path, keep_default_na=False,
                        dtype={"sample_id": str, "target_vessel_id": str})
    return prepare_frames(train, catalog, folds, heldout_fold)


def select_pilot(validation, per_language, seed):
    if per_language <= 0:
        return validation.sort_values("sample_id").reset_index(drop=True)
    parts = [group.sort_values("sample_id").sample(min(per_language, len(group)), random_state=seed)
             for _, group in validation.groupby("hangul", sort=True)]
    return pd.concat(parts).sort_values("sample_id").reset_index(drop=True)
