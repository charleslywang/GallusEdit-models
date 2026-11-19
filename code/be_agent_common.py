#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AGENT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = AGENT_ROOT / "registry" / "be_model_registry.json"

TARGET_COLUMNS = {
    "target_mean_target_editing_efficiency",
    "target_mean_target_editing_efficiency_no_indel",
    "target_mean_pure_efficiency_over_total",
    "target_mean_purity_among_target_edited",
    "target_weighted_logit_efficiency",
}


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"模型注册表不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def iter_models(registry: dict[str, Any]) -> list[dict[str, Any]]:
    return list(registry.get("models", []))


def resolve_models(registry: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    models = iter_models(registry)
    if not requested:
        return models
    aliases: dict[str, dict[str, Any]] = {}
    for model in models:
        aliases[model["short_name"].upper()] = model
        aliases[model["protein_name"]] = model
    resolved = []
    missing = []
    for item in requested:
        key = item.upper() if len(item) == 1 else item
        if key in aliases:
            resolved.append(aliases[key])
        else:
            missing.append(item)
    if missing:
        known = ", ".join([m["short_name"] + ":" + m["protein_name"] for m in models])
        raise ValueError(f"未知 BE 模型: {missing}; 可用模型: {known}")
    return resolved


def sigmoid(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def select_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    *,
    sample_weight_col: str,
    id_col: str = "normalized_sgrna_name",
) -> tuple[list[str], list[str]]:
    exclude = {
        id_col,
        target_col,
        "sample_weight",
        sample_weight_col,
        "target_weighted_raw_efficiency",
        "target_mean_raw_efficiency_arithmetic",
        "target_raw_bin",
        "target_bin_weight",
        "target_mean_target_editing_efficiency_no_indel",
        "target_mean_pure_efficiency_over_total",
        "target_mean_purity_among_target_edited",
    } | TARGET_COLUMNS
    feature_cols = [c for c in df.columns if c not in exclude]
    categorical_cols = [
        c for c in feature_cols if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    return feature_cols, categorical_cols


def prepare_train_predict_matrices(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], int]:
    train_x = train_df[feature_cols].copy()
    pred_x = predict_df.copy()
    imputed_missing = 0
    for col in feature_cols:
        if col not in pred_x.columns:
            pred_x[col] = np.nan
            imputed_missing += 1
    pred_x = pred_x[feature_cols].copy()

    for col in categorical_cols:
        train_x[col] = train_x[col].fillna("NA").astype(str)
        mode = train_x[col].mode(dropna=True)
        fill = str(mode.iloc[0]) if len(mode) else "NA"
        pred_x[col] = pred_x[col].fillna(fill).astype(str)

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    for col in numeric_cols:
        train_x[col] = pd.to_numeric(train_x[col], errors="coerce")
        pred_x[col] = pd.to_numeric(pred_x[col], errors="coerce")
        fill = float(train_x[col].median())
        if pd.isna(fill):
            fill = 0.0
        train_x[col] = train_x[col].fillna(fill)
        pred_x[col] = pred_x[col].fillna(fill)

    categorical_indices = [feature_cols.index(c) for c in categorical_cols]
    return train_x, pred_x, categorical_indices, imputed_missing


def fit_feature_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    categorical_fills: dict[str, str] = {}
    numeric_fills: dict[str, float] = {}
    for col in categorical_cols:
        series = train_df[col].fillna("NA").astype(str)
        mode = series.mode(dropna=True)
        categorical_fills[col] = str(mode.iloc[0]) if len(mode) else "NA"
    for col in [c for c in feature_cols if c not in categorical_cols]:
        values = pd.to_numeric(train_df[col], errors="coerce")
        fill = float(values.median())
        if pd.isna(fill):
            fill = 0.0
        numeric_fills[col] = fill
    return {
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "categorical_indices": [feature_cols.index(c) for c in categorical_cols],
        "categorical_fills": categorical_fills,
        "numeric_fills": numeric_fills,
    }


def apply_feature_preprocessor(
    df: pd.DataFrame,
    preprocessor: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    feature_cols = list(preprocessor["feature_cols"])
    categorical_cols = set(preprocessor["categorical_cols"])
    categorical_fills = dict(preprocessor["categorical_fills"])
    numeric_fills = dict(preprocessor["numeric_fills"])
    x = df.copy()
    missing_cols = [col for col in feature_cols if col not in x.columns]
    imputed_missing = len(missing_cols)
    if missing_cols:
        x = pd.concat(
            [x, pd.DataFrame(np.nan, index=x.index, columns=missing_cols)],
            axis=1,
        )
    x = x[feature_cols].copy()
    for col in feature_cols:
        if col in categorical_cols:
            x[col] = x[col].fillna(categorical_fills.get(col, "NA")).astype(str)
        else:
            x[col] = pd.to_numeric(x[col], errors="coerce").fillna(float(numeric_fills.get(col, 0.0)))
    return x, imputed_missing


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, low_memory=False)
