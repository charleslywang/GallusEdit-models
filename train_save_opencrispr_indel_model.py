#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from code.be_agent_common import apply_feature_preprocessor, fit_feature_preprocessor


AGENT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = AGENT_ROOT / "opencrispr_indel_model_registry.json"
DEFAULT_OUTPUT_DIR = AGENT_ROOT / "final_models_indel"

INDEL_LEAKAGE_COLUMNS = {
    "normalized_sgrna_name",
    "sample_weight",
    "target_weighted_raw_efficiency",
    "target_mean_raw_efficiency_arithmetic",
    "target_raw_bin",
    "target_bin_weight",
    "target_raw_efficiency_from_mean_efficiency",
    "target_percent_from_mean_efficiency",
    "target_mean_target_editing_efficiency",
    "target_mean_target_editing_efficiency_no_indel",
    "target_mean_pure_efficiency_over_total",
    "target_mean_purity_among_target_edited",
    "target_weighted_logit_efficiency",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and save final indel/nuclease TabPFN models for downstream agent inference.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="*", default=None, help="Model short names or protein names. Default: all registered indel models.")
    parser.add_argument("--target-column", default="target_mean_target_editing_efficiency")
    parser.add_argument("--sample-weight-column", default="sample_weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-mode", default="fit_preprocessors", choices=["fit_preprocessors", "low_memory", "batched"])
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=0, help="Smoke-test only. 0 uses all final training rows.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing model artifacts.")
    return parser.parse_args()


def load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"OpenCRISPR indel 注册表不存在: {REGISTRY_PATH}")
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def resolve_model_path(registry: dict[str, Any], arg_model_path: str) -> str:
    if arg_model_path.strip():
        return arg_model_path
    default_path = Path(registry.get("tabpfn_default_model_path", ""))
    if default_path.exists():
        return str(default_path)
    return "auto"


def resolve_models(registry: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    models = list(registry.get("models", []))
    if not requested:
        return models
    aliases: dict[str, dict[str, Any]] = {}
    for model in models:
        aliases[model["short_name"].lower()] = model
        aliases[model["protein_name"].lower()] = model
        aliases[model.get("protein_group", "").lower()] = model
    resolved = []
    missing = []
    for item in requested:
        model = aliases.get(item.lower())
        if model:
            resolved.append(model)
        else:
            missing.append(item)
    if missing:
        known = ", ".join([m["short_name"] for m in models])
        raise ValueError(f"未知 indel 模型: {missing}; 可用模型: {known}")
    return resolved


def select_indel_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    *,
    sample_weight_col: str,
) -> tuple[list[str], list[str]]:
    exclude = set(INDEL_LEAKAGE_COLUMNS)
    exclude.add(target_col)
    exclude.add(sample_weight_col)
    feature_cols = [c for c in df.columns if c not in exclude]
    categorical_cols = [
        c for c in feature_cols if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    return feature_cols, categorical_cols


def main() -> None:
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("POSTHOG_DISABLED", "true")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    from tabpfn import TabPFNRegressor

    args = parse_args()
    registry = load_registry()
    models = resolve_models(registry, args.models)
    model_path = resolve_model_path(registry, args.model_path)

    summary_rows = []
    for model_config in models:
        short_name = model_config["short_name"]
        model_dir = args.output_dir / short_name
        artifact_path = model_dir / f"{short_name}.final_tabpfn_indel_model.joblib"
        metadata_path = model_dir / f"{short_name}.final_tabpfn_indel_model.metadata.json"
        if artifact_path.exists() and not args.overwrite:
            raise FileExistsError(f"模型文件已存在，若要覆盖请加 --overwrite: {artifact_path}")
        model_dir.mkdir(parents=True, exist_ok=True)

        training_path = Path(model_config["training_tsv"])
        train_df = pd.read_csv(training_path, sep="\t", low_memory=False)
        protein_group = model_config.get("protein_group", short_name)
        if "protein_group" in train_df.columns:
            train_df = train_df[train_df["protein_group"] == protein_group].copy()
        if train_df.empty:
            raise ValueError(f"{training_path} 中没有 protein_group={protein_group} 的训练数据")
        if args.target_column not in train_df.columns:
            raise ValueError(f"{training_path} 缺少 target column: {args.target_column}")
        train_df = train_df[train_df[args.target_column].notna()].copy().reset_index(drop=True)
        if args.max_train_rows > 0 and len(train_df) > args.max_train_rows:
            train_df = train_df.sample(n=args.max_train_rows, random_state=args.random_state).reset_index(drop=True)

        feature_cols, categorical_cols = select_indel_feature_columns(
            train_df,
            args.target_column,
            sample_weight_col=args.sample_weight_column,
        )
        preprocessor = fit_feature_preprocessor(train_df, feature_cols, categorical_cols)
        train_x, _ = apply_feature_preprocessor(train_df, preprocessor)
        y_train = pd.to_numeric(train_df[args.target_column], errors="coerce").astype(float).to_numpy()

        model = TabPFNRegressor(
            model_path=model_path,
            device=args.device,
            fit_mode=args.fit_mode,
            n_estimators=args.n_estimators,
            n_preprocessing_jobs=args.n_preprocessing_jobs,
            categorical_features_indices=preprocessor["categorical_indices"],
            random_state=args.random_state,
        )
        print(f"[FIT] {short_name} indel n={len(train_df)} features={len(feature_cols)}")
        model.fit(train_x, y_train)

        artifact = {
            "model": model,
            "preprocessor": preprocessor,
            "model_config": model_config,
            "target_column": args.target_column,
            "sample_weight_column": args.sample_weight_column,
            "model_path": model_path,
            "fit_params": {
                "device": args.device,
                "fit_mode": args.fit_mode,
                "n_estimators": args.n_estimators,
                "n_preprocessing_jobs": args.n_preprocessing_jobs,
                "random_state": args.random_state,
            },
            "n_train": int(len(train_df)),
            "n_features": int(len(feature_cols)),
            "n_categorical_features": int(len(categorical_cols)),
        }
        joblib.dump(artifact, artifact_path, compress=3)
        metadata = {k: v for k, v in artifact.items() if k not in {"model", "preprocessor"}}
        metadata["artifact_path"] = str(artifact_path)
        metadata["feature_cols"] = feature_cols
        metadata["categorical_cols"] = categorical_cols
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_rows.append(metadata)
        print(f"[SAVED] {artifact_path}")

    summary_path = args.output_dir / "final_indel_model_artifacts_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = args.output_dir / "final_indel_model_manifest.tsv"
    pd.DataFrame(
        [
            {
                "short_name": row["model_config"]["short_name"],
                "protein_name": row["model_config"]["protein_name"],
                "protein_group": row["model_config"].get("protein_group", ""),
                "model_task": row["model_config"]["model_task"],
                "prediction_transform": row["model_config"].get("prediction_transform", ""),
                "n_train": row["n_train"],
                "n_features": row["n_features"],
                "n_categorical_features": row["n_categorical_features"],
                "artifact_path": row["artifact_path"],
                "metadata_path": str(args.output_dir / row["model_config"]["short_name"] / f"{row['model_config']['short_name']}.final_tabpfn_indel_model.metadata.json"),
                "training_tsv": row["model_config"]["training_tsv"],
            }
            for row in summary_rows
        ]
    ).to_csv(manifest_path, sep="\t", index=False)
    print(f"[DONE] {summary_path}")


if __name__ == "__main__":
    main()
