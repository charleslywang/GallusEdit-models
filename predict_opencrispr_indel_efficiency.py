#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from code.be_agent_common import apply_feature_preprocessor, fit_feature_preprocessor, read_table, sigmoid
from code.hfcas12max_indel_features import build_feature_frame as build_hfcas12_feature_frame
from code.opencrispr_indel_features import build_feature_frame as build_spcas9_feature_frame
from train_save_opencrispr_indel_model import select_indel_feature_columns


AGENT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = AGENT_ROOT / "registry" / "indel_model_registry.json"
DEFAULT_SAVED_MODEL_ROOT = AGENT_ROOT / "models" / "indel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict indel/nuclease editing efficiency for new sgRNAs.")
    parser.add_argument("--input", type=Path, required=True, help="CSV/TSV. Either feature table or sequence table.")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction TSV.")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "features", "sequence"],
        default="auto",
        help="features: table already contains model features; sequence: target63_sequence for SpCas9/OpenCRISPR, target66_sequence for hfCas12MAX (sgRNA and PAM optional).",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Original protein names. Default: all registered indel models.")
    parser.add_argument("--target-column", default="target_mean_target_editing_efficiency")
    parser.add_argument("--sample-weight-column", default="sample_weight")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fit-mode", default="fit_preprocessors", choices=["fit_preprocessors", "low_memory", "batched"])
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--predict-batch-size", type=int, default=500)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--saved-model-root", type=Path, default=DEFAULT_SAVED_MODEL_ROOT)
    parser.add_argument("--saved-model-mode", choices=["auto", "require", "never"], default="auto")
    parser.add_argument("--max-train-rows", type=int, default=0, help="Debug/smoke-test only. 0 uses all final training rows.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true", help="Build/validate feature matrices, but do not fit TabPFN.")
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
    by_protein_name = {model["protein_name"].casefold(): model for model in models}
    resolved = []
    missing = []
    for item in requested:
        model = by_protein_name.get(item.casefold())
        if model:
            resolved.append(model)
        else:
            missing.append(item)
    if missing:
        known = ", ".join([m["protein_name"] for m in models])
        raise ValueError(f"未知 indel 模型: {missing}; 可用模型: {known}")
    return resolved


def infer_input_mode(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    sequence_col_sets = [
        {"target63_sequence"},
        {"sequence_63nt"},
        {"target66_sequence"},
        {"sequence_66bp"},
        {"sgrna_sequence", "pam", "target63_sequence"},
        {"sgrna_sequence", "target66_sequence"},
        {"sgRNA", "PAM", "sequence_63nt"},
        {"sgRNA", "sequence_66bp"},
    ]
    if any(cols.issubset(df.columns) for cols in sequence_col_sets):
        return "sequence"
    return "features"


def saved_model_path(saved_model_root: Path, short_name: str) -> Path:
    return saved_model_root / short_name / f"{short_name}.final_tabpfn_indel_model.joblib"


def saved_model_metadata_path(saved_model_root: Path, short_name: str) -> Path:
    return saved_model_root / short_name / f"{short_name}.final_tabpfn_indel_model.metadata.json"


def load_saved_model_artifact(artifact_path: Path, metadata_path: Path) -> dict[str, Any]:
    fit_device = ""
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fit_device = str(metadata.get("fit_params", {}).get("device", ""))
    if fit_device.startswith("cuda"):
        import torch

    if fit_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "保存的 OpenCRISPR indel final model 是在 CUDA 环境下训练并序列化的，当前 Python 进程无法使用 CUDA，"
            f"因此不能加载: {artifact_path}. 请在可见 GPU 的环境中运行，例如先设置 CUDA_VISIBLE_DEVICES。"
        )
    return joblib.load(artifact_path)


def metadata_preprocessor(metadata: dict[str, Any]) -> dict[str, Any]:
    preprocessor = {
        "feature_cols": metadata["feature_cols"],
        "categorical_cols": metadata["categorical_cols"],
        "categorical_indices": [metadata["feature_cols"].index(c) for c in metadata["categorical_cols"]],
        "categorical_fills": {},
        "numeric_fills": {},
    }
    for col in metadata["feature_cols"]:
        if col in set(metadata["categorical_cols"]):
            preprocessor["categorical_fills"][col] = "NA"
        else:
            preprocessor["numeric_fills"][col] = 0.0
    return preprocessor


def validate_saved_metadata(metadata: dict[str, Any], model_config: dict[str, Any], metadata_path: Path) -> None:
    saved_config = metadata.get("model_config", {})
    checks = {
        "short_name": model_config.get("short_name"),
        "protein_group": model_config.get("protein_group"),
        "training_tsv": model_config.get("training_tsv"),
        "prediction_transform": model_config.get("prediction_transform"),
        "target_scale": model_config.get("target_scale"),
    }
    mismatches = []
    for key, expected in checks.items():
        if expected is None:
            continue
        observed = saved_config.get(key)
        if observed != expected:
            mismatches.append(f"{key}: saved={observed!r}, registry={expected!r}")
    if mismatches:
        raise RuntimeError(
            "保存的 indel final model metadata 与当前 registry 不一致，可能是旧模型。"
            f"请重新运行保存脚本覆盖该模型: {metadata_path}. mismatch: " + "; ".join(mismatches)
        )


def prepare_prediction_features(
    input_df: pd.DataFrame,
    *,
    input_mode: str,
    model_config: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    if input_mode == "features":
        pred_df = input_df.copy()
        pred_df["protein_group"] = model_config.get("protein_group", model_config["short_name"])
        pred_df["editor_type"] = model_config["editor_type"]
        return pred_df, "precomputed_features"
    builder = model_config.get("sequence_feature_builder", "spcas9_63nt_pam3")
    if builder == "hfcas12max_66nt_pam3":
        pred_df = build_hfcas12_feature_frame(
            input_df,
            protein_group=model_config.get("protein_group", model_config["short_name"]),
            editor_type=model_config["editor_type"],
        )
    elif builder == "spcas9_63nt_pam3":
        pred_df = build_spcas9_feature_frame(
            input_df,
            protein_group=model_config.get("protein_group", model_config["short_name"]),
            editor_type=model_config["editor_type"],
        )
    else:
        raise ValueError(f"未知 sequence_feature_builder: {builder}")
    return pred_df, "sequence_with_training_imputation"


def predict_in_batches(model: object, x: pd.DataFrame, batch_size: int) -> np.ndarray:
    if batch_size <= 0 or len(x) <= batch_size:
        return np.asarray(model.predict(x), dtype=float)
    preds = []
    for start in range(0, len(x), batch_size):
        preds.append(np.asarray(model.predict(x.iloc[start : start + batch_size]), dtype=float))
    return np.concatenate(preds, axis=0)


def transform_predictions(values: np.ndarray, model_config: dict[str, Any]) -> np.ndarray:
    transform = model_config.get("prediction_transform", "sigmoid")
    if transform == "sigmoid":
        return np.clip(sigmoid(values), 0.0, 1.0)
    if transform == "identity_clip":
        return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    raise ValueError(f"未知 prediction_transform: {transform}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("POSTHOG_DISABLED", "true")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    registry = load_registry()
    models = resolve_models(registry, args.models)
    model_path = resolve_model_path(registry, args.model_path)
    input_df = read_table(args.input)
    input_mode = infer_input_mode(input_df, args.input_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    validation_rows = []
    for model_config in models:
        short_name = model_config["short_name"]
        artifact_path = saved_model_path(args.saved_model_root, short_name)
        metadata_path = saved_model_metadata_path(args.saved_model_root, short_name)
        use_saved = args.saved_model_mode in {"auto", "require"} and artifact_path.exists()
        if args.saved_model_mode == "require" and not artifact_path.exists():
            raise FileNotFoundError(f"缺少保存好的 indel final model: {artifact_path}")

        pred_df, feature_mode = prepare_prediction_features(input_df, input_mode=input_mode, model_config=model_config)
        train_df = None
        train_x = None
        if use_saved:
            if args.validate_only:
                if not metadata_path.exists():
                    raise FileNotFoundError(f"缺少保存模型 metadata: {metadata_path}")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                validate_saved_metadata(metadata, model_config, metadata_path)
                preprocessor = metadata_preprocessor(metadata)
                pred_x, imputed_missing = apply_feature_preprocessor(pred_df, preprocessor)
                model = None
                n_train = int(metadata["n_train"])
                n_features = int(metadata["n_features"])
                n_categorical_features = int(metadata["n_categorical_features"])
            else:
                if metadata_path.exists():
                    validate_saved_metadata(json.loads(metadata_path.read_text(encoding="utf-8")), model_config, metadata_path)
                artifact = load_saved_model_artifact(artifact_path, metadata_path)
                model = artifact["model"]
                preprocessor = artifact["preprocessor"]
                pred_x, imputed_missing = apply_feature_preprocessor(pred_df, preprocessor)
                n_train = int(artifact["n_train"])
                n_features = int(artifact["n_features"])
                n_categorical_features = int(artifact["n_categorical_features"])
        else:
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
            pred_x, imputed_missing = apply_feature_preprocessor(pred_df, preprocessor)
            n_train = len(train_df)
            n_features = len(feature_cols)
            n_categorical_features = len(categorical_cols)
            model = None

        validation_rows.append(
            {
                "protein_name": model_config["protein_name"],
                "model_task": model_config["model_task"],
                "target_scale": model_config.get("target_scale", ""),
                "prediction_transform": model_config.get("prediction_transform", ""),
                "feature_mode": feature_mode,
                "model_backend": "saved_final_model" if use_saved else "fit_from_training_tsv",
                "saved_model_path": str(artifact_path) if use_saved else "",
                "n_train": n_train,
                "n_predict": len(pred_x),
                "n_features": n_features,
                "n_categorical_features": n_categorical_features,
                "missing_features_imputed_from_training": imputed_missing,
            }
        )
        if args.validate_only:
            continue

        if model is None:
            from tabpfn import TabPFNRegressor

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
            model.fit(train_x, y_train)

        pred_model_scale = predict_in_batches(model, pred_x, args.predict_batch_size)
        pred_raw = transform_predictions(pred_model_scale, model_config)
        names = (
            pred_df["normalized_sgrna_name"].astype(str).to_numpy()
            if "normalized_sgrna_name" in pred_df.columns
            else np.array([f"query_{i + 1}" for i in range(len(pred_df))])
        )
        for idx, name in enumerate(names):
            all_rows.append(
                {
                    "normalized_sgrna_name": name,
                    "protein_name": model_config["protein_name"],
                    "editor_type": model_config["editor_type"],
                    "model_task": model_config["model_task"],
                    "target_scale": model_config.get("target_scale", ""),
                    "prediction_transform": model_config.get("prediction_transform", ""),
                    "predicted_indel_efficiency": float(pred_raw[idx]),
                    "predicted_indel_efficiency_percent": float(pred_raw[idx] * 100.0),
                    "predicted_model_scale": float(pred_model_scale[idx]),
                    "feature_mode": feature_mode,
                    "model_backend": "saved_final_model" if use_saved else "fit_from_training_tsv",
                    "saved_model_path": str(artifact_path) if use_saved else "",
                    "missing_features_imputed_from_training": imputed_missing,
                    "training_tsv": model_config["training_tsv"],
                }
            )

    if args.validate_only:
        args.output.write_text(pd.DataFrame(validation_rows).to_csv(sep="\t", index=False), encoding="utf-8")
        return

    pd.DataFrame(all_rows).to_csv(args.output, sep="\t", index=False)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "input_mode": input_mode,
                "model_path": model_path,
                "saved_model_root": str(args.saved_model_root),
                "saved_model_mode": args.saved_model_mode,
                "models": validation_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
