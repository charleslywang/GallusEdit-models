#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from code.be_agent_common import (
    apply_feature_preprocessor,
    load_registry,
    prepare_train_predict_matrices,
    read_table,
    resolve_models,
    select_feature_columns,
    sigmoid,
)
from code.be_short50_features import build_short50_feature_frame
from code.be_short40_features import build_short40_feature_frame
from code.be_short30_features import build_short30_feature_frame
from code.be_sgrna20_features import build_sgrna20_feature_frame
from code.be_sequence_features import build_basic_feature_frame


DEFAULT_SAVED_MODEL_ROOT = Path(__file__).resolve().parent / "models" / "be"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict BE editing efficiency for new sgRNAs using the registered final BE TabPFN datasets."
    )
    parser.add_argument("--input", type=Path, required=True, help="CSV/TSV. Either feature table or sequence table.")
    parser.add_argument("--output", type=Path, required=True, help="Output prediction TSV.")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "features", "sequence", "short30", "short40", "short50", "sgrna20"],
        default="auto",
        help=(
            "features: table already contains model features; sequence: needs target63_sequence "
            "(sgrna_sequence optional); short30: needs sgrna_sequence/refs and short_context30_sequence/surro_target; "
            "short40: needs BE_Endo-style target40_sequence; short50: needs BE-Hive-style target50_sequence/Input; "
            "sgrna20: needs only 20 nt sgrna_sequence."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional registered protein names. Default: all registered BE models.",
    )
    parser.add_argument("--best-only", action="store_true", help="Only output the highest predicted BE per sgRNA.")
    parser.add_argument("--target-column", default="target_mean_target_editing_efficiency")
    parser.add_argument("--sample-weight-column", default="sample_weight")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fit-mode", default="fit_preprocessors", choices=["fit_preprocessors", "low_memory", "batched"])
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--n-preprocessing-jobs", type=int, default=1)
    parser.add_argument("--predict-batch-size", type=int, default=500)
    parser.add_argument("--model-path", default="")
    parser.add_argument(
        "--saved-model-root",
        type=Path,
        default=DEFAULT_SAVED_MODEL_ROOT,
        help="Directory containing saved final model artifacts.",
    )
    parser.add_argument(
        "--saved-model-mode",
        choices=["auto", "require", "never"],
        default="auto",
        help="auto uses saved final models when present; require fails if missing; never fits from training TSV.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Debug/smoke-test only. 0 uses all final training rows.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true", help="Build/validate feature matrices, but do not fit TabPFN.")
    return parser.parse_args()


def infer_input_mode(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    if {"target63_sequence"}.issubset(df.columns):
        return "sequence"
    if {"sgrna_sequence", "target63_sequence"}.issubset(df.columns):
        return "sequence"
    if {"sgrna_sequence", "short_context30_sequence"}.issubset(df.columns) or {"refs", "surro_target"}.issubset(df.columns):
        return "short30"
    if {"target40_sequence"}.issubset(df.columns) or {"short_context40_sequence"}.issubset(df.columns):
        return "short40"
    if {"target50_sequence"}.issubset(df.columns) or {"short_context50_sequence"}.issubset(df.columns) or {"Input"}.issubset(df.columns):
        return "short50"
    if {"sgrna_sequence"}.issubset(df.columns) or {"Reference"}.issubset(df.columns) or {"Sequence"}.issubset(df.columns):
        return "sgrna20"
    return "features"


def predict_in_batches(model: object, x: pd.DataFrame, batch_size: int) -> np.ndarray:
    import torch

    def is_cuda_oom(error: Exception) -> bool:
        text = f"{type(error).__name__}: {error}".lower()
        return "cuda" in text and ("outofmemory" in text or "out of memory" in text or "oom" in text)

    def predict_one_batch(batch: pd.DataFrame) -> np.ndarray:
        current_size = len(batch)
        while True:
            try:
                return np.asarray(model.predict(batch.iloc[:current_size]), dtype=float)
            except Exception as error:
                if current_size <= 1 or not is_cuda_oom(error):
                    raise
                next_size = max(1, current_size // 2)
                log(
                    f"[WARN] CUDA OOM at predict batch_size={current_size}; retry with batch_size={next_size}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if next_size == len(batch):
                    raise
                first = predict_one_batch(batch.iloc[:next_size])
                rest = predict_one_batch(batch.iloc[next_size:])
                return np.concatenate([first, rest], axis=0)

    if batch_size <= 0 or len(x) <= batch_size:
        return predict_one_batch(x)
    preds = []
    for start in range(0, len(x), batch_size):
        preds.append(predict_one_batch(x.iloc[start : start + batch_size]))
    return np.concatenate(preds, axis=0)


def resolve_model_path(registry: dict, arg_model_path: str) -> str:
    if arg_model_path.strip():
        return arg_model_path
    default_path = Path(registry.get("tabpfn_default_model_path", ""))
    if default_path.exists():
        return str(default_path)
    return "auto"


def prepare_prediction_features(
    input_df: pd.DataFrame,
    *,
    input_mode: str,
    model_config: dict,
    feature_cache: dict[tuple[str, str], tuple[pd.DataFrame, str]] | None = None,
    rnafold_feature_cache: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, str]:
    if input_mode == "features":
        pred_df = input_df.copy()
        pred_df["protein_group"] = model_config["short_name"]
        pred_df["editor_type"] = model_config["editor_type"]
        return pred_df, "precomputed_features"
    if input_mode == "short30":
        cache_key = (input_mode, model_config["editor_type"])
        if feature_cache is not None and cache_key in feature_cache:
            pred_df, feature_mode = feature_cache[cache_key]
            pred_df = pred_df.copy()
        else:
            pred_df = build_short30_feature_frame(
                input_df,
                protein_group=model_config["short_name"],
                editor_type=model_config["editor_type"],
            )
            feature_mode = "short30_sequence_with_training_imputation"
            if feature_cache is not None:
                feature_cache[cache_key] = (pred_df.copy(), feature_mode)
        pred_df["protein_group"] = model_config["short_name"]
        return pred_df, "short30_sequence_with_training_imputation"
    if input_mode == "short50":
        cache_key = (input_mode, model_config["editor_type"])
        if feature_cache is not None and cache_key in feature_cache:
            pred_df, feature_mode = feature_cache[cache_key]
            pred_df = pred_df.copy()
        else:
            pred_df = build_short50_feature_frame(
                input_df,
                protein_group=model_config["short_name"],
                editor_type=model_config["editor_type"],
            )
            feature_mode = "short50_sequence_with_training_imputation"
            if feature_cache is not None:
                feature_cache[cache_key] = (pred_df.copy(), feature_mode)
        pred_df["protein_group"] = model_config["short_name"]
        return pred_df, "short50_sequence_with_training_imputation"
    if input_mode == "short40":
        cache_key = (input_mode, model_config["editor_type"])
        if feature_cache is not None and cache_key in feature_cache:
            pred_df, feature_mode = feature_cache[cache_key]
            pred_df = pred_df.copy()
        else:
            pred_df = build_short40_feature_frame(
                input_df,
                protein_group=model_config["short_name"],
                editor_type=model_config["editor_type"],
            )
            feature_mode = "short40_sequence_with_training_imputation"
            if feature_cache is not None:
                feature_cache[cache_key] = (pred_df.copy(), feature_mode)
        pred_df["protein_group"] = model_config["short_name"]
        return pred_df, "short40_sequence_with_training_imputation"
    if input_mode == "sgrna20":
        cache_key = (input_mode, model_config["editor_type"])
        if feature_cache is not None and cache_key in feature_cache:
            pred_df, feature_mode = feature_cache[cache_key]
            pred_df = pred_df.copy()
        else:
            pred_df = build_sgrna20_feature_frame(
                input_df,
                protein_group=model_config["short_name"],
                editor_type=model_config["editor_type"],
            )
            feature_mode = "sgrna20_sequence_with_training_imputation"
            if feature_cache is not None:
                feature_cache[cache_key] = (pred_df.copy(), feature_mode)
        pred_df["protein_group"] = model_config["short_name"]
        return pred_df, "sgrna20_sequence_with_training_imputation"
    if input_mode == "sequence":
        cache_key = (input_mode, model_config["editor_type"])
        if feature_cache is not None and cache_key in feature_cache:
            pred_df, feature_mode = feature_cache[cache_key]
            pred_df = pred_df.copy()
            log(f"[MODEL {model_config['protein_name']}] reuse sequence features")
        else:
            pred_df = build_basic_feature_frame(
                input_df,
                protein_group=model_config["short_name"],
                editor_type=model_config["editor_type"],
                rnafold_feature_cache=rnafold_feature_cache,
            )
            feature_mode = "sequence_with_training_imputation"
            if feature_cache is not None:
                feature_cache[cache_key] = (pred_df, feature_mode)
        pred_df["protein_group"] = model_config["short_name"]
        return pred_df, feature_mode
    pred_df = build_basic_feature_frame(
        input_df,
        protein_group=model_config["short_name"],
        editor_type=model_config["editor_type"],
        rnafold_feature_cache=rnafold_feature_cache,
    )
    return pred_df, "sequence_with_training_imputation"


def saved_model_path(saved_model_root: Path, short_name: str) -> Path:
    return saved_model_root / short_name / f"{short_name}.final_tabpfn_be_model.joblib"


def saved_model_metadata_path(saved_model_root: Path, short_name: str) -> Path:
    return saved_model_root / short_name / f"{short_name}.final_tabpfn_be_model.metadata.json"


def load_saved_model_artifact(artifact_path: Path) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "保存的 BE final model 是在 CUDA 环境下训练并序列化的，当前 Python 进程无法使用 CUDA，"
            f"因此不能加载: {artifact_path}. 请在可见 GPU 的环境中运行，例如先设置 CUDA_VISIBLE_DEVICES。"
        )
    return joblib.load(artifact_path)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("POSTHOG_DISABLED", "true")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    registry = load_registry()
    model_path = resolve_model_path(registry, args.model_path)
    input_df = read_table(args.input)
    input_mode = infer_input_mode(input_df, args.input_mode)
    models = resolve_models(registry, args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    TabPFNRegressor = None
    if not args.validate_only:
        from tabpfn import TabPFNRegressor

    all_rows = []
    validation_rows = []
    feature_cache: dict[tuple[str, str], tuple[pd.DataFrame, str]] = {}
    rnafold_feature_cache: dict[str, dict[str, float]] = {}
    for model_config in models:
        log(f"[MODEL {model_config['protein_name']}] prepare features: mode={input_mode}")
        artifact_path = saved_model_path(args.saved_model_root, model_config["short_name"])
        metadata_path = saved_model_metadata_path(args.saved_model_root, model_config["short_name"])
        use_saved = args.saved_model_mode in {"auto", "require"} and artifact_path.exists()
        if args.saved_model_mode == "require" and not artifact_path.exists():
            raise FileNotFoundError(f"缺少保存好的 final model: {artifact_path}")

        training_path = Path(model_config["training_tsv"])
        pred_df, feature_mode = prepare_prediction_features(
            input_df,
            input_mode=input_mode,
            model_config=model_config,
            feature_cache=feature_cache,
            rnafold_feature_cache=rnafold_feature_cache,
        )

        if use_saved:
            if args.validate_only:
                if not metadata_path.exists():
                    raise FileNotFoundError(f"缺少保存模型 metadata: {metadata_path}")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                preprocessor = {
                    "feature_cols": metadata["feature_cols"],
                    "categorical_cols": metadata["categorical_cols"],
                    "categorical_indices": [
                        metadata["feature_cols"].index(c) for c in metadata["categorical_cols"]
                    ],
                    "categorical_fills": {},
                    "numeric_fills": {},
                }
                for col in metadata["feature_cols"]:
                    if col in set(metadata["categorical_cols"]):
                        preprocessor["categorical_fills"][col] = "NA"
                    else:
                        preprocessor["numeric_fills"][col] = 0.0
                pred_x, imputed_missing = apply_feature_preprocessor(pred_df, preprocessor)
                model = None
                n_train = int(metadata["n_train"])
                n_features = int(metadata["n_features"])
                n_categorical_features = int(metadata["n_categorical_features"])
                categorical_indices = preprocessor["categorical_indices"]
            else:
                log(f"[MODEL {model_config['protein_name']}] load saved model")
                artifact = load_saved_model_artifact(artifact_path)
                model = artifact["model"]
                preprocessor = artifact["preprocessor"]
                log(f"[MODEL {model_config['protein_name']}] apply preprocessor")
                pred_x, imputed_missing = apply_feature_preprocessor(pred_df, preprocessor)
                n_train = int(artifact["n_train"])
                n_features = int(artifact["n_features"])
                n_categorical_features = int(artifact["n_categorical_features"])
                categorical_indices = preprocessor["categorical_indices"]
            train_df = None
            train_x = None
        else:
            train_df = pd.read_csv(training_path, sep="\t", low_memory=False)
            if "protein_group" in train_df.columns:
                train_df = train_df[train_df["protein_group"] == model_config["short_name"]].copy()
            if train_df.empty:
                raise ValueError(f"{training_path} 中没有 protein_group={model_config['short_name']} 的训练数据")
            if args.target_column not in train_df.columns:
                raise ValueError(f"{training_path} 缺少 target column: {args.target_column}")
            train_df = train_df[train_df[args.target_column].notna()].copy().reset_index(drop=True)
            if args.max_train_rows > 0 and len(train_df) > args.max_train_rows:
                train_df = train_df.sample(n=args.max_train_rows, random_state=args.random_state).reset_index(drop=True)
            feature_cols, categorical_cols = select_feature_columns(
                train_df,
                args.target_column,
                sample_weight_col=args.sample_weight_column,
            )
            train_x, pred_x, categorical_indices, imputed_missing = prepare_train_predict_matrices(
                train_df,
                pred_df,
                feature_cols,
                categorical_cols,
            )
            n_train = len(train_df)
            n_features = len(feature_cols)
            n_categorical_features = len(categorical_cols)

        validation_rows.append(
            {
                "protein_name": model_config["protein_name"],
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

        if not use_saved:
            y_train = pd.to_numeric(train_df[args.target_column], errors="coerce").astype(float).to_numpy()
            model = TabPFNRegressor(
                model_path=model_path,
                device=args.device,
                fit_mode=args.fit_mode,
                n_estimators=args.n_estimators,
                n_preprocessing_jobs=args.n_preprocessing_jobs,
                categorical_features_indices=categorical_indices,
                random_state=args.random_state,
            )
            model.fit(train_x, y_train)
        log(f"[MODEL {model_config['protein_name']}] predict n={len(pred_x)}")
        pred_logit = predict_in_batches(model, pred_x, args.predict_batch_size)
        log(f"[MODEL {model_config['protein_name']}] collect predictions")
        pred_raw = np.clip(sigmoid(pred_logit), 0.0, 1.0)
        names = (
            pred_df["normalized_sgrna_name"].astype(str).to_numpy()
            if "normalized_sgrna_name" in pred_df.columns
            else np.array([f"query_{i + 1}" for i in range(len(pred_df))])
        )
        for idx, name in enumerate(names):
            all_rows.append(
                {
                    "protein_name": model_config["protein_name"],
                    "normalized_sgrna_name": name,
                    "editor_type": model_config["editor_type"],
                    "predicted_editing_efficiency": float(pred_raw[idx]),
                    "predicted_editing_efficiency_percent": float(pred_raw[idx] * 100.0),
                    "predicted_logit": float(pred_logit[idx]),
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

    out_df = pd.DataFrame(all_rows)
    if args.best_only:
        out_df = (
            out_df.sort_values(
                ["normalized_sgrna_name", "predicted_editing_efficiency"],
                ascending=[True, False],
            )
            .groupby("normalized_sgrna_name", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
    out_df.to_csv(args.output, sep="\t", index=False)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "input_mode": input_mode,
                "best_only": args.best_only,
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
