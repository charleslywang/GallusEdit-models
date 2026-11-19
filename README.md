# GallusEdit models

Prediction models for CRISPR base-editing and nuclease indel editing efficiency.
GallusEdit contains nine final models: three adenine base editors (ABE), three
cytosine base editors (CBE) and three nuclease/indel editors.

All models are fitted TabPFN v2.5 regressors. Each artifact stores the fitted
model, the feature-column order, categorical feature indices, imputation values
and the training metadata.

## Models

### Nuclease (indel) models

| Display name | Directory / short name | Task | Input layout | Training guides | Features | OOF Pearson | OOF R² | OOF MAE |
|---|---|---|---|---:|---:|---:|---:|---:|
| SpCas9 | `models/indel/SpCas9` | SpCas9 indel editing efficiency | 20 nt sgRNA + 63 bp context + PAM3 | 10000 | 1057 | 0.8051 | 0.6405 | 0.1121 |
| OpenCRISPR-1 | `models/indel/OpenCRISPR` | OpenCRISPR-1 indel editing efficiency | 20 nt sgRNA + 63 bp context + PAM3 | 11218 | 1057 | 0.8100 | 0.6301 | 0.0955 |
| hfCas12Max | `models/indel/hfCas12MAX` | hfCas12Max indel editing efficiency | 23 nt sgRNA + 66 bp context, TTN PAM | 10000 | 1026 | 0.7918 | 0.6246 | 0.0507 |

### Adenine base-editor models

| Display name | Directory / short name | Type | Training guides | Features | OOF Pearson (raw) | OOF R² (raw) | OOF MAE (raw) |
|---|---|---|---:|---:|---:|---:|---:|
| ABE8e-nSpCas9 | `models/be/A` | ABE | 12518 | 570 | 0.8306 | 0.6837 | 0.0895 |
| ABE8e-nSpCas9-NG | `models/be/B` | ABE | 12515 | 570 | 0.8269 | 0.6639 | 0.0927 |
| ABEmax-nSpRY | `models/be/C` | ABE | 12519 | 570 | 0.8255 | 0.6325 | 0.0511 |

### Cytosine base-editor models

| Display name | Directory / short name | Type | Training guides | Features | OOF Pearson (raw) | OOF R² (raw) | OOF MAE (raw) |
|---|---|---|---:|---:|---:|---:|---:|
| BE4max | `models/be/D` | CBE | 12514 | 570 | 0.7477 | 0.5450 | 0.1163 |
| rAPOBEC1-BE3 | `models/be/E` | CBE | 12486 | 570 | 0.8070 | 0.6358 | 0.1052 |
| QBEmax | `models/be/G` | CBE | 12529 | 570 | 0.7165 | 0.5025 | 0.1163 |

BE metrics are out-of-fold predictions on the raw 0–1 efficiency scale.
Per-model metadata (training sizes, feature lists, provenance) is stored next to
each model file as `*.metadata.json`; a machine-readable summary is in
`models/model_manifest.tsv` (includes SHA-256 of every artifact).

## Repository layout

```text
.
├── models/
│   ├── be/{A,B,C,D,E,G}/            # BE metadata (.joblib downloaded from Releases)
│   ├── indel/{SpCas9,OpenCRISPR,hfCas12MAX}/
│   └── model_manifest.tsv           # names, sizes, SHA-256, metrics
├── registry/
│   ├── be_model_registry.json       # BE model registry
│   └── indel_model_registry.json    # nuclease/indel model registry
├── code/                            # sequence/structure feature builders
├── predict_be_efficiency.py         # BE inference entry point
├── predict_opencrispr_indel_efficiency.py  # indel inference entry point
├── examples/                        # example inputs + expected output
└── scripts/
    ├── download_models.py           # fetch Release assets into models/
    └── verify_models.py             # SHA-256 + metadata consistency check
```

## Download the model files

Model artifacts are attached to the `v1.0.0` GitHub Release (they are larger
than the 100 MB git file limit). With an authenticated `gh` or token:

```bash
export GH_TOKEN=<your_github_token>          # needs repo read access (private repo)
python scripts/download_models.py --tag v1.0.0
python scripts/verify_models.py
```

`download_models.py` places each file in the directory expected by the
inference scripts:

```text
models/be/A/A.final_tabpfn_be_model.joblib
models/indel/SpCas9/SpCas9.final_tabpfn_indel_model.joblib
...
```

## Environment

Validated environment: Python 3.11, `tabpfn==6.4.1`, `pandas==2.3.3`,
`numpy==2.3.5`, `scikit-learn==1.6.1`, ViennaRNA 2.7.2 (`RNAfold`), CUDA GPU.

```bash
conda create -n gallusedit python=3.11 -y
conda activate gallusedit
pip install -r requirements.txt
conda install -c bioconda viennarna=2.7.2 -y
```

The artifacts were fitted and serialized with CUDA, so loading them requires a
visible GPU (`CUDA_VISIBLE_DEVICES=0`). TabPFN v2.5 weights
(`tabpfn-v2.5-regressor-v2.5_default.ckpt`) must be available locally; pass
`--model-path /path/to/tabpfn-v2.5-regressor-v2.5_default.ckpt` if TabPFN does
not resolve them automatically.

## Quick start

Run from the repository root (the scripts import the `code` package).

### Base editors (all six BE models)

Input CSV/TSV columns: `normalized_sgrna_name,sgrna_sequence,target63_sequence`.

```bash
CUDA_VISIBLE_DEVICES=0 python predict_be_efficiency.py \
  --input examples/example_sequence_input.csv \
  --input-mode sequence \
  --output outputs/be_predictions.tsv \
  --saved-model-mode require \
  --device cuda
```

Return only the best-predicted editor per sgRNA:

```bash
CUDA_VISIBLE_DEVICES=0 python predict_be_efficiency.py \
  --input examples/example_sequence_input.csv \
  --input-mode sequence \
  --output outputs/best_be.tsv \
  --best-only --saved-model-mode require --device cuda
```

### Nuclease / indel models

SpCas9 and OpenCRISPR-1 input columns:
`normalized_sgrna_name,sgrna_sequence,pam,target63_sequence`
(`target63_sequence = upstream20 + sgRNA20 + PAM3 + downstream20`).

hfCas12Max input columns:
`normalized_sgrna_name,sgrna_sequence,target66_sequence`
(`target66_sequence = upstream20 + TTN PAM + sgRNA23 + downstream20`).

```bash
CUDA_VISIBLE_DEVICES=0 python predict_opencrispr_indel_efficiency.py \
  --input examples/example_opencrispr_indel_input.csv \
  --input-mode sequence \
  --models SpCas9 OpenCRISPR \
  --output outputs/spcas9_opencrispr_indel.tsv \
  --saved-model-mode require --device cuda

CUDA_VISIBLE_DEVICES=0 python predict_opencrispr_indel_efficiency.py \
  --input examples/example_hfcas12max_indel_input.csv \
  --input-mode sequence \
  --models hfCas12MAX \
  --output outputs/hfcas12max_indel.tsv \
  --saved-model-mode require --device cuda
```

Use `--models` with display names (`ABE8e-nSpCas9`, `BE4max`, ...) or with the
single-letter short names for BE models.

### Output fields

- `predicted_editing_efficiency` — BE efficiency, 0–1.
- `predicted_indel_efficiency` — indel efficiency, 0–1
  (`predicted_indel_efficiency_percent` for 0–100).
- `model_backend=saved_final_model` confirms the packaged artifact was loaded.

## Validation without a GPU

Metadata-only checks (no model loading):

```bash
python predict_be_efficiency.py \
  --input examples/example_sequence_input.csv \
  --input-mode sequence \
  --output outputs/validate.tsv \
  --saved-model-mode require --validate-only
```

## Notes and limitations

- These models predict editing efficiency for the specific editors and input
  layouts listed above; they are not a genome-wide off-target model.
- The BE models expect the 82 nt full sgRNA scaffold context used during
  training; the SpCas9/OpenCRISPR-1 indel models use a 76 nt core scaffold.
- Predictions outside the training sequence/PAM distribution should be treated
  as extrapolation.

## License

Models, metadata and code are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See `LICENSE`.

## Citation

If you use GallusEdit, please cite this repository and the accompanying paper
(see `CITATION.cff`).

---

## 中文说明

GallusEdit 包含 9 个最终模型：

- 核酸酶/indel 模型 3 个：SpCas9、OpenCRISPR-1、hfCas12Max
- 腺嘌呤碱基编辑器模型 3 个：ABE8e-nSpCas9、ABE8e-nSpCas9-NG、ABEmax-nSpRY
- 胞嘧啶碱基编辑器模型 3 个：BE4max、rAPOBEC1-BE3、QBEmax

模型文件超过 GitHub 单文件 100 MB 限制，统一放在 `v1.0.0` Release 附件中。
克隆仓库后执行：

```bash
export GH_TOKEN=<你的 GitHub Token>   # 私有仓库需要读权限
python scripts/download_models.py --tag v1.0.0
python scripts/verify_models.py
```

随后按上面的 Quick start 命令运行 BE 或 indel 预测。模型在 CUDA 环境训练并
序列化，加载时需要可见 GPU；同时需要本地 TabPFN v2.5 权重。
