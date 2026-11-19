# GallusEdit models

GallusEdit provides nine models for predicting CRISPR editing efficiency:
three adenine base editors (ABE), three cytosine base editors (CBE) and three
nuclease/indel editors.

## Models

### Nuclease (indel) models

| Model | Task | Input layout |
|---|---|---|
| SpCas9 | Indel editing efficiency | 63 nt target sequence |
| OpenCRISPR-1 | Indel editing efficiency | 63 nt target sequence |
| hfCas12Max | Indel editing efficiency | 66 nt target sequence |

### Base-editor models (6)

| Model | Task | Input layout |
|---|---|---|
| ABE8e-nSpCas9 | Base editing efficiency | 63 nt target sequence |
| ABE8e-nSpCas9-NG | Base editing efficiency | 63 nt target sequence |
| ABEmax-nSpRY | Base editing efficiency | 63 nt target sequence |
| BE4max | Base editing efficiency | 63 nt target sequence |
| rAPOBEC1-BE3 | Base editing efficiency | 63 nt target sequence |
| QBEmax | Base editing efficiency | 63 nt target sequence |

Target sequence layout: 63 nt = 20 nt upstream + 20 nt sgRNA + PAM3 + 20 nt
downstream; 66 nt = 20 nt upstream + TTN PAM + 23 nt sgRNA + 20 nt downstream.

## Repository layout

```text
.
├── models/
│   ├── be/{A,B,C,D,E,G}/            # BE metadata (.joblib downloaded from Releases)
│   └── indel/{SpCas9,OpenCRISPR,hfCas12MAX}/
├── registry/                        # model registries
├── code/                            # sequence/structure feature builders
├── predict_be_efficiency.py         # BE inference entry point
├── predict_opencrispr_indel_efficiency.py  # indel inference entry point
├── examples/                        # example inputs
└── scripts/
    ├── download_models.py           # fetch Release assets into models/
    └── verify_models.py             # integrity check
```

## Install

```bash
conda create -n gallusedit python=3.11 -y
conda activate gallusedit
pip install -r requirements.txt
conda install -c conda-forge -c bioconda viennarna=2.7.2 -y
```

TabPFN is installed by `requirements.txt`, and its regressor weights are
downloaded automatically on first use. A CUDA GPU is required. ViennaRNA
provides the `RNAfold` executable used for base-editor sequence features
(install it with conda, as above).

## Download the model files

```bash
# GH_TOKEN is optional; set it only if you hit GitHub API rate limits
export GH_TOKEN=<your_github_token>
python scripts/download_models.py --tag v1.0.0
python scripts/verify_models.py
```

Each file is placed in the directory expected by the inference scripts.

## Usage

Run from the repository root.

### Base editors

Input column: `target63_sequence` (`normalized_sgrna_name` optional).

```bash
CUDA_VISIBLE_DEVICES=0 python predict_be_efficiency.py \
  --input examples/example_sequence_input.csv \
  --input-mode sequence \
  --output outputs/be_predictions.tsv \
  --saved-model-mode require --device cuda
```

Only the best-predicted editor per sgRNA:

```bash
CUDA_VISIBLE_DEVICES=0 python predict_be_efficiency.py \
  --input examples/example_sequence_input.csv \
  --input-mode sequence \
  --output outputs/best_be.tsv \
  --best-only --saved-model-mode require --device cuda
```

### Nuclease / indel models

SpCas9 and OpenCRISPR-1 input column: `target63_sequence`
(`upstream20 + sgRNA20 + PAM3 + downstream20`).

hfCas12Max input column: `target66_sequence`
(`upstream20 + TTN PAM + sgRNA23 + downstream20`).

`normalized_sgrna_name` is optional. `sgrna_sequence` / `pam` may also be
supplied; if present they must match the target sequence.

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

`--models` accepts display names (`ABE8e-nSpCas9`, `BE4max`, ...) or the
single-letter short names for BE models.

### Output

- `predicted_editing_efficiency` — BE efficiency, 0–1.
- `predicted_indel_efficiency` — indel efficiency, 0–1
  (`predicted_indel_efficiency_percent` for 0–100).

## License

Released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
See `LICENSE`.

## Citation

If you use GallusEdit, please cite this repository and the accompanying paper
(see `CITATION.cff`).

## Acknowledgements

The models are built on [TabPFN](https://github.com/PriorLabs/TabPFN). If you
use GallusEdit, please also cite Hollmann et al., *Nature* (2025).
