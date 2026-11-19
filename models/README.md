# Model artifacts

Each model directory contains a small `*.metadata.json` (tracked by git) and a
large `*.joblib` artifact (distributed through the GitHub Release).

## Base-editor models

| Display name | Directory | Artifact | Prediction target |
|---|---|---|---|
| ABE8e-nSpCas9 | `be/A` | `A.final_tabpfn_be_model.joblib` | BE efficiency, 0–1 |
| ABE8e-nSpCas9-NG | `be/B` | `B.final_tabpfn_be_model.joblib` | BE efficiency, 0–1 |
| ABEmax-nSpRY | `be/C` | `C.final_tabpfn_be_model.joblib` | BE efficiency, 0–1 |
| BE4max | `be/D` | `D.final_tabpfn_be_model.joblib` | BE efficiency, 0–1 |
| rAPOBEC1-BE3 | `be/E` | `E.final_tabpfn_be_model.joblib` | BE efficiency, 0–1 |
| QBEmax | `be/G` | `G.final_tabpfn_be_model.joblib` | BE efficiency, 0–1 |

BE inputs (sequence mode): `normalized_sgrna_name`, `sgrna_sequence`,
`target63_sequence` (20 nt upstream + 20 nt sgRNA + PAM3 + 20 nt downstream).

## Nuclease / indel models

| Display name | Directory | Artifact | Prediction target |
|---|---|---|---|
| SpCas9 | `indel/SpCas9` | `SpCas9.final_tabpfn_indel_model.joblib` | indel efficiency, 0–1 |
| OpenCRISPR-1 | `indel/OpenCRISPR` | `OpenCRISPR.final_tabpfn_indel_model.joblib` | indel efficiency, 0–1 |
| hfCas12Max | `indel/hfCas12MAX` | `hfCas12MAX.final_tabpfn_indel_model.joblib` | indel efficiency, 0–1 |

SpCas9 / OpenCRISPR-1 inputs: `normalized_sgrna_name`, `sgrna_sequence`, `pam`,
`target63_sequence`. hfCas12Max inputs: `normalized_sgrna_name`,
`sgrna_sequence`, `target66_sequence` (`upstream20 + TTN PAM + sgRNA23 +
downstream20`).

`model_manifest.tsv` lists display names, short names, file sizes, SHA-256
checksums, training sizes, feature counts and out-of-fold metrics.
