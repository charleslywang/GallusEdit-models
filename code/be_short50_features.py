#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import pandas as pd

from .be_sequence_features import (
    STANDARD_SGRNA_SCAFFOLD_DNA,
    add_kmer_counts,
    add_positional_one_hot,
    add_sequence_summary,
    align_complementarity_features,
    clean_sequence,
    dna_to_rna,
    run_rnafold,
    self_complementarity_features,
)


def _get_first_present(input_row: pd.Series, names: tuple[str, ...]) -> object:
    for name in names:
        if name in input_row and pd.notna(input_row[name]):
            return input_row[name]
    raise ValueError(f"输入缺少列: {'/'.join(names)}")


def build_short50_feature_frame(
    input_df: pd.DataFrame,
    *,
    protein_group: str,
    editor_type: str,
) -> pd.DataFrame:
    """Build reliable features from BE-Hive-style 50 nt target contexts.

    BE-Hive defines the 50 nt sequence as positions -19..30, with spacer
    positions 1..20 at Python slice [20:40] and PAM at [40:43]. The current
    final BE models were trained with 63 bp target-context features, so this
    adapter emits sgRNA/RNA-structure features plus audit columns and lets the
    saved model preprocessor fill unavailable 63 bp/genomic features.
    """

    rows = []
    spacer_feature_cache: dict[str, dict[str, object]] = {}
    progress_every = int(os.environ.get("BE_AGENT_FEATURE_PROGRESS_EVERY", "1000"))
    editable_base = "A" if editor_type == "ABE" else "C"
    for idx, input_row in input_df.reset_index(drop=True).iterrows():
        if progress_every > 0 and (idx == 0 or (idx + 1) % progress_every == 0):
            print(
                f"[FEATURE short50 {editor_type}] {idx + 1}/{len(input_df)}",
                file=sys.stderr,
                flush=True,
            )
        target50 = clean_sequence(
            _get_first_present(input_row, ("target50_sequence", "short_context50_sequence", "Input")),
            name="target50_sequence/short_context50_sequence/Input",
        )
        if len(target50) != 50:
            raise ValueError(f"第 {idx + 1} 行 target50/Input 长度不是 50: {len(target50)}")
        sgrna = target50[20:40]
        pam = target50[40:43]
        row: dict[str, object] = {
            "normalized_sgrna_name": input_row.get("normalized_sgrna_name", f"short50_query_{idx + 1}"),
            "protein_group": protein_group,
            "editor_type": editor_type,
            "input_sgrna_sequence": sgrna,
            "input_target50_sequence": target50,
            "input_target50_upstream20": target50[:20],
            "input_target50_pam": pam,
            "input_target50_downstream7": target50[43:50],
            "target50_has_ngg_pam": int(len(pam) == 3 and pam[1:] == "GG"),
        }
        cache_key = f"{editor_type}:{sgrna}"
        if cache_key not in spacer_feature_cache:
            spacer_features: dict[str, object] = {}
            add_sequence_summary(spacer_features, "sgrna", sgrna)
            add_positional_one_hot(spacer_features, "sgrna", sgrna, 20)
            add_kmer_counts(spacer_features, "sgrna_2mer", sgrna, 2)
            add_kmer_counts(spacer_features, "sgrna_3mer", sgrna, 3)
            spacer_features["editable_base_count_sgrna"] = sgrna.count(editable_base)
            spacer_features.update(self_complementarity_features("sgrna_self_comp", sgrna))
            spacer_features.update(
                align_complementarity_features(
                    "spacer_scaffold",
                    dna_to_rna(sgrna),
                    dna_to_rna(STANDARD_SGRNA_SCAFFOLD_DNA),
                )
            )
            spacer_features.update(run_rnafold(dna_to_rna(sgrna + STANDARD_SGRNA_SCAFFOLD_DNA), spacer_len=len(sgrna)))
            spacer_feature_cache[cache_key] = spacer_features
        row.update(spacer_feature_cache[cache_key])
        rows.append(row)
    return pd.DataFrame(rows)
