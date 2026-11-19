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


def build_short40_feature_frame(
    input_df: pd.DataFrame,
    *,
    protein_group: str,
    editor_type: str,
) -> pd.DataFrame:
    """Build reliable features from BE_Endo-style 40 nt target contexts."""

    rows = []
    spacer_feature_cache: dict[str, dict[str, object]] = {}
    progress_every = int(os.environ.get("BE_AGENT_FEATURE_PROGRESS_EVERY", "1000"))
    editable_base = "A" if editor_type == "ABE" else "C"
    for idx, input_row in input_df.reset_index(drop=True).iterrows():
        if progress_every > 0 and (idx == 0 or (idx + 1) % progress_every == 0):
            print(
                f"[FEATURE short40 {editor_type}] {idx + 1}/{len(input_df)}",
                file=sys.stderr,
                flush=True,
            )
        target40 = clean_sequence(
            _get_first_present(input_row, ("target40_sequence", "short_context40_sequence", "Sequence")),
            name="target40_sequence/short_context40_sequence/Sequence",
        )
        if len(target40) != 40:
            raise ValueError(f"第 {idx + 1} 行 target40/Sequence 长度不是 40: {len(target40)}")

        sgrna = target40[10:30]
        pam = target40[30:33]
        row: dict[str, object] = {
            "normalized_sgrna_name": input_row.get("normalized_sgrna_name", f"short40_query_{idx + 1}"),
            "protein_group": protein_group,
            "editor_type": editor_type,
            "input_sgrna_sequence": sgrna,
            "input_target40_sequence": target40,
            "input_target40_upstream10": target40[:10],
            "input_target40_pam": pam,
            "input_target40_downstream7": target40[33:40],
            "target40_has_ngg_pam": int(len(pam) == 3 and pam[1:] == "GG"),
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
