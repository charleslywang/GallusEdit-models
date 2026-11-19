#!/usr/bin/env python3
from __future__ import annotations

import itertools
import re
from collections import Counter

import pandas as pd


BASES = ["A", "C", "G", "T"]
DINUCS = ["".join(x) for x in itertools.product(BASES, repeat=2)]
TRINUCS = ["".join(x) for x in itertools.product(BASES, repeat=3)]
PAM_LEVELS = ["AGG", "CGG", "GGG", "TGG"]
EXPECTED_SGRNA_LEN = 20
EXPECTED_PAM_LEN = 3
EXPECTED_CONTEXT_LEN = 63
UPSTREAM_LEN = 20
PROTOSPACER_LEN = 20
PAM_LEN = 3
SPCAS9_CUT_AFTER_POS_1BASED = 37
SPCAS9_CUT_BEFORE_POS_1BASED = 38
SPCAS9_CUT_LEFT_INDEX0 = SPCAS9_CUT_AFTER_POS_1BASED - 1
STANDARD_SGRNA_SCAFFOLD_DNA = (
    "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGC"
)


def clean_dna(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).upper().replace("U", "T"))


def is_acgt(seq: str) -> bool:
    return bool(seq) and set(seq).issubset(BASES)


def gc_content(seq: str) -> float:
    seq = clean_dna(seq)
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def longest_homopolymer(seq: str) -> int:
    seq = clean_dna(seq)
    if not seq:
        return 0
    best = 1
    current = 1
    for idx in range(1, len(seq)):
        if seq[idx] == seq[idx - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def max_run(seq: str, base: str) -> int:
    runs = re.findall(f"{base}+", clean_dna(seq))
    return max((len(run) for run in runs), default=0)


def kmer_counts(seq: str, k: int) -> Counter[str]:
    seq = clean_dna(seq)
    counts: Counter[str] = Counter()
    if len(seq) < k:
        return counts
    for idx in range(len(seq) - k + 1):
        kmer = seq[idx : idx + k]
        if is_acgt(kmer):
            counts[kmer] += 1
    return counts


def add_basic_features(out: dict[str, object], prefix: str, seq: str) -> None:
    seq = clean_dna(seq)
    out[f"{prefix}_length"] = len(seq)
    out[f"{prefix}_gc_content"] = gc_content(seq)
    out[f"{prefix}_at_content"] = 1.0 - gc_content(seq) if seq else 0.0
    out[f"{prefix}_longest_homopolymer"] = longest_homopolymer(seq)
    for base in BASES:
        out[f"{prefix}_{base}_count"] = seq.count(base)
        out[f"{prefix}_{base}_fraction"] = seq.count(base) / len(seq) if seq else 0.0
        out[f"{prefix}_max_run_{base}"] = max_run(seq, base)


def add_position_onehot(out: dict[str, object], prefix: str, seq: str, expected_len: int) -> None:
    seq = clean_dna(seq)
    for idx in range(expected_len):
        base = seq[idx] if idx < len(seq) else "N"
        for dna_base in BASES:
            out[f"{prefix}_pos{idx + 1}_{dna_base}"] = int(base == dna_base)


def add_kmer_features(out: dict[str, object], prefix: str, seq: str, k: int) -> None:
    kmers = DINUCS if k == 2 else TRINUCS
    counts = kmer_counts(seq, k)
    for kmer in kmers:
        out[f"{prefix}_{k}mer_{kmer}_count"] = int(counts.get(kmer, 0))


def reverse_complement_dna(seq: str) -> str:
    return clean_dna(seq).translate(str.maketrans("ACGT", "TGCA"))[::-1]


def complementarity_features(prefix: str, seq_a: str, seq_b: str) -> dict[str, object]:
    a = clean_dna(seq_a).replace("T", "U")
    b_rc = reverse_complement_dna(seq_b).replace("T", "U")
    best_total = 0
    best_fraction = 0.0
    best_contiguous = 0
    best_shift = 0
    for shift in range(-(len(a) - 1), len(b_rc)):
        total = 0
        contiguous = 0
        max_contiguous = 0
        overlap = 0
        for idx_a, base in enumerate(a):
            idx_b = idx_a + shift
            if idx_b < 0 or idx_b >= len(b_rc):
                contiguous = 0
                continue
            overlap += 1
            if base == b_rc[idx_b]:
                total += 1
                contiguous += 1
                max_contiguous = max(max_contiguous, contiguous)
            else:
                contiguous = 0
        if overlap == 0:
            continue
        fraction = total / overlap
        if (total, max_contiguous, fraction) > (best_total, best_contiguous, best_fraction):
            best_total = total
            best_contiguous = max_contiguous
            best_fraction = fraction
            best_shift = shift
    return {
        f"{prefix}_max_total_matches": best_total,
        f"{prefix}_max_match_fraction": best_fraction,
        f"{prefix}_max_contiguous": best_contiguous,
        f"{prefix}_best_shift": best_shift,
    }


def window_around_left_cut_base(context63: str, window_size: int) -> str:
    half = window_size // 2
    start = SPCAS9_CUT_LEFT_INDEX0 - half
    end = start + window_size
    if start < 0 or end > len(context63):
        raise ValueError(f"cut window out of bounds: len={len(context63)}, window={window_size}")
    return context63[start:end]


def cut_microhomology_features(context63: str) -> dict[str, object]:
    seq = clean_dna(context63)
    left = seq[:SPCAS9_CUT_AFTER_POS_1BASED]
    right = seq[SPCAS9_CUT_AFTER_POS_1BASED:]
    out: dict[str, object] = {
        "cut_boundary_after_pos1based": SPCAS9_CUT_AFTER_POS_1BASED,
        "cut_boundary_before_pos1based": SPCAS9_CUT_BEFORE_POS_1BASED,
    }
    max_mh = 0
    total_mh = 0
    for length in range(1, 7):
        is_match = int(left[-length:] == right[:length])
        out[f"cut_left_right_mh_len{length}_match"] = is_match
        max_mh = max(max_mh, length if is_match else 0)
        total_mh += is_match
    out["cut_left_right_max_microhomology_len"] = max_mh
    out["cut_left_right_total_microhomology_hits_1to6"] = total_mh
    return out


def validate_sequences(row_id: str, sgrna: str, pam: str, context63: str) -> None:
    problems = []
    if len(sgrna) != EXPECTED_SGRNA_LEN or not is_acgt(sgrna):
        problems.append(f"sgRNA length/base invalid: {sgrna}")
    if len(pam) != EXPECTED_PAM_LEN or not is_acgt(pam):
        problems.append(f"PAM length/base invalid: {pam}")
    if len(context63) != EXPECTED_CONTEXT_LEN or not is_acgt(context63):
        problems.append(f"sequence_63nt length/base invalid: {context63}")
    if context63[UPSTREAM_LEN : UPSTREAM_LEN + PROTOSPACER_LEN] != sgrna:
        problems.append("sequence_63nt positions 21-40 must equal sgRNA")
    pam_start = UPSTREAM_LEN + PROTOSPACER_LEN
    if context63[pam_start : pam_start + PAM_LEN] != pam:
        problems.append("sequence_63nt positions 41-43 must equal PAM")
    if problems:
        raise ValueError(f"{row_id}: " + "; ".join(problems))


def build_sequence_features(
    row_id: str,
    sgrna: str,
    pam: str,
    context63: str,
    protein_group: str = "OpenCRISPR",
    editor_type: str = "OpenCRISPR_indel",
) -> dict[str, object]:
    sgrna = clean_dna(sgrna)
    pam = clean_dna(pam)
    context63 = clean_dna(context63)
    validate_sequences(row_id, sgrna, pam, context63)
    seed8 = sgrna[-8:]
    seed10 = sgrna[-10:]
    protospacer = context63[20:40]
    pam_region = context63[40:43]
    out: dict[str, object] = {
        "normalized_sgrna_name": str(row_id),
        "protein_group": protein_group,
        "editor_type": editor_type,
        "target63_layout": "upstream20_protospacer20_pam3_downstream20",
        "pam_gc_content": gc_content(pam),
        "pam_is_ngg": int(len(pam) == 3 and pam[1:] == "GG"),
    }
    for level in PAM_LEVELS:
        out[f"pam_{level}"] = int(pam == level)
    add_position_onehot(out, "pam", pam, EXPECTED_PAM_LEN)
    add_basic_features(out, "sgrna", sgrna)
    add_position_onehot(out, "sgrna", sgrna, EXPECTED_SGRNA_LEN)
    add_kmer_features(out, "sgrna", sgrna, 2)
    add_kmer_features(out, "sgrna", sgrna, 3)
    add_basic_features(out, "seed8", seed8)
    add_position_onehot(out, "seed8", seed8, 8)
    add_kmer_features(out, "seed8", seed8, 2)
    add_basic_features(out, "seed10", seed10)
    add_position_onehot(out, "seed10", seed10, 10)
    add_kmer_features(out, "seed10", seed10, 2)
    add_basic_features(out, "target63", context63)
    add_position_onehot(out, "target63", context63, EXPECTED_CONTEXT_LEN)
    add_basic_features(out, "protospacer20", protospacer)
    add_kmer_features(out, "protospacer20", protospacer, 2)
    add_kmer_features(out, "protospacer20", protospacer, 3)
    add_basic_features(out, "pam_region", pam_region)
    for window_size in [5, 7, 9, 15]:
        cut_window = window_around_left_cut_base(context63, window_size)
        prefix = f"cut_window{window_size}"
        add_basic_features(out, prefix, cut_window)
        add_kmer_features(out, prefix, cut_window, 2)
        if window_size >= 7:
            add_kmer_features(out, prefix, cut_window, 3)
    out.update(cut_microhomology_features(context63))
    out.update(complementarity_features("sgrna_self_comp", sgrna, sgrna))
    out.update(complementarity_features("spacer_scaffold", sgrna, STANDARD_SGRNA_SCAFFOLD_DNA))
    out["spacer_polyT_4plus"] = int("TTTT" in sgrna)
    out["seed10_polyT_4plus"] = int("TTTT" in seed10)
    out["sgrna_gc_lt_30pct"] = int(gc_content(sgrna) < 0.30)
    out["sgrna_gc_gt_70pct"] = int(gc_content(sgrna) > 0.70)
    out["seed10_gc_lt_30pct"] = int(gc_content(seed10) < 0.30)
    out["seed10_gc_gt_70pct"] = int(gc_content(seed10) > 0.70)
    return out


def get_value(row: pd.Series, names: list[str]) -> object:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    raise ValueError(f"输入缺少列，候选列名: {names}")


def build_feature_frame(input_df: pd.DataFrame, *, protein_group: str = "OpenCRISPR", editor_type: str = "OpenCRISPR_indel") -> pd.DataFrame:
    rows = []
    for idx, row in input_df.reset_index(drop=True).iterrows():
        row_id = row.get("normalized_sgrna_name", row.get("ID", f"query_{idx + 1}"))
        sgrna = get_value(row, ["sgrna_sequence", "sgRNA", "sgRNA_sequence", "sgrna", "spacer"])
        pam = get_value(row, ["pam", "PAM"])
        context63 = get_value(row, ["target63_sequence", "sequence_63nt", "sequence_63bp", "context63", "target_context_63nt_sequence"])
        rows.append(
            build_sequence_features(
                row_id=str(row_id),
                sgrna=str(sgrna),
                pam=str(pam),
                context63=str(context63),
                protein_group=protein_group,
                editor_type=editor_type,
            )
        )
    return pd.DataFrame(rows)
