#!/usr/bin/env python3
from __future__ import annotations

import itertools
import re
from collections import Counter

import pandas as pd


BASES = ["A", "C", "G", "T"]
DINUCS = ["".join(x) for x in itertools.product(BASES, repeat=2)]
TRINUCS = ["".join(x) for x in itertools.product(BASES, repeat=3)]
EXPECTED_UPSTREAM_LEN = 20
EXPECTED_PAM_LEN = 3
EXPECTED_SGRNA_LEN = 23
EXPECTED_DOWNSTREAM_LEN = 20
EXPECTED_CONTEXT_LEN = 66


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
    denom = max(len(clean_dna(seq)) - k + 1, 1)
    for kmer in kmers:
        count = int(counts.get(kmer, 0))
        out[f"{prefix}_{k}mer_{kmer}_count"] = count
        out[f"{prefix}_{k}mer_{kmer}_fraction"] = count / denom


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


def validate_sequences(row_id: str, sgrna: str, context66: str) -> tuple[str, str, str, str]:
    sgrna = clean_dna(sgrna)
    context66 = clean_dna(context66)
    problems = []
    if sgrna and (len(sgrna) != EXPECTED_SGRNA_LEN or not is_acgt(sgrna)):
        problems.append(f"sgRNA length/base invalid: {sgrna}")
    if len(context66) != EXPECTED_CONTEXT_LEN or not is_acgt(context66):
        problems.append(f"sequence_66bp length/base invalid: {context66}")
    upstream = context66[:EXPECTED_UPSTREAM_LEN]
    pam = context66[EXPECTED_UPSTREAM_LEN : EXPECTED_UPSTREAM_LEN + EXPECTED_PAM_LEN]
    spacer = context66[EXPECTED_UPSTREAM_LEN + EXPECTED_PAM_LEN : EXPECTED_UPSTREAM_LEN + EXPECTED_PAM_LEN + EXPECTED_SGRNA_LEN]
    downstream = context66[-EXPECTED_DOWNSTREAM_LEN:]
    if not re.fullmatch(r"TT[ACGT]", pam):
        problems.append(f"PAM is not TTN: {pam}")
    if sgrna and spacer != sgrna:
        problems.append("sgRNA is not equal to sequence_66bp[24:46]")
    if problems:
        raise ValueError(f"{row_id}: " + "; ".join(problems))
    return upstream, pam, spacer, downstream


def build_sequence_features(
    row_id: str,
    sgrna: str,
    context66: str,
    *,
    protein_group: str = "haCas12Max",
    editor_type: str = "haCas12Max_indel",
) -> dict[str, object]:
    upstream, pam, spacer, downstream = validate_sequences(row_id, sgrna, context66)
    context66 = clean_dna(context66)
    seed8 = spacer[:8]
    seed10 = spacer[:10]
    seed12 = spacer[:12]
    pam_proximal_window = pam + spacer[:13]
    pam_distal_window = spacer[-12:] + downstream[:4]
    out: dict[str, object] = {
        "normalized_sgrna_name": str(row_id),
        "protein_group": protein_group,
        "editor_type": editor_type,
        "pam_sequence_is_ttn": int(bool(re.fullmatch(r"TT[ACGT]", pam))),
        "pam_gc_content": gc_content(pam),
        "pam_pos1_is_T": int(pam[0] == "T"),
        "pam_pos3_is_G": int(pam[2] == "G"),
        "pam_pos3_is_C_or_G": int(pam[2] in {"C", "G"}),
    }
    add_position_onehot(out, "pam", pam, EXPECTED_PAM_LEN)
    add_basic_features(out, "pam", pam)
    add_kmer_features(out, "pam", pam, 2)
    add_kmer_features(out, "pam", pam, 3)
    for prefix, seq, expected_len in [
        ("sgrna", spacer, EXPECTED_SGRNA_LEN),
        ("seed8", seed8, 8),
        ("seed10", seed10, 10),
        ("seed12", seed12, 12),
        ("upstream20", upstream, EXPECTED_UPSTREAM_LEN),
        ("downstream20", downstream, EXPECTED_DOWNSTREAM_LEN),
        ("target66", context66, EXPECTED_CONTEXT_LEN),
        ("pam_proximal16", pam_proximal_window, 16),
        ("pam_distal16", pam_distal_window, 16),
    ]:
        add_basic_features(out, prefix, seq)
        add_position_onehot(out, prefix, seq, expected_len)
        add_kmer_features(out, prefix, seq, 2)
        if len(seq) >= 3:
            add_kmer_features(out, prefix, seq, 3)
    out.update(complementarity_features("sgrna_self_comp", spacer, spacer))
    out["spacer_polyT_4plus"] = int("TTTT" in spacer)
    out["seed10_polyT_4plus"] = int("TTTT" in seed10)
    out["sgrna_gc_lt_30pct"] = int(gc_content(spacer) < 0.30)
    out["sgrna_gc_gt_70pct"] = int(gc_content(spacer) > 0.70)
    out["seed10_gc_lt_30pct"] = int(gc_content(seed10) < 0.30)
    out["seed10_gc_gt_70pct"] = int(gc_content(seed10) > 0.70)
    return out


def get_value(row: pd.Series, names: list[str]) -> object:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    raise ValueError(f"输入缺少列，候选列名: {names}")


def optional_value(row: pd.Series, names: list[str]) -> str:
    for name in names:
        if name in row and not pd.isna(row[name]) and str(row[name]).strip():
            return str(row[name])
    return ""


def build_feature_frame(input_df: pd.DataFrame, *, protein_group: str = "haCas12Max", editor_type: str = "haCas12Max_indel") -> pd.DataFrame:
    rows = []
    for idx, row in input_df.reset_index(drop=True).iterrows():
        row_id = row.get("normalized_sgrna_name", row.get("ID", f"query_{idx + 1}"))
        sgrna = optional_value(row, ["sgrna_sequence", "sgRNA", "sgRNA_sequence", "sgrna", "spacer"])
        context66 = get_value(row, ["target66_sequence", "sequence_66bp", "sequence_66nt", "context66", "target_context_66nt_sequence"])
        rows.append(
            build_sequence_features(
                row_id=str(row_id),
                sgrna=str(sgrna),
                context66=str(context66),
                protein_group=protein_group,
                editor_type=editor_type,
            )
        )
    return pd.DataFrame(rows)
