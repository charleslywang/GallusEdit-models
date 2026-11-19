#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path
import os
import re
import subprocess
import sys

import pandas as pd


BASES = ("A", "C", "G", "T")
STANDARD_SGRNA_SCAFFOLD_DNA = (
    "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAG"
    "TGGCACCGAGTCGGTGCTTTTTT"
)


def clean_sequence(value: object, *, name: str) -> str:
    seq = str(value).strip().upper().replace("U", "T")
    if not seq:
        raise ValueError(f"{name} 为空")
    bad = sorted(set(seq) - set(BASES))
    if bad:
        raise ValueError(f"{name} 含有非 A/C/G/T 碱基: {bad}")
    return seq


def longest_homopolymer(seq: str) -> int:
    best = 0
    current = 0
    last = ""
    for base in seq:
        if base == last:
            current += 1
        else:
            current = 1
            last = base
        best = max(best, current)
    return best


def longest_true_run(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def dna_to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def reverse_complement_rna(seq: str) -> str:
    complement = {"A": "U", "U": "A", "C": "G", "G": "C"}
    return "".join(complement[base] for base in reversed(seq))


def align_complementarity_features(prefix: str, seq_a_rna: str, seq_b_rna: str) -> dict[str, float]:
    seq_b_rc = reverse_complement_rna(seq_b_rna)
    best_total = 0
    best_fraction = 0.0
    best_contiguous = 0
    best_shift = 0

    for shift in range(-(len(seq_a_rna) - 1), len(seq_b_rc)):
        total = 0
        contiguous = 0
        max_contiguous = 0
        overlap = 0
        for idx_a in range(len(seq_a_rna)):
            idx_b = idx_a + shift
            if idx_b < 0 or idx_b >= len(seq_b_rc):
                contiguous = 0
                continue
            overlap += 1
            if seq_a_rna[idx_a] == seq_b_rc[idx_b]:
                total += 1
                contiguous += 1
                max_contiguous = max(max_contiguous, contiguous)
            else:
                contiguous = 0
        if overlap == 0:
            continue
        fraction = total / overlap
        if (
            total > best_total
            or (total == best_total and max_contiguous > best_contiguous)
            or (total == best_total and max_contiguous == best_contiguous and fraction > best_fraction)
        ):
            best_total = total
            best_fraction = fraction
            best_contiguous = max_contiguous
            best_shift = shift

    return {
        f"{prefix}_max_total_matches": best_total,
        f"{prefix}_max_match_fraction": best_fraction,
        f"{prefix}_max_contiguous": best_contiguous,
        f"{prefix}_best_shift": best_shift,
    }


def self_complementarity_features(prefix: str, dna_seq: str) -> dict[str, float]:
    rna_seq = dna_to_rna(dna_seq)
    return align_complementarity_features(prefix, rna_seq, rna_seq)


def resolve_rnafold_bin() -> str:
    env_bin = os.environ.get("RNAFOLD_BIN", "").strip()
    if env_bin:
        return env_bin
    conda_bin = Path(sys.executable).resolve().parent / "RNAfold"
    if conda_bin.exists():
        return str(conda_bin)
    return "RNAfold"


def parse_dot_bracket(structure: str) -> list[int]:
    open_to_close = {"(": ")", "[": "]", "{": "}", "<": ">"}
    close_to_open = {value: key for key, value in open_to_close.items()}
    stacks = {char: [] for char in open_to_close}
    pairing = [-1] * len(structure)
    for idx, char in enumerate(structure):
        if char in open_to_close:
            stacks[char].append(idx)
        elif char in close_to_open:
            opener = close_to_open[char]
            if not stacks[opener]:
                raise ValueError(f"dot-bracket 不平衡: {structure}")
            left_idx = stacks[opener].pop()
            pairing[left_idx] = idx
            pairing[idx] = left_idx
    if any(stacks[char] for char in stacks):
        raise ValueError(f"dot-bracket 不平衡: {structure}")
    return pairing


def extract_energy_and_structure(structure_line: str) -> tuple[str, float]:
    match = re.match(r"^([().\[\]{}<>]+)\s+\(\s*([-+]?\d+(?:\.\d+)?)\)$", structure_line.strip())
    if not match:
        raise ValueError(f"无法解析 RNAfold 输出: {structure_line}")
    return match.group(1), float(match.group(2))


def rnafold_feature_from_output(full_rna_seq: str, structure_line: str, spacer_len: int) -> dict[str, float]:
    structure, mfe = extract_energy_and_structure(structure_line)
    if len(structure) != len(full_rna_seq):
        raise ValueError("RNAfold 输出结构长度与输入序列长度不一致")
    pairing = parse_dot_bracket(structure)
    paired = [partner != -1 for partner in pairing]

    spacer_paired = paired[:spacer_len]
    spacer_5prime = spacer_paired[:10]
    spacer_3prime = spacer_paired[-10:]
    junction_slice = paired[max(spacer_len - 5, 0): spacer_len + 5]
    spacer_partner_is_scaffold = [
        pairing[idx] >= spacer_len if pairing[idx] != -1 else False
        for idx in range(spacer_len)
    ]

    return {
        "full_sgrna_mfe": mfe,
        "full_sgrna_mfe_per_nt": mfe / len(full_rna_seq),
        "full_sgrna_paired_fraction": sum(paired) / len(paired),
        "spacer_paired_count": sum(spacer_paired),
        "spacer_paired_fraction": sum(spacer_paired) / spacer_len,
        "spacer_unpaired_fraction": 1.0 - (sum(spacer_paired) / spacer_len),
        "spacer_5prime_paired_fraction": sum(spacer_5prime) / len(spacer_5prime),
        "spacer_3prime_paired_fraction": sum(spacer_3prime) / len(spacer_3prime),
        "spacer_longest_paired_run": longest_true_run(spacer_paired),
        "spacer_to_scaffold_paired_count": sum(spacer_partner_is_scaffold),
        "spacer_to_scaffold_paired_fraction": sum(spacer_partner_is_scaffold) / spacer_len,
        "spacer_5prime_to_scaffold_paired_fraction": sum(spacer_partner_is_scaffold[:10]) / 10,
        "spacer_3prime_to_scaffold_paired_fraction": sum(spacer_partner_is_scaffold[-10:]) / 10,
        "junction_paired_fraction": sum(junction_slice) / len(junction_slice),
    }


def run_rnafold(full_rna_seq: str, spacer_len: int) -> dict[str, float]:
    rnafold_bin = resolve_rnafold_bin()
    proc = subprocess.run(
        [rnafold_bin, "--noPS"],
        input=full_rna_seq + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError(f"RNAfold 输出行数异常: 期望 2, 实际 {len(lines)}")
    if lines[0] != full_rna_seq:
        raise ValueError("RNAfold 返回序列顺序异常")
    return rnafold_feature_from_output(full_rna_seq, lines[1], spacer_len=spacer_len)


def add_sequence_summary(row: dict[str, object], prefix: str, seq: str) -> None:
    counts = Counter(seq)
    length = len(seq)
    for base in BASES:
        row[f"{prefix}_{base}_count"] = counts.get(base, 0)
        row[f"{prefix}_{base}_fraction"] = counts.get(base, 0) / length
    row[f"{prefix}_length"] = length
    row[f"{prefix}_gc_content"] = (counts.get("G", 0) + counts.get("C", 0)) / length
    row[f"{prefix}_longest_homopolymer"] = longest_homopolymer(seq)


def add_positional_one_hot(row: dict[str, object], prefix: str, seq: str, expected_len: int) -> None:
    for idx in range(1, expected_len + 1):
        base_at_pos = seq[idx - 1] if idx <= len(seq) else ""
        for base in BASES:
            row[f"{prefix}_pos{idx}_{base}"] = 1 if base_at_pos == base else 0


def add_kmer_counts(row: dict[str, object], prefix: str, seq: str, k: int) -> None:
    counts = Counter(seq[i : i + k] for i in range(0, max(len(seq) - k + 1, 0)))
    for kmer_tuple in product(BASES, repeat=k):
        kmer = "".join(kmer_tuple)
        row[f"{prefix}_{kmer}_count"] = counts.get(kmer, 0)


def add_window_features(row: dict[str, object], prefix: str, seq: str, center_len: int, editable_base: str) -> str:
    start = max((len(seq) - center_len) // 2, 0)
    window = seq[start : start + center_len]
    counts = Counter(window)
    row[f"{prefix}_center{center_len}_gc_content"] = (
        counts.get("G", 0) + counts.get("C", 0)
    ) / len(window)
    row[f"{prefix}_center{center_len}_editable_base_count"] = counts.get(editable_base, 0)
    row[f"{prefix}_center{center_len}_longest_homopolymer"] = longest_homopolymer(window)
    return window


def build_basic_feature_frame(
    input_df: pd.DataFrame,
    *,
    protein_group: str,
    editor_type: str,
    rnafold_feature_cache: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    rows = []
    editable_base = "A" if editor_type == "ABE" else "C"
    rnafold_cache = rnafold_feature_cache if rnafold_feature_cache is not None else {}
    for idx, input_row in input_df.reset_index(drop=True).iterrows():
        if "target63_sequence" not in input_row:
            raise ValueError("序列输入至少需要列: target63_sequence")
        target63 = clean_sequence(input_row["target63_sequence"], name="target63_sequence")
        if len(target63) != 63:
            raise ValueError(f"第 {idx + 1} 行 target63_sequence 长度不是 63: {len(target63)}")
        sgrna_value = input_row.get("sgrna_sequence")
        if sgrna_value is None or pd.isna(sgrna_value) or str(sgrna_value).strip() == "":
            sgrna = target63[20:40]
        else:
            sgrna = clean_sequence(sgrna_value, name="sgrna_sequence")
            if sgrna != target63[20:40]:
                raise ValueError(
                    f"第 {idx + 1} 行 sgrna_sequence 与 target63_sequence 第 21-40 位不一致"
                )
        if len(sgrna) != 20:
            raise ValueError(f"第 {idx + 1} 行 sgrna_sequence 长度不是 20: {len(sgrna)}")
        row: dict[str, object] = {
            "normalized_sgrna_name": input_row.get("normalized_sgrna_name", f"query_{idx + 1}"),
            "protein_group": protein_group,
            "editor_type": editor_type,
            "input_sgrna_sequence": sgrna,
            "input_target63_sequence": target63,
        }
        add_sequence_summary(row, "sgrna", sgrna)
        add_sequence_summary(row, "target63", target63)
        add_positional_one_hot(row, "sgrna", sgrna, 20)
        add_positional_one_hot(row, "target63", target63, 63)
        for center_len in (5, 7, 9):
            window = add_window_features(row, "target63", target63, center_len, editable_base)
            if center_len == 9:
                add_kmer_counts(row, "target63_center9_2mer", window, 2)
                add_kmer_counts(row, "target63_center9_3mer", window, 3)
        add_kmer_counts(row, "sgrna_2mer", sgrna, 2)
        add_kmer_counts(row, "sgrna_3mer", sgrna, 3)
        row["editable_base_count_sgrna"] = sgrna.count(editable_base)
        row["editable_base_count_target63"] = target63.count(editable_base)
        row.update(self_complementarity_features("sgrna_self_comp", sgrna))
        row.update(
            align_complementarity_features(
                "spacer_scaffold",
                dna_to_rna(sgrna),
                dna_to_rna(STANDARD_SGRNA_SCAFFOLD_DNA),
            )
        )
        if sgrna not in rnafold_cache:
            rnafold_cache[sgrna] = run_rnafold(
                dna_to_rna(sgrna + STANDARD_SGRNA_SCAFFOLD_DNA),
                spacer_len=len(sgrna),
            )
        row.update(rnafold_cache[sgrna])
        rows.append(row)
    return pd.DataFrame(rows)
