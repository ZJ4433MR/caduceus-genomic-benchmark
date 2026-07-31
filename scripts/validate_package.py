#!/usr/bin/env python3
"""Validate completeness, anonymization, and archive-size boundaries."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bib", ".cfg", ".csv", ".json", ".md", ".py", ".sh", ".tex",
    ".toml", ".tsv", ".txt", ".yaml", ".yml",
}
FORBIDDEN_PATTERNS = {
    "Overleaf token": re.compile(r"\bolp_[A-Za-z0-9]{16,}\b"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})\b"),
    "local Windows profile": re.compile(r"(?i)[A-Z]:\\Users\\[^\\\s\"']+"),
    "cluster account": re.compile(r"\bu23133\b"),
    "cluster home": re.compile(r"/(?:share/home|ssdfs/datahome)/"),
    "private staging repository": re.compile(r"github\.com/ZJ4433MR", re.IGNORECASE),
}
EXCLUDED_PARTS = {
    ".git", ".cache", ".pytest_cache", "__pycache__", "dist", "outputs",
    "package_build",
}
REQUIRED = (
    "README.md",
    "INSTALLATION.md",
    "REPRODUCE.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "caduceus_env.yml",
    "docs/CHECKLIST_MAPPING.md",
    "docs/DATA_AND_LICENSES.md",
    "docs/HARDWARE.md",
    "docs/HYPERPARAMETERS.md",
    "docs/IMPLEMENTATION_MAP.md",
    "docs/LEGACY_IDENTIFIERS.md",
    "data/manifests/datasets.csv",
    "data/representative/README.md",
    "data/representative/etgp_metadata_subset.tsv",
    "data/synthetic/dna_classification.csv",
    "checkpoints/manifest.csv",
    "configs/model/flipped_gari_gb.yaml",
    "configs/model/flipped_gari_vep.yaml",
    "configs/model/flipped_gari_etgp.yaml",
    "configs/model/flipped_gari_lm_gb.yaml",
    "configs/model/flipped_gari_lm_vep.yaml",
    "configs/model/flipped_gari_lm_etgp.yaml",
    "configs/model/diagnostics/flipped_gari_vep_a0b0.yaml",
    "configs/model/diagnostics/flipped_gari_vep_a0b1.yaml",
    "configs/model/diagnostics/flipped_gari_vep_a1b0.yaml",
    "configs/model/diagnostics/flipped_gari_vep_a1b1.yaml",
    "results/README.md",
    "scripts/build_anonymous_archive.py",
    "scripts/launch/pretrain_flipped_gari.sh",
    "scripts/launch/run_genomicbenchmarks.sh",
    "scripts/launch/run_vep.sh",
    "scripts/launch/run_etgp.sh",
    "scripts/select_dnalongbench450k_epoch_by_global_val.py",
)


def iter_package_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS or part.startswith(".staging_") for part in relative.parts):
            continue
        yield path, relative


def validate_required(errors: list[str]) -> None:
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")


def validate_files(errors: list[str]) -> None:
    binary_checkpoint_suffixes = {".ckpt", ".pt", ".pth", ".safetensors"}
    for path, relative in iter_package_files():
        if path.suffix.lower() in binary_checkpoint_suffixes:
            errors.append(f"checkpoint binary must not be archived: {relative}")
        if path.stat().st_size > 20 * 1024 * 1024:
            errors.append(f"unexpected file larger than 20 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} found in {relative}")


def validate_csv(errors: list[str]) -> None:
    manifest = ROOT / "data/manifests/datasets.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4:
        errors.append(f"dataset manifest should contain 4 records, found {len(rows)}")
    if any(row.get("redistributed", "").lower() != "false" for row in rows):
        errors.append("third-party dataset manifest must mark redistributed=false")

    with (ROOT / "checkpoints/manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        checkpoints = list(csv.DictReader(handle))
    for row in checkpoints:
        digest = row.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"invalid checkpoint digest for {row.get('artifact_id')}")

    with (ROOT / "data/representative/etgp_metadata_subset.tsv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        representative = list(csv.DictReader(handle, delimiter="\t"))
    expected_pairs = {
        ("train", "positive"),
        ("train", "negative"),
        ("valid", "positive"),
        ("valid", "negative"),
        ("test", "positive"),
        ("test", "negative"),
    }
    observed_pairs = {(row.get("subset"), row.get("target")) for row in representative}
    if len(representative) != 6 or observed_pairs != expected_pairs:
        errors.append(
            "representative ETGP subset must contain one positive and one negative "
            "row from each of train, valid, and test"
        )


def validate_config_dependencies(errors: list[str]) -> None:
    """Ensure every concrete Hydra defaults reference resolves in the package."""
    default_pattern = re.compile(r"^\s*-\s+(?:override\s+)?/([^:]+):\s*([^#]+)")
    for config_path in (ROOT / "configs").rglob("*.yaml"):
        for line in config_path.read_text(encoding="utf-8-sig").splitlines():
            match = default_pattern.match(line)
            if not match:
                continue
            group, raw_names = match.groups()
            raw_names = raw_names.strip()
            if raw_names in {"???", "null"}:
                continue
            if raw_names.startswith("[") and raw_names.endswith("]"):
                names = [name.strip() for name in raw_names[1:-1].split(",")]
            else:
                names = [raw_names]
            for name in names:
                dependency = ROOT / "configs" / group / f"{name}.yaml"
                if not dependency.is_file():
                    relative = config_path.relative_to(ROOT)
                    errors.append(f"missing Hydra dependency for {relative}: {dependency.relative_to(ROOT)}")


def main() -> None:
    errors: list[str] = []
    validate_required(errors)
    validate_files(errors)
    validate_csv(errors)
    validate_config_dependencies(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    count = sum(1 for _ in iter_package_files())
    print(f"PACKAGE_VALID files={count} anonymous_scan=passed")


if __name__ == "__main__":
    main()
