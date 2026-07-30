#!/usr/bin/env python3
"""Download the DNALongBench ETGP data from Harvard Dataverse.

The Dataverse file labels differ from the paths referenced by the benchmark
configs for tabular files and fasta files. This downloader preserves the
directory layout while saving tabular data with its original `.tsv` filename
and unpacking `hg19.fa.gz` / `hg38.fa.gz` to the `.fa` paths expected by the
configs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


DATAVERSE_API = "https://dataverse.harvard.edu/api"
DATASETS = {
    "etgp": "doi:10.7910/DVN/CTEQXX",
}
HEADERS = {"User-Agent": "Mozilla/5.0 DNALONGBENCH-downloader/1.0"}


def _urlopen_json(url: str, retries: int = 3):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch {url}") from last_exc


def _dataset_files(persistent_id: str):
    url = f"{DATAVERSE_API}/datasets/:persistentId/?persistentId={persistent_id}"
    payload = _urlopen_json(url)
    return payload["data"]["latestVersion"]["files"]


def _download(url: str, destination: Path, expected_size: int | None = None, retries: int = 3):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_size is not None and destination.stat().st_size == expected_size:
        print(f"skip {destination} ({expected_size} bytes)")
        return

    part = destination.with_suffix(destination.suffix + ".part")
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            print(f"download {url} -> {destination}")
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise IOError(
                    f"size mismatch for {destination}: expected {expected_size}, got {part.stat().st_size}"
                )
            part.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, IOError) as exc:
            last_exc = exc
            if part.exists():
                part.unlink()
            if attempt == retries:
                break
            print(f"retry {attempt}/{retries} after {exc}", file=sys.stderr)
            time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_exc


def _target_name(data_file: dict) -> tuple[str, int | None, bool]:
    if data_file.get("tabularData") and data_file.get("originalFileName"):
        return data_file["originalFileName"], data_file.get("originalFileSize"), True
    return data_file["filename"], data_file.get("filesize"), False


def _download_file(
    file_record: dict,
    root: Path,
    decompress_fasta: bool = True,
    skip_fasta_gz: bool = False,
):
    data_file = file_record["dataFile"]
    directory = file_record.get("directoryLabel") or data_file.get("directoryLabel") or ""
    filename, expected_size, needs_original = _target_name(data_file)
    if skip_fasta_gz and filename.endswith(".fa.gz"):
        print(f"skip fasta archive {directory}/{filename}")
        return
    target = root / directory / filename
    file_id = data_file["id"]
    url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    if needs_original:
        url += "?format=original"
    _download(url, target, expected_size)

    if decompress_fasta and target.name.endswith(".fa.gz"):
        fasta_target = target.with_suffix("")
        if fasta_target.exists() and fasta_target.stat().st_size > 0:
            print(f"skip decompress {fasta_target}")
            return
        print(f"decompress {target} -> {fasta_target}")
        with gzip.open(target, "rb") as src, open(fasta_target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/dnalongbench"),
        help="Destination data root.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(DATASETS),
        default=["etgp"],
        help="DNALongBench tasks to download.",
    )
    parser.add_argument(
        "--no-decompress",
        action="store_true",
        help="Download .fa.gz files but do not unpack them.",
    )
    parser.add_argument(
        "--skip-fasta-gz",
        action="store_true",
        help="Do not download large .fa.gz reference archives.",
    )
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        print(f"== {task} {DATASETS[task]} ==")
        for file_record in _dataset_files(DATASETS[task]):
            _download_file(
                file_record,
                args.root,
                decompress_fasta=not args.no_decompress,
                skip_fasta_gz=args.skip_fasta_gz,
            )
    print(f"done: {args.root}")


if __name__ == "__main__":
    main()
