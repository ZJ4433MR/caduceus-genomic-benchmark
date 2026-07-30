#!/usr/bin/env python3
"""Build a deterministic anonymous AAAI code-and-data ZIP."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import validate_package


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUTPUT = DIST / "flipped-gari-aaai27-code-data.zip"
ARCHIVE_ROOT = Path("flipped-gari-aaai27-code-data")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_package.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_reported_results.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/test_window_readout.py")], check=True)

    files = sorted(validate_package.iter_package_files(), key=lambda item: item[1].as_posix())
    manifest_lines: list[str] = []
    payloads: list[tuple[Path, bytes]] = []
    for path, relative in files:
        data = path.read_bytes()
        payloads.append((relative, data))
        manifest_lines.append(f"{digest(data)}  {relative.as_posix()}")

    manifest = ("\n".join(manifest_lines) + "\n").encode("ascii")
    DIST.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, data in payloads:
            info = zipfile.ZipInfo((ARCHIVE_ROOT / relative).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
        info = zipfile.ZipInfo(
            (ARCHIVE_ROOT / "PACKAGE_MANIFEST.sha256").as_posix(),
            date_time=(2026, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        archive.writestr(info, manifest)

    print(f"ARCHIVE_READY path={OUTPUT} bytes={OUTPUT.stat().st_size} sha256={digest(OUTPUT.read_bytes())}")


if __name__ == "__main__":
    main()
