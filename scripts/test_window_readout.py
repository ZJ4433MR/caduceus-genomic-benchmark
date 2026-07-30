#!/usr/bin/env python3
"""Test the VEP readout arithmetic without importing GPU dependencies."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vep_embeddings.py"


def load_bounds_function():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pooled_window_bounds"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, object] = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["pooled_window_bounds"]


def main() -> None:
    bounds = load_bounds_function()
    for width in (1, 2, 1024, 1536):
        start, end = bounds(width)
        assert end - start == width

    start, end = bounds(1024)
    indices = [offset + 512 for offset in range(start, end)]
    assert indices[0] == 0
    assert indices[-1] == 1023
    assert len(indices) == len(set(indices)) == 1024

    source = SOURCE.read_text(encoding="utf-8")
    assert "--window_size_bp" in source
    assert "Set --window_size_bp no larger than the downstream input length." in source
    print("WINDOW_READOUT_OK width=1024 unique_indices=1024")


if __name__ == "__main__":
    main()
