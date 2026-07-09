#!/usr/bin/env bash
set -euo pipefail

python -m compileall -q src caduceus scripts train.py
bash scripts/test_imports.sh

