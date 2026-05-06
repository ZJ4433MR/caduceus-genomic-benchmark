#!/usr/bin/env bash
set -euo pipefail

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
elif command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  echo "conda: missing"
  exit 0
fi

conda env list
conda activate caduceus_env

python - <<'PY'
import importlib.util
import sys

print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch error", repr(exc))

for name in ["transformers", "genomic_benchmarks", "sklearn", "pandas", "mamba_ssm", "causal_conv1d"]:
    print(name, "ok" if importlib.util.find_spec(name) else "missing")
PY
