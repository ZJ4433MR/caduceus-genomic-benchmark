#!/usr/bin/env bash
# Standalone import checks for a local (e.g. micromamba) Python environment.
# Activate your env first, or set PYTHON to the interpreter you want to test.
#
#   micromamba activate caduceus_env   # example
#   ./scripts/test_imports.sh
#
#   PYTHON=/path/to/python ./scripts/test_imports.sh

set -o pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_YML="${ENV_YML:-$SCRIPT_DIR/../caduceus_env.yml}"

echo "==> Testing imports with: $PYTHON"
echo

( "$PYTHON" -c "import torch; print('�?Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU Available:', torch.cuda.is_available())" || echo '�?Failed to import torch' )
( "$PYTHON" -c "import torchdata; print('�?torchdata version:', torchdata.__version__)" || echo '�?Failed to import torchdata' )
( "$PYTHON" -c "import torchmetrics; print('�?torchmetrics version:', torchmetrics.__version__)" || echo '�?Failed to import torchmetrics' )
( "$PYTHON" -c "import torchtext; print('�?torchtext version:', torchtext.__version__)" || echo '�?Failed to import torchtext' )
( "$PYTHON" -c "import flash_attn; print('�?flash_attn version:', flash_attn.__version__)" || echo '�?Failed to import flash_attn' )
( "$PYTHON" -c "import causal_conv1d; print('�?causal_conv1d version:', causal_conv1d.__version__)" || echo '�?Failed to import causal_conv1d' )
( "$PYTHON" -c "import mamba_ssm; print('�?mamba-ssm version:', mamba_ssm.__version__)" || echo '�?Failed to import mamba_ssm' )

echo '�?Test completed. Check above for any import failures.'
