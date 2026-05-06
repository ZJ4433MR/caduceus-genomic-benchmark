#!/usr/bin/env bash
set -euo pipefail

export SEEDS="1 2 3 4 5"
export EPOCHS=10
export OUTPUT_ROOT="outputs/downstream"
export RUN_PREFIX="paper_setting_mouse_enhancers"

bash "$(dirname "${BASH_SOURCE[0]}")/run_mouse_enhancers_ph_ps_multiseed.sh"
