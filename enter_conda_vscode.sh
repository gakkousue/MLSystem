#!/usr/bin/env bash
# ===== Open new shell with conda env (Linux / VS Code) =====

CONDA_ENV_NAME="pytorch_3_11_14"

echo "Opening new shell with conda env: $CONDA_ENV_NAME"

bash --rcfile <(cat <<EOF
# conda を bash にロード（conda init 済み想定）
source "\$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
source "\$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true

conda activate $CONDA_ENV_NAME

echo "Conda environment activated: $CONDA_ENV_NAME"
EOF
)
