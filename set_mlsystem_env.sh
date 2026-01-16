#!/usr/bin/env bash
# ===== MLsystem environment setup (Linux/macOS) =====

# このスクリプトのあるディレクトリ
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# env_config.json の絶対パス
export MLSYSTEM_CONFIG="$SCRIPT_DIR/env_config.json"

echo "MLSYSTEM_CONFIG set to:"
echo "$MLSYSTEM_CONFIG"
