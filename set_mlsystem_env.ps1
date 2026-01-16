# ===== MLsystem environment setup (Windows / PowerShell) =====

# このスクリプト自身のディレクトリ
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# env_config.json の絶対パス
$EnvConfigPath = Join-Path $ScriptDir "env_config.json"
$EnvConfigPath = (Resolve-Path $EnvConfigPath).Path

# 環境変数を設定（現在のセッションのみ）
$env:MLSYSTEM_CONFIG = $EnvConfigPath

Write-Host "MLSYSTEM_CONFIG set to:"
Write-Host $env:MLSYSTEM_CONFIG
