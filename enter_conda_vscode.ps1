# ===== Open new VSCode PowerShell with conda env =====

$CondaEnvName = "pytorch_3_11_14"

Write-Host "Opening new PowerShell terminal with conda env: $CondaEnvName"

powershell -NoExit -Command "conda activate $CondaEnvName"
