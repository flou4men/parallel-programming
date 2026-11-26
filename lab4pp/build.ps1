# Сборка л/р №4 (CUDA) напрямую через nvcc.
# Запускать из "Developer PowerShell for Visual Studio", чтобы nvcc нашёл cl.exe.
param(
    [string]$Exe = "matrix_cuda.exe",
    [string]$Arch = "sm_89"
)
$ErrorActionPreference = "Stop"
nvcc -O2 -arch=$Arch main.cu -o $Exe
if (-not (Test-Path $Exe)) { throw "Build failed: $Exe was not created" }
"Built: $Exe ($Arch)"
