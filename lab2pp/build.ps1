# Сборка л/р №2 (OpenMP) компилятором MSVC.
# Запускать из "Developer PowerShell for Visual Studio"
# (или из обычного PowerShell, если cl.exe уже есть в PATH).
param([string]$Exe = "matrix.exe")
$ErrorActionPreference = "Stop"
cl /nologo /EHsc /O2 /openmp /std:c++17 main.cpp /Fe:$Exe
if (-not (Test-Path $Exe)) { throw "Build failed: $Exe was not created" }
"Built: $Exe"
