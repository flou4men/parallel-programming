# Сборка л/р №5 компилятором MSVC.
param([string]$Exe = "mandelbrot.exe")
$ErrorActionPreference = "Stop"
cl /nologo /EHsc /O2 /std:c++17 main.cpp /Fe:$Exe
if (-not (Test-Path $Exe)) { throw "Build failed: $Exe was not created" }
"Built: $Exe"
