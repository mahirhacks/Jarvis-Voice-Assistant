# Jarvis 3.0 — Safe Launcher
# Sets memory allocator env vars before running Python to prevent
# MKL/OpenBLAS allocation failures on Windows.

$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
$env:CT2_NUM_THREADS = "1"
$env:TOKENIZERS_PARALLELISM = "false"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

& $python main.py
