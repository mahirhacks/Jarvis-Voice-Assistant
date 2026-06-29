# Jarvis 3.0 — first-time setup (Windows)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Creating virtual environment..."
python -m venv venv

$pip = Join-Path $PSScriptRoot "venv\Scripts\pip.exe"
$python = Join-Path $PSScriptRoot "venv\Scripts\python.exe"

Write-Host "Installing dependencies..."
& $pip install --upgrade pip
& $pip install -r requirements.txt

Write-Host "Installing Playwright Firefox..."
& $python -m playwright install firefox

if (-not (Test-Path "config\settings.local.json")) {
    Copy-Item "config\settings.local.json.example" "config\settings.local.json"
    Write-Host "Created config\settings.local.json — set your microphone device IDs."
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. ollama pull qwen3.5:4b"
Write-Host "  2. .\venv\Scripts\python.exe scripts\list_mics.py"
Write-Host "  3. Edit config\settings.local.json with your mic index"
Write-Host "  4. .\run_jarvis.ps1"
