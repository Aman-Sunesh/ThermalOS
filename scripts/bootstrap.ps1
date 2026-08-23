$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not (Test-Path ".venv")) { python -m venv .venv }
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
& ".\.venv\Scripts\python.exe" ".\scripts\build_demo_dataset.py"
Write-Host "ThermalOS ready. Run:" -ForegroundColor Green
Write-Host ".\.venv\Scripts\python.exe -m streamlit run app.py"
