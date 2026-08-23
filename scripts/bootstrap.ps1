$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not (Test-Path ".venv")) { python -m venv .venv }

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed with exit code $LASTEXITCODE"
    }
}

$Py = ".\.venv\Scripts\python.exe"

& $Py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE"
}

& $Py -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    throw "ThermalOS dependency installation failed with exit code $LASTEXITCODE"
}

if (
    -not (Test-Path ".\data\sample\miami_demo_tiles.csv") -or
    -not (Test-Path ".\data\sample\houston_demo_tiles.csv")
) {
    & $Py ".\scripts\build_demo_dataset.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Demo dataset generation failed with exit code $LASTEXITCODE"
    }
}

Write-Host "ThermalOS ready. Run:" -ForegroundColor Green
Write-Host "$Py -m streamlit run app.py"
