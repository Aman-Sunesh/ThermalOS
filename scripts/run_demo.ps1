$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "Run scripts\bootstrap.ps1 first." }
& $Py -m streamlit run app.py
