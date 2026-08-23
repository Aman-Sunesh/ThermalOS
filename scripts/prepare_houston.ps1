param(
    [switch]$Refresh,
    [switch]$SkipFortyGuard,
    [switch]$SkipCensus,
    [int]$SatelliteSamplesPerArea = 15
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    $Py = "python"
}

Write-Host "============================================================"
Write-Host "ThermalOS - Houston development-city preparation"
Write-Host "============================================================"
Write-Host "Python: $Py"
Write-Host ""

if (-not $SkipFortyGuard) {
    Write-Host "[1/5] FortyGuard minimal Houston harvest"
    $harvestArgs = @("scripts\harvest_city.py", "--city", "houston", "--mode", "minimal")
    if ($Refresh) { $harvestArgs += "--refresh" }
    & $Py @harvestArgs
    if ($LASTEXITCODE -ne 0) { throw "Houston FortyGuard harvest failed." }

    Write-Host "[2/5] FortyGuard satellite-only morphology"
    $satArgs = @("scripts\harvest_satellite_only.py", "--city", "houston", "--samples-per-area", "$SatelliteSamplesPerArea")
    if ($Refresh) { $satArgs += "--refresh" }
    & $Py @satArgs
    if ($LASTEXITCODE -ne 0) { throw "Houston satellite harvest failed." }
} else {
    Write-Host "[1-2/5] Skipping FortyGuard by request. Existing interim files will be reused."
}

if (-not $SkipCensus) {
    Write-Host "[3/5] Census 2024 geography + ACS"
    & $Py "scripts\fetch_census_geography.py" "--city" "houston"
    if ($LASTEXITCODE -ne 0) { throw "Houston Census geography fetch failed." }
    & $Py "scripts\fetch_acs.py" "--city" "houston" "--geography" "block-group"
    if ($LASTEXITCODE -ne 0) { throw "Houston ACS block-group fetch failed." }
    & $Py "scripts\fetch_acs.py" "--city" "houston" "--geography" "tract"
    if ($LASTEXITCODE -ne 0) { throw "Houston ACS tract fetch failed." }
} else {
    Write-Host "[3/5] Skipping Census by request."
}

Write-Host "[4/5] Build Houston canonical development tile table"
& $Py "scripts\build_city_features.py" "--city" "houston"
if ($LASTEXITCODE -ne 0) { throw "Houston feature build failed." }

Write-Host "[5/5] Run Miami-Houston development transfer diagnostic"
& $Py "scripts\compare_cities.py" "--budget" "2000000"
if ($LASTEXITCODE -ne 0) { throw "Cross-city comparison failed." }

Write-Host ""
Write-Host "Houston development preparation complete."
Write-Host "Review data\processed\houston_provenance.json before reporting real-data results."
Write-Host "Optional next enrichment: Houston METRO static GTFS from the official METRO portal."
