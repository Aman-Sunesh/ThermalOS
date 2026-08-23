param(
    [string]$City = "miami",
    [double]$Budget = 2000000,
    [int]$RobustnessScenarios = 8
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

function Run-Step([string]$Name, [string[]]$Args) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Name
    Write-Host "============================================================"
    & $Py @Args
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

Run-Step "0. Regression tests" @("-m", "pytest", "-q")
Run-Step "1. Decision Robustness Engine" @("scripts\run_robustness.py", "--city", $City, "--budget", "$Budget", "--scenarios", "$RobustnessScenarios", "--pool-size", "1000")
Run-Step "2. Policy Stress Lab" @("scripts\run_policy_stress_lab.py", "--city", $City, "--budget", "$Budget")
Run-Step "3. ThermalVerify" @("scripts\prepare_verification.py", "--city", $City, "--budget", "$Budget")
Run-Step "4. HeatOps + Cooling Access" @("scripts\run_heatops.py", "--city", $City, "--budget", "60000")
Run-Step "5. Thermal Copilot" @("scripts\run_copilot.py", "--city", $City, "--prompt", "Give me a `$3M balanced plan with at least 50% equity and no neighborhood over 40%.")
Run-Step "6. Evidence Ledger / Trust Center" @("scripts\audit_evidence.py", "--city", $City)
Run-Step "7. Capital Decision Dossier" @("scripts\generate_dossier.py", "--city", $City, "--budget", "$Budget")
Run-Step "8. Miami-Houston Cross-city Transfer" @("scripts\compare_cities.py", "--budget", "$Budget")

Write-Host ""
Write-Host "All advanced-layer smoke tests completed."
Write-Host "Adaptive learning is intentionally excluded because it requires reviewed post-deployment evidence."
