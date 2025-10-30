# Run script helper for Windows PowerShell
# Usage: open PowerShell, cd to this folder or double-click this script
# It will create a virtualenv (.venv) if missing, activate it, install requirements, then run the main script

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir
$venvPath = Join-Path $scriptDir ".venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment at $venvPath..."
    python -m venv $venvPath
}

# Activate venv for this PowerShell session (use temporary bypass if needed)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
& "$venvPath\Scripts\Activate.ps1"

# Install dependencies if any
if (Test-Path (Join-Path $scriptDir 'requirements.txt')) {
    Write-Host "Installing requirements..."
    pip install -r requirements.txt
}

# Run the optimization script (file in this folder)
Write-Host "Running optimasi_menu_ga_pso.py..."
python optimasi_menu_ga_pso.py

# Keep window open when double-clicked
Write-Host "Done. Press Enter to close..."
Read-Host | Out-Null
