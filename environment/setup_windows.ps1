# Windows PowerShell 5.1 / PowerShell 7. Does not read data, train or create ZIPs.
[CmdletBinding()]
param()
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($env:OS -ne 'Windows_NT') {
    throw 'This script is for Windows x86_64. Use setup_server.sh on Linux.'
}
if (-not [Environment]::Is64BitOperatingSystem -or
    $env:PROCESSOR_ARCHITECTURE -eq 'ARM64' -or
    $env:PROCESSOR_ARCHITEW6432 -eq 'ARM64') {
    throw 'The pinned PyTorch wheel requires Windows x86_64 (not ARM64).'
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: winget install --id astral-sh.uv -e ; then reopen PowerShell.'
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot '.venv-server'
$pythonExe = Join-Path $venvRoot 'Scripts\python.exe'
$requirements = Join-Path $PSScriptRoot 'windows-requirements.txt'
$reportPath = Join-Path $projectRoot 'reports\local_environment.windows.json'

function Invoke-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw "uv failed with exit code $LASTEXITCODE" }
}

if (Test-Path -LiteralPath $venvRoot) {
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        throw 'Existing .venv-server is not a Windows environment. Preserve/move it before setup.'
    }
    & $pythonExe -c "import sys,platform; assert sys.version_info[:3] == (3,11,17); assert platform.system() == 'Windows'; assert platform.machine().lower() in ('amd64','x86_64')"
    if ($LASTEXITCODE -ne 0) { throw 'Existing environment differs from Python 3.11.17 / Windows x86_64; it was not replaced.' }
} else {
    Invoke-Uv python install 3.11.17 --no-bin
    Invoke-Uv venv --python 3.11.17 $venvRoot
}

Invoke-Uv pip install --python $pythonExe -r $requirements
Invoke-Uv pip check --python $pythonExe
$freeze = & uv pip freeze --python $pythonExe
if ($LASTEXITCODE -ne 0) { throw 'Could not record installed packages.' }
# Native output redirection in Windows PowerShell can write UTF-16; use UTF-8.
$freeze | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'windows-freeze.local.txt') -Encoding UTF8

if (-not $env:MPLCONFIGDIR) { $env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib' }
& $pythonExe (Join-Path $PSScriptRoot 'check_environment.py') --output $reportPath
if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed; inspect the report.' }
Write-Host 'Environment ready. Activate in PowerShell with:'
Write-Host ('. "' + (Join-Path $venvRoot 'Scripts\Activate.ps1') + '"')
Write-Host 'No dataset was read and no model or ZIP was generated.'
