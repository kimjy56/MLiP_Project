# Run from a normal local PowerShell. Downloads only the pinned official base.
[CmdletBinding()]
param([string]$ExperimentId = "LM-LFM25-05")
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Existing .venv Python not found.' }
    & $pythonExe -c "import torch,transformers; assert transformers.__version__ == '4.57.6'; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'Transformers/CUDA check failed.' }
    if ((Test-Path -LiteralPath (Join-Path $projectRoot "artifacts\$ExperimentId")) -or
        (Test-Path -LiteralPath (Join-Path $projectRoot "project_log\experiments\$ExperimentId.json"))) {
        throw 'Experiment ID already exists. Pass a fresh -ExperimentId.'
    }
    $env:HF_HUB_DISABLE_XET = '1'
    $env:HF_HUB_OFFLINE = '0'
    $env:TRANSFORMERS_OFFLINE = '0'
    & $pythonExe tools/fetch_lfm25.py
    if ($LASTEXITCODE -ne 0) { throw 'LFM2.5 download failed; training not started.' }
    & $pythonExe -m unittest discover -s tests -p 'test_lfm_lora.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'LFM2 hybrid LoRA test failed.' }
    & $pythonExe -m unittest discover -s tests -p 'test_*channel*.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed; training not started.' }
    & $pythonExe experiments/train_noisy_channel.py --experiment-id $ExperimentId --device cuda --steps 600 --pretrained artifacts/pretrained/lfm2.5-350m --confusions-csv artifacts/INNER-CONFUSIONS-01/confusions.csv --confusions-record project_log/experiments/INNER-CONFUSIONS-01.json
    if ($LASTEXITCODE -ne 0) { throw 'Training failed; see the experiment JSON.' }
} finally {
    Pop-Location
}
