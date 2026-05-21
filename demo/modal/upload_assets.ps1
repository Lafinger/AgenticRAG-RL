param(
    [string]$DataVolume = "agentic-rag-rl-data",
    [string]$ModelsVolume = "agentic-rag-rl-models",
    [string]$OutputsVolume = "agentic-rag-rl-outputs",
    [string]$ProjectDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path,
    [string]$TrainParquet = "data\novel_eval\grpo_agentic_train.parquet",
    [string]$ValParquet = "data\novel_eval\grpo_agentic_val.parquet",
    [string]$IndexDir = "data\novel\indexes",
    [string]$BgeModelDir = "models\bge-m3",
    [string]$RerankerModelDir = "models\bge-reranker-v2-m3",
    [string]$MergedModelDir = "models\Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged",
    [string]$ModalExe = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

function Resolve-ModalExecutable {
    param([string]$ExplicitPath)
    if ($ExplicitPath) {
        if (Test-Path -LiteralPath $ExplicitPath) {
            return (Resolve-Path -LiteralPath $ExplicitPath).Path
        }
        $explicitCommand = Get-Command $ExplicitPath -ErrorAction SilentlyContinue
        if ($explicitCommand) {
            return $explicitCommand.Source
        }
        throw "Command not found: $ExplicitPath. Install Modal CLI first: python -m pip install modal; modal setup"
    }

    $venvModal = Join-Path $ProjectDir ".venv\Scripts\modal.exe"
    if (Test-Path -LiteralPath $venvModal) {
        return (Resolve-Path -LiteralPath $venvModal).Path
    }

    $command = Get-Command "modal" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw "Command not found: modal. Install Modal CLI first: python -m pip install modal; modal setup"
}

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return (Resolve-Path -LiteralPath $Path).Path
    }
    return (Resolve-Path -LiteralPath (Join-Path $ProjectDir $Path)).Path
}

function Invoke-Modal {
    param([string[]]$ModalArgs)
    & $script:ModalCommand @ModalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "modal $($ModalArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Ensure-Volume {
    param([string]$Name)
    Write-Host "Ensuring Modal Volume: $Name"
    & $script:ModalCommand volume create $Name
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "modal volume create $Name returned exit code $LASTEXITCODE. If the volume already exists, this is expected; otherwise fix Modal authentication or environment first."
    }
}

function Upload-Path {
    param(
        [string]$Volume,
        [string]$LocalPath,
        [string]$RemotePath
    )
    $resolved = Resolve-ProjectPath $LocalPath
    if (-not (Test-Path -LiteralPath $resolved)) {
        throw "Local path does not exist: $resolved"
    }
    Write-Host "Uploading $resolved -> ${Volume}:$RemotePath"
    Invoke-Modal -ModalArgs @("volume", "put", "--force", $Volume, $resolved, $RemotePath)
}

$script:ModalCommand = Resolve-ModalExecutable $ModalExe
Write-Host "Using Modal CLI: $script:ModalCommand"

Ensure-Volume $DataVolume
Ensure-Volume $ModelsVolume
Ensure-Volume $OutputsVolume

Upload-Path $DataVolume $TrainParquet "/novel_eval/grpo_agentic_train.parquet"
Upload-Path $DataVolume $ValParquet "/novel_eval/grpo_agentic_val.parquet"
Upload-Path $DataVolume $IndexDir "/novel/indexes"

Upload-Path $ModelsVolume $BgeModelDir "/bge-m3"
Upload-Path $ModelsVolume $RerankerModelDir "/bge-reranker-v2-m3"
Upload-Path $ModelsVolume $MergedModelDir "/Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged"

Write-Host "Modal asset upload finished."
Write-Host "Next check:"
Write-Host "  modal run .\demo\modal\modal_grpo_tool_agent.py::check_inputs"
