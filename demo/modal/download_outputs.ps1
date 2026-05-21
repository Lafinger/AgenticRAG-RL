param(
    [string]$OutputsVolume = "agentic-rag-rl-outputs",
    [string]$RemotePath = "/grpo_tool_agent_react_v4",
    [string]$LocalDestination = (Join-Path (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path "training\outputs\modal_grpo_tool_agent_react_v4"),
    [string]$ModalExe = "",
    [switch]$SkipHuggingFaceWeights
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

    $projectDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
    $venvModal = Join-Path $projectDir ".venv\Scripts\modal.exe"
    if (Test-Path -LiteralPath $venvModal) {
        return (Resolve-Path -LiteralPath $venvModal).Path
    }

    $command = Get-Command "modal" -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    throw "Command not found: modal. Install Modal CLI first: python -m pip install modal; modal setup"
}

$script:ModalCommand = Resolve-ModalExecutable $ModalExe
Write-Host "Using Modal CLI: $script:ModalCommand"

function Convert-RemotePathForModal {
    param([string]$Path)
    $normalized = $Path.Replace("\", "/")
    if (-not $normalized.StartsWith("/")) {
        $normalized = "/$normalized"
    }
    return $normalized
}

function Get-VolumeEntries {
    param([string]$Path)
    $json = & $script:ModalCommand volume ls --json $OutputsVolume (Convert-RemotePathForModal $Path)
    if ($LASTEXITCODE -ne 0) {
        throw "modal volume ls failed with exit code $LASTEXITCODE"
    }
    if (-not $json) {
        return @()
    }
    return @($json | ConvertFrom-Json)
}

function Get-VolumeFiles {
    param([string]$Path)
    $files = New-Object System.Collections.Generic.List[string]
    foreach ($entry in Get-VolumeEntries $Path) {
        if ($entry.Type -eq "dir") {
            foreach ($child in Get-VolumeFiles $entry.Filename) {
                $files.Add($child)
            }
        } elseif ($entry.Type -eq "file") {
            $files.Add($entry.Filename)
        }
    }
    return $files
}

function Invoke-RecursiveDownloadFallback {
    $remoteRoot = (Convert-RemotePathForModal $RemotePath).Trim("/")
    $files = Get-VolumeFiles $remoteRoot
    foreach ($file in $files) {
        if ($SkipHuggingFaceWeights -and $file -like "*/huggingface/model-*.safetensors") {
            Write-Host "Skipping duplicate HF weight: $file"
            continue
        }

        $relativePath = $file
        if ($relativePath -eq $remoteRoot) {
            $relativePath = Split-Path -Leaf $file
        } elseif ($relativePath.StartsWith("$remoteRoot/")) {
            $relativePath = $relativePath.Substring($remoteRoot.Length + 1)
        }
        $localPath = Join-Path $LocalDestination ($relativePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
        $localParent = Split-Path -Parent $localPath
        if ($localParent) {
            New-Item -ItemType Directory -Force -Path $localParent | Out-Null
        }
        Write-Host "Downloading $file -> $localPath"
        & $script:ModalCommand volume get --force $OutputsVolume (Convert-RemotePathForModal $file) $localPath
        if ($LASTEXITCODE -ne 0) {
            throw "modal volume get failed with exit code $LASTEXITCODE"
        }
    }
}

$parent = Split-Path -Parent $LocalDestination
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

Write-Host "Downloading ${OutputsVolume}:$RemotePath -> $LocalDestination"
$isWindowsPath = [System.IO.Path]::DirectorySeparatorChar -eq '\'
if ($isWindowsPath -or $SkipHuggingFaceWeights) {
    Write-Host "Using recursive file download fallback."
    Invoke-RecursiveDownloadFallback
} else {
    & $script:ModalCommand volume get --force $OutputsVolume $RemotePath $LocalDestination
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Directory download failed with exit code $LASTEXITCODE; falling back to recursive file download."
        Invoke-RecursiveDownloadFallback
    }
}

Write-Host "Download finished: $LocalDestination"
