param(
    [string]$OutputsVolume = "agentic-rag-rl-outputs",
    [string]$RemotePath = "/grpo_tool_agent_react_v4",
    [string]$LocalDestination = (Join-Path (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path "training\outputs\modal_grpo_tool_agent_react_v4")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Get-Command modal -ErrorAction SilentlyContinue)) {
    throw "Command not found: modal. Install Modal CLI first: python -m pip install modal; modal setup"
}

$parent = Split-Path -Parent $LocalDestination
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

Write-Host "Downloading ${OutputsVolume}:$RemotePath -> $LocalDestination"
& modal volume get --force $OutputsVolume $RemotePath $LocalDestination
if ($LASTEXITCODE -ne 0) {
    throw "modal volume get failed with exit code $LASTEXITCODE"
}

Write-Host "Download finished: $LocalDestination"
