[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CheckpointDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$VerlDir = ""
)

$ErrorActionPreference = "Stop"
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$projectDir = (Resolve-Path (Join-Path $ScriptRoot "..")).Path
$VerlDir = if ($VerlDir) { $VerlDir } else { Join-Path $projectDir "verl" }
$actorDir = Join-Path $CheckpointDir "actor"

$command = "python -m verl.model_merger --local_dir `"$actorDir`" --target_dir `"$OutputDir`""
Write-Host $command
Push-Location $VerlDir
try {
    Invoke-Expression $command
}
finally {
    Pop-Location
}
