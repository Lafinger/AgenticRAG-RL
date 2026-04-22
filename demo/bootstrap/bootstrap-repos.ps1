[CmdletBinding()]
param(
    [string]$WorkspaceRoot = ""
)

$ErrorActionPreference = "Stop"
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $WorkspaceRoot) {
    $WorkspaceRoot = (Resolve-Path (Join-Path $ScriptRoot "..")).Path
}

$lockPath = Join-Path $ScriptRoot "version-lock.json"
$lock = Get-Content -Raw $lockPath | ConvertFrom-Json

function Ensure-GitRepo {
    param(
        [string]$RepoUrl,
        [string]$TargetPath,
        [string]$Ref
    )

    if (-not (Test-Path $TargetPath)) {
        git clone $RepoUrl $TargetPath
    }

    Push-Location $TargetPath
    try {
        git fetch --all --tags
        if ($Ref) {
            try {
                git checkout $Ref
            }
            catch {
                Write-Warning "无法切换到 $Ref，保留当前分支。"
            }
        }
    }
    finally {
        Pop-Location
    }
}

Ensure-GitRepo -RepoUrl $lock.llamafactory_repo -TargetPath (Join-Path $WorkspaceRoot "LLaMA-Factory") -Ref $lock.llamafactory_ref
Ensure-GitRepo -RepoUrl $lock.verl_repo -TargetPath (Join-Path $WorkspaceRoot "verl") -Ref $lock.verl_ref

Write-Host "bootstrap complete"
