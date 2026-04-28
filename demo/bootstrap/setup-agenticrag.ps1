[CmdletBinding()]
param(
    [string]$WorkspaceRoot = "",
    [string]$PythonExe = "python",
    [string]$VenvName = ".venv-agenticrag"
)

$ErrorActionPreference = "Stop"
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $WorkspaceRoot) {
    $WorkspaceRoot = (Resolve-Path (Join-Path $ScriptRoot "..")).Path
}

$venvPath = Join-Path $WorkspaceRoot $VenvName

if (-not (Test-Path $venvPath)) {
    & $PythonExe -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $WorkspaceRoot "requirements.txt")

Write-Host "agenticrag environment ready at $venvPath"
