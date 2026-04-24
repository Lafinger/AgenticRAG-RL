[CmdletBinding()]
param(
    [string]$WorkspaceRoot = "",
    [string]$PythonExe = "python",
    [string]$VenvName = ".venv-verl",
    [switch]$AllowWindows
)

$ErrorActionPreference = "Stop"
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $WorkspaceRoot) {
    $WorkspaceRoot = (Resolve-Path (Join-Path $ScriptRoot "..")).Path
}

if ($env:OS -eq "Windows_NT" -and -not $AllowWindows) {
    Write-Warning "原生 Windows 11 不建议直接安装 verl/vLLM/flash-attn。请优先在 WSL2 或远端 Linux GPU 环境执行。"
    Write-Warning "如果你只想先装轻量依赖验证脚本入口，可追加 -AllowWindows。"
    return
}

$venvPath = Join-Path $WorkspaceRoot $VenvName
if (-not (Test-Path $venvPath)) {
    & $PythonExe -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $WorkspaceRoot "requirements-verl.txt")

$verlPath = Join-Path $WorkspaceRoot "verl"
if (Test-Path $verlPath) {
    & $venvPython -m pip install -e $verlPath
}

Write-Host "verl environment ready at $venvPath"
