# Lexora Worker - Windows Installer
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"
$RepoUrl = "https://github.com/Alex0715/lexora-worker"
$InstallDir = "$env:USERPROFILE\.lexora-worker"
$VenvDir = "$InstallDir\venv"

function Write-Step   { param($msg) Write-Host "> $msg" -ForegroundColor Cyan }
function Write-OK     { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn   { param($msg) Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Fail   { param($msg) Write-Host "[X] $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "================================================" -ForegroundColor White
Write-Host "       Lexora Worker - Installer                " -ForegroundColor White
Write-Host "  Distributed AI Compute Node Setup            " -ForegroundColor White
Write-Host "================================================" -ForegroundColor White
Write-Host ""

# -- check Python ----------------------------------------------------------
Write-Step "Checking Python installation..."
$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd -c "import sys; print(sys.version_info >= (3,9))" 2>$null
        if ($ver -eq "True") { $PythonCmd = $cmd; break }
    } catch {}
}

if (-not $PythonCmd) {
    Write-Warn "Python 3.9+ not found."
    Write-Warn "Download from: https://python.org/downloads"
    Write-Fail "Please install Python 3.9+ and re-run this script."
}

$PythonVer = & $PythonCmd --version
Write-OK "Python found: $PythonVer"

# -- detect GPU --------------------------------------------------------------
Write-Step "Detecting GPU..."
# vLLM does not ship Windows wheels, so always use the "windows" extra
# Base Windows extras (torch + transformers + accelerate).
# NVIDIA GPUs also get the image extra (diffusers + Pillow) for FLUX support.
$Extras = "windows"
try {
    $NvidiaSmi = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($NvidiaSmi) {
        $GpuName = ($NvidiaSmi | Select-Object -First 1).Split(",")[0].Trim()
        Write-OK "NVIDIA GPU detected: $GpuName"
        $Extras = "windows,image"
        Write-OK "Image generation (FLUX) support will be installed"
    } else {
        Write-Warn "No NVIDIA GPU detected - using CPU backend (slower)"
    }
} catch {
    Write-Warn "No NVIDIA GPU detected - using CPU backend (slower)"
}

# -- check Visual C++ Redistributable -----------------------------------------
# PyTorch on Windows requires the MSVC runtime DLLs (vcruntime140.dll,
# vcruntime140_1.dll). Without them, "import torch" fails with WinError 126.
Write-Step "Checking Visual C++ Redistributable..."
$VcDll = Join-Path $env:SystemRoot "System32\vcruntime140_1.dll"
if (Test-Path $VcDll) {
    Write-OK "Visual C++ Redistributable found"
} else {
    Write-Step "Downloading and installing Visual C++ Redistributable..."
    $VcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    $VcRedistExe = "$env:TEMP\vc_redist.x64.exe"
    Invoke-WebRequest $VcRedistUrl -OutFile $VcRedistExe
    Start-Process -FilePath $VcRedistExe -ArgumentList "/install", "/quiet", "/norestart" -Wait
    Write-OK "Visual C++ Redistributable installed"
}

# -- create install dir & venv -----------------------------------------------
Write-Step "Setting up install directory at $InstallDir..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

if (-not (Test-Path $VenvDir)) {
    Write-Step "Creating virtual environment..."
    & $PythonCmd -m venv $VenvDir
    Write-OK "Virtual environment created"
}

$Pip = "$VenvDir\Scripts\pip.exe"
$WorkerBin = "$VenvDir\Scripts\lexora-worker.exe"

# -- install worker package ---------------------------------------------------
Write-Step "Installing lexora-worker..."
& $Pip install --quiet --upgrade pip

$ScriptDir = if ($MyInvocation.MyCommand.Path) { Split-Path -Parent $MyInvocation.MyCommand.Path } else { $null }
$WorkerPyproject = if ($ScriptDir) { Join-Path $ScriptDir "pyproject.toml" } else { $null }

if (Test-Path $WorkerPyproject) {
    $WorkerDir = Join-Path $ScriptDir "worker"
    if ($Extras) {
        & $Pip install --quiet -e "$WorkerDir[$Extras]"
    } else {
        & $Pip install --quiet -e $WorkerDir
    }
} else {
    Write-Step "Downloading from GitHub..."
    $TmpZip = "$env:TEMP\lexora-worker.zip"
    Invoke-WebRequest "$RepoUrl/archive/refs/heads/main.zip" -OutFile $TmpZip
    $TmpDir = "$env:TEMP\lexora-worker-src"
    Expand-Archive $TmpZip -DestinationPath $TmpDir -Force
    $ExtractedDir = Get-ChildItem $TmpDir | Select-Object -First 1
    # In the public worker repo pyproject.toml is at the root, not in worker\
    if ($Extras) {
        & $Pip install --quiet -e "$($ExtractedDir.FullName)[$Extras]"
    } else {
        & $Pip install --quiet -e "$($ExtractedDir.FullName)\worker"
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Fail "Failed to install lexora-worker (pip exited with code $LASTEXITCODE). See output above for details."
}

Write-OK "lexora-worker installed"

# -- add to PATH ---------------------------------------------------------------
$BinPath = "$VenvDir\Scripts"
$CurrentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($CurrentPath -notlike "*$BinPath*") {
    [Environment]::SetEnvironmentVariable("PATH", "$BinPath;$CurrentPath", "User")
    $env:PATH = "$BinPath;$env:PATH"
    Write-OK "Added to user PATH"
}

# -- run setup wizard -----------------------------------------------------------
if (-not (Test-Path $WorkerBin)) {
    Write-Fail "Installation did not produce $WorkerBin. Try removing $InstallDir and re-running this script."
}

Write-Host ""
Write-Host "Installation complete! Starting setup wizard..." -ForegroundColor White
Write-Host ""

& $WorkerBin setup
