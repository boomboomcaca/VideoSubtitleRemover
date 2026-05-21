# Video Subtitle Remover Pro Launcher
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " VIDEO SUBTITLE REMOVER PRO" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "First-time setup required." -ForegroundColor Yellow
    Write-Host "Preparing the runtime and dependencies..." -ForegroundColor Yellow
    Write-Host ""
    python setup.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Setup did not complete. Review the messages above, then try again." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit $LASTEXITCODE
    }
}

if (Test-Path ".\venv\Scripts\pythonw.exe") {
    Start-Process -FilePath ".\venv\Scripts\pythonw.exe" -ArgumentList "VideoSubtitleRemover.py"
    exit 0
}

if (Test-Path ".\venv\Scripts\python.exe") {
    Start-Process -FilePath ".\venv\Scripts\python.exe" -ArgumentList "VideoSubtitleRemover.py"
    exit 0
}

Write-Host "The Python runtime could not be found in the virtual environment." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
exit 1
