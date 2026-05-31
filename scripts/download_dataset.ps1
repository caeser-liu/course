param(
    [string]$Dataset = "qingyi/wm811k-wafer-map",
    [string]$OutputDir = "archive"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Error "Kaggle CLI is not installed. Install it with: pip install kaggle"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

Write-Host "Downloading Kaggle dataset '$Dataset' to '$OutputDir'..."
kaggle datasets download -d $Dataset -p $OutputDir --unzip

Write-Host ""
Write-Host "Download complete. Expected raw file: $OutputDir\LSWMD.pkl"
Write-Host "Next step: python data_loader.py"
