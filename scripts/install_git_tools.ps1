$ErrorActionPreference = "Stop"

$temp = Join-Path $env:TEMP "course-git-tools"
New-Item -ItemType Directory -Force -Path $temp | Out-Null

Write-Host "Downloading latest Git for Windows installer..."
$gitRelease = Invoke-RestMethod `
    -Uri "https://api.github.com/repos/git-for-windows/git/releases/latest" `
    -Headers @{ "User-Agent" = "course-wafer-project" }
$gitAsset = $gitRelease.assets |
    Where-Object { $_.name -match "^Git-.*-64-bit\.exe$" } |
    Select-Object -First 1
if (-not $gitAsset) {
    throw "Could not find Git for Windows x64 installer asset."
}
$gitInstaller = Join-Path $temp $gitAsset.name
Invoke-WebRequest -Uri $gitAsset.browser_download_url -OutFile $gitInstaller

Write-Host "Installing Git for Windows..."
Start-Process -FilePath $gitInstaller `
    -ArgumentList "/VERYSILENT", "/NORESTART", "/NOCANCEL", "/SP-" `
    -Wait

Write-Host "Downloading latest GitHub CLI installer..."
$ghRelease = Invoke-RestMethod `
    -Uri "https://api.github.com/repos/cli/cli/releases/latest" `
    -Headers @{ "User-Agent" = "course-wafer-project" }
$ghAsset = $ghRelease.assets |
    Where-Object { $_.name -match "^gh_.*_windows_amd64\.msi$" } |
    Select-Object -First 1
if (-not $ghAsset) {
    throw "Could not find GitHub CLI Windows amd64 MSI asset."
}
$ghInstaller = Join-Path $temp $ghAsset.name
Invoke-WebRequest -Uri $ghAsset.browser_download_url -OutFile $ghInstaller

Write-Host "Installing GitHub CLI..."
Start-Process -FilePath "msiexec.exe" `
    -ArgumentList "/i", $ghInstaller, "/qn", "/norestart" `
    -Wait

$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") +
    ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

Write-Host ""
Write-Host "Installed versions:"
& "C:\Program Files\Git\cmd\git.exe" --version
& "C:\Program Files\GitHub CLI\gh.exe" --version

Write-Host ""
Write-Host "Restart PowerShell or Codex after installation so PATH is refreshed."
