<#
.SYNOPSIS
  First install of SpeechForge on a Windows GPU host (SPEC-speechforge 10.2).
  Run once, from an elevated PowerShell (the firewall rule and the crash-dump
  key need admin). Everything after this is `speechforge update`.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File install.ps1 `
      -Bundle C:\Users\me\Downloads\speechforge.bundle -BoxAddress 100.x.y.z

.PARAMETER Bundle      The git bundle to install from (SpeechForge has no remote).
.PARAMETER Tag         Release tag in the bundle (default: the newest).
.PARAMETER BoxAddress  The client box's tailnet address: the only source the
                       firewall rule admits.
.PARAMETER Root        Install root (default C:\SpeechForge).
.PARAMETER Device      CUDA ordinal (PCI-bus order) to serve on (default 3).
.PARAMETER ExpectName  Card name check (default "3090").
#>
param(
  [Parameter(Mandatory = $true)][string]$Bundle,
  [Parameter(Mandatory = $true)][string]$BoxAddress,
  [string]$Tag = "",
  [string]$Root = "C:\SpeechForge",
  [int]$Device = 3,
  [string]$ExpectName = "3090",
  [int]$Port = 1236
)
$ErrorActionPreference = "Stop"
function Step($n, $text) { Write-Host "`n[$n/9] $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "  FAIL: $text" -ForegroundColor Red; exit 1 }

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Fail "run this from an elevated PowerShell (firewall + crash-dump settings need admin)" }

# ------------------------------------------------------------------ 1. checks
Step 1 "Windows, driver, disk, tailscale"
$build = [int](Get-CimInstance Win32_OperatingSystem).BuildNumber
if ($build -lt 22000) { Fail "Windows 11 required (build $build): NVIDIA ships no new Windows 10 drivers" }
$drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
if ([int]($drv.Split(".")[0]) -lt 580) { Fail "NVIDIA driver $drv is older than R580 (CUDA 13)" }
Write-Host "  Windows build $build, driver $drv"
$free = (Get-PSDrive -Name ($Root.Substring(0, 1))).Free / 1GB
if ($free -lt 20) { Fail ("only {0:N0} GiB free on {1}: need 20 (models ~6 GB x releases)" -f $free, $Root.Substring(0, 2)) }
$ts = (& tailscale ip -4 2>$null | Select-Object -First 1)
if (-not $ts) { Fail "tailscale is not running (the server binds the tailnet address)" }
$ts = $ts.Trim()
Write-Host "  tailnet address $ts"

# ------------------------------------------------------------------ 2. uv
Step 2 "uv"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  if (Get-Command winget -ErrorAction SilentlyContinue) {
    winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
  } else { Fail "install uv first: https://docs.astral.sh/uv/" }
  $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" + $env:Path
}
uv --version

# ------------------------------------------------------------------ 3. layout
Step 3 "layout $Root\{releases,data,current}"
git bundle verify $Bundle | Out-Null
if (-not $Tag) { $Tag = (git ls-remote --tags $Bundle | ForEach-Object { ($_ -split "refs/tags/")[1] } |
                          Where-Object { $_ -notlike "*^{}" } | Sort-Object | Select-Object -Last 1) }
if (-not $Tag) { Fail "the bundle has no tags" }
$rel = Join-Path $Root "releases\$Tag"
$data = Join-Path $Root "data"
New-Item -ItemType Directory -Force -Path (Join-Path $Root "releases"), $data, (Join-Path $data "state") | Out-Null
if (Test-Path $rel) { Fail "$rel already exists (use speechforge update for a new release)" }
git clone --quiet --branch $Tag $Bundle $rel
$current = Join-Path $Root "current"
if (Test-Path $current) { cmd /c rmdir "$current" | Out-Null }
New-Item -ItemType Junction -Path $current -Target $rel | Out-Null
icacls $data /grant:r "${env:USERNAME}:(OI)(CI)F" | Out-Null

# ------------------------------------------------------------------ 4. environment
Step 4 "uv sync --locked --extra gpu"
Push-Location $rel
uv sync --locked --extra gpu --extra dev
Pop-Location
$py = Join-Path $rel ".venv\Scripts\python.exe"
$env:SPEECHFORGE_DATA_DIR = $data

# ------------------------------------------------------------------ 5. config + models
Step 5 "config.yaml and models (hash-verified)"
$cfg = Join-Path $data "config.yaml"
if (-not (Test-Path $cfg)) {
@"
server:
  hosts: ["127.0.0.1", "$ts"]
  port: $Port
gpu:
  devices: [$Device]
  expect_name: "$ExpectName"
"@ | Set-Content -Encoding UTF8 $cfg
}
& $py -m speechforge fetch-models
if ($LASTEXITCODE -ne 0) { Fail "model download or hash check failed" }

# ------------------------------------------------------------------ 6. voices dir
Step 6 "voice conditionals folder (biometric-grade: your user only, not indexed, not OneDrive)"
$voices = Join-Path $env:LOCALAPPDATA "SpeechForge\voices"
if ($voices -match "OneDrive") { Fail "LOCALAPPDATA is inside OneDrive; set voices.dir in config.yaml" }
New-Item -ItemType Directory -Force -Path $voices | Out-Null
icacls $voices /inheritance:r /grant:r "${env:USERNAME}:(OI)(CI)F" | Out-Null
attrib +I $voices   # not content-indexed (Windows Search)

# ------------------------------------------------------------------ 7. token
Step 7 "bearer token (shown ONCE: copy it to the box's speechforge.env now)"
& $py -m speechforge token new
Write-Host "  On the box: ~/.config/local-chat/speechforge.env  ->  SPEECHFORGE_TOKEN=<the line above>" -ForegroundColor Yellow
icacls (Join-Path $data "speechforge.env") /inheritance:r /grant:r "${env:USERNAME}:F" | Out-Null

# ------------------------------------------------------------------ 8. firewall + crash dumps
Step 8 "firewall rule (TCP $Port from $BoxAddress only) and mini crash dumps"
Get-NetFirewallRule -Name SpeechForge -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -Name SpeechForge -DisplayName SpeechForge -Direction Inbound -Action Allow `
  -Protocol TCP -LocalPort $Port -LocalAddress $ts -RemoteAddress $BoxAddress -Profile Any | Out-Null
$wer = "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\python.exe"
New-Item -Force -Path $wer | Out-Null
New-ItemProperty -Force -Path $wer -Name DumpType -PropertyType DWord -Value 1 | Out-Null   # mini, no heap
New-ItemProperty -Force -Path $wer -Name DumpCount -PropertyType DWord -Value 3 | Out-Null
New-ItemProperty -Force -Path $wer -Name DumpFolder -PropertyType ExpandString `
  -Value (Join-Path $data "state\dumps") | Out-Null

# ------------------------------------------------------------------ 9. start at logon + doctor
Step 9 "Startup-folder launcher and doctor"
$startup = [Environment]::GetFolderPath("Startup")
@"
@echo off
start "SpeechForge" /min "$current\launchers\Start SpeechForge.bat"
"@ | Set-Content -Encoding ASCII (Join-Path $startup "SpeechForge.bat")
$env:SPEECHFORGE_ROOT = $Root
& $py -m speechforge doctor
Write-Host "`nInstalled $Tag. Start it now: `"$current\launchers\Start SpeechForge.bat`" (or log off/on)." -ForegroundColor Green
