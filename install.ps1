<#
.SYNOPSIS
  First install of SpeechForge on a Windows GPU host (SPEC-speechforge 10.2).
  Run once, from an elevated PowerShell (the firewall rule and the crash-dump
  key need admin). Everything after this is `speechforge update`. Running it
  again is safe: an existing release, config and token are kept and said so.

.EXAMPLE
  # From the public repository (newest v* tag), admitting one client box:
  powershell -ExecutionPolicy Bypass -File install.ps1 -BoxAddress 100.x.y.z

.EXAMPLE
  # Offline, from a git bundle, a chosen tag, data on another drive:
  powershell -ExecutionPolicy Bypass -File install.ps1 `
      -Source C:\Users\me\Downloads\speechforge.bundle -Tag v0.2.0 `
      -BoxAddress 100.x.y.z -DataDir D:\SpeechForge\data

.PARAMETER Source      Where the code comes from: a git URL (default: the public
                       GitHub repository) or a local git bundle / clone path
                       (offline). `-Bundle` is the old name and still works.
.PARAMETER Tag         Release tag to install (default: the newest v* tag at the
                       source, by version order).
.PARAMETER BoxAddress  The client box's tailnet address: the only remote the
                       firewall rule admits. When omitted the rule admits the
                       whole tailnet range (100.64.0.0/10) -- see step 8 for
                       why that is weaker; the bearer token still applies.
.PARAMETER Root        Install root (default C:\SpeechForge): releases\<tag>,
                       current (junction), and the data folder unless -DataDir.
.PARAMETER DataDir     Config, token file, models, logs, state (default
                       <Root>\data). Another location is remembered in
                       <Root>\local-env.bat, which every release's launcher reads.
.PARAMETER Device      CUDA ordinal (PCI-bus order) to serve on (default 3).
.PARAMETER ExpectName  Card name check (default "3090").
.PARAMETER Port        TCP port (default 1236).
.PARAMETER SkipFirewall  Do not touch the Windows firewall (you add the rule).
.PARAMETER SkipStartup   Do not install the Startup-folder launcher.
#>
param(
  [Alias("Bundle")][string]$Source = "https://github.com/LaserLloyd/SpeechForge.git",
  [string]$Tag = "",
  [string]$BoxAddress = "",
  [string]$Root = "C:\SpeechForge",
  [string]$DataDir = "",
  [int]$Device = 3,
  [string]$ExpectName = "3090",
  [int]$Port = 1236,
  [switch]$SkipFirewall,
  [switch]$SkipStartup
)
$ErrorActionPreference = "Stop"
$TailnetRange = "100.64.0.0/10"
function Step($n, $text) { Write-Host "`n[$n/9] $text" -ForegroundColor Cyan }
function Note($text) { Write-Host "  $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "  FAIL: $text" -ForegroundColor Red; exit 1 }

# A release tag is "v<major>.<minor>[.<patch>][-pre]". Newest by version, not by
# text (v0.10.0 is newer than v0.9.0); a pre-release sorts before its final.
function Select-NewestTag([string[]]$Tags) {
  $rows = foreach ($t in $Tags) {
    if ($t -match '^v(\d+(?:\.\d+)*)(.*)$') {
      $nums = $Matches[1]
      if ($nums -notmatch '\.') { $nums = "$nums.0" }
      [pscustomobject]@{ Tag = $t; Ver = [version]$nums; Final = [int]($Matches[2] -eq "") }
    }
  }
  $pick = $rows | Sort-Object -Property Ver, Final | Select-Object -Last 1
  if ($pick) { $pick.Tag } else { "" }
}

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
  [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Fail "run this from an elevated PowerShell (firewall + crash-dump settings need admin)" }
if (-not $DataDir) { $DataDir = Join-Path $Root "data" }
$BoxAddress = $BoxAddress.Trim()
if ($BoxAddress -and $BoxAddress -notmatch '^\d{1,3}(\.\d{1,3}){3}(/\d{1,2})?$') {
  Fail "-BoxAddress must be an IPv4 address or CIDR range (got '$BoxAddress')"
}

# ------------------------------------------------------------------ 1. checks
Step 1 "Windows, driver, disk, git, tailscale"
$build = [int](Get-CimInstance Win32_OperatingSystem).BuildNumber
if ($build -lt 22000) { Fail "Windows 11 required (build $build): NVIDIA ships no new Windows 10 drivers" }
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "nvidia-smi not found: install the NVIDIA driver (R580+)" }
$drv = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
if ([int]($drv.Split(".")[0]) -lt 580) { Fail "NVIDIA driver $drv is older than R580 (CUDA 13)" }
Write-Host "  Windows build $build, driver $drv"
$free = (Get-PSDrive -Name ($Root.Substring(0, 1))).Free / 1GB
if ($free -lt 20) { Fail ("only {0:N0} GiB free on {1}: need 20 (models ~6 GB x releases)" -f $free, $Root.Substring(0, 2)) }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Fail "git is not installed (https://git-scm.com/download/win)" }
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

# ------------------------------------------------------------------ 3. source + layout
Step 3 "source, tag, layout $Root\{releases,current} + data $DataDir"
$isUrl = $Source -match '^[a-z][a-z0-9+.-]*://' -or $Source -match '^[\w.-]+@[\w.-]+:'
if (-not $isUrl) {
  if (-not (Test-Path $Source)) { Fail "-Source '$Source' is neither a git URL nor an existing bundle/clone path" }
  $Source = (Resolve-Path $Source).Path
  if (Test-Path $Source -PathType Leaf) {
    git bundle verify $Source | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "'$Source' is not a valid git bundle" }
  }
  Write-Host "  source: local $Source"
} else {
  Write-Host "  source: $Source"
}
# One ls-remote, nothing fetched yet. Works the same for a URL, a bundle or a clone.
# (stderr is left on the console on purpose: under ErrorActionPreference=Stop a
# redirected native stderr line would abort the script before our message.)
$refs = @(& git ls-remote --tags $Source)
if ($LASTEXITCODE -ne 0) { Fail "git ls-remote failed for $Source (offline? use -Source <path to a bundle>)" }
$tags = @($refs | ForEach-Object { ($_ -split "refs/tags/")[1] } |
          Where-Object { $_ -and $_ -notlike "*^{}" -and $_ -like "v*" } | Sort-Object -Unique)
if (-not $tags) { Fail "no release tags (v*) at $Source" }
if (-not $Tag) {
  $Tag = Select-NewestTag $tags
  if (-not $Tag) { Fail "no tag at $Source looks like v<major>.<minor>.<patch>" }
} elseif ($tags -notcontains $Tag) {
  Fail "tag $Tag is not at $Source (have: $($tags -join ', '))"
}
Write-Host "  release tag $Tag (available: $($tags -join ', '))"

$rel = Join-Path $Root "releases\$Tag"
$data = $DataDir
New-Item -ItemType Directory -Force -Path (Join-Path $Root "releases"), $data, (Join-Path $data "state") | Out-Null
if (Test-Path $rel) {
  Note "$rel already exists: keeping it as it is (a re-run). Delete the folder to re-clone, or use"
  Note "'speechforge update' for a different release."
} else {
  git clone --quiet --branch $Tag --depth 1 $Source $rel
  if ($LASTEXITCODE -ne 0) { Fail "git clone of $Tag from $Source failed" }
}
$current = Join-Path $Root "current"
if (Test-Path $current) {
  # .Target is a string in PowerShell 7 and a string[] in 5.1.
  $was = [string](@((Get-Item $current).Target) | Select-Object -First 1)
  if ($was -and -not $was.TrimEnd('\').EndsWith("\releases\$Tag", [StringComparison]::OrdinalIgnoreCase)) {
    Note "current pointed at $was; re-pointing it at $rel"
  }
  cmd /c rmdir "$current" | Out-Null
}
New-Item -ItemType Junction -Path $current -Target $rel | Out-Null
icacls $data /grant:r "${env:USERNAME}:(OI)(CI)F" | Out-Null
# A data folder outside <Root>\data is not where the launchers look by default;
# <Root>\local-env.bat is read by every release's launchers (and survives updates).
$rootEnv = Join-Path $Root "local-env.bat"
if ($data -ne (Join-Path $Root "data")) {
@"
@echo off
REM Written by install.ps1: the data folder is not <root>\data.
set "SPEECHFORGE_ROOT=$Root"
set "SPEECHFORGE_DATA_DIR=$data"
"@ | Set-Content -Encoding ASCII $rootEnv
  Write-Host "  wrote $rootEnv (SPEECHFORGE_DATA_DIR=$data)"
} elseif (Test-Path $rootEnv) {
  # A re-run without -DataDir after one with it: this script now uses <Root>\data
  # while every launcher still reads the old location from this file. Say so
  # rather than delete a file the operator may have edited by hand.
  Note "$rootEnv exists and may point the launchers at another data folder than $data."
  Note "Delete it, or re-run with the same -DataDir, so the launchers and this install agree."
}

# ------------------------------------------------------------------ 4. environment
Step 4 "uv sync --locked --extra gpu"
Push-Location $rel
uv sync --locked --extra gpu --extra dev
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "uv sync failed" }
Pop-Location
$py = Join-Path $rel ".venv\Scripts\python.exe"
$env:SPEECHFORGE_DATA_DIR = $data
$env:SPEECHFORGE_ROOT = $Root

# ------------------------------------------------------------------ 5. config + models
Step 5 "config.yaml and models (hash-verified)"
$cfg = Join-Path $data "config.yaml"
if (Test-Path $cfg) {
  Note "config.yaml exists: left untouched (-Device/-ExpectName/-Port not applied to it)"
} else {
@"
server:
  hosts: ["127.0.0.1", "$ts"]
  port: $Port
gpu:
  devices: [$Device]
  expect_name: "$ExpectName"
"@ | Set-Content -Encoding UTF8 $cfg
  Write-Host "  wrote $cfg"
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
Step 7 "bearer token"
$envFile = Join-Path $data "speechforge.env"
$hasToken = (Test-Path $envFile) -and ((Select-String -Path $envFile -Pattern '^\s*(export\s+)?SPEECHFORGE_TOKEN\s*=' -Quiet))
if ($hasToken) {
  Note "a token already exists in ${envFile}: left untouched (fingerprint below)."
  Note "Rotate it with: `"$py`" -m speechforge token rotate"
  & $py -m speechforge token fingerprint
} else {
  Write-Host "  shown ONCE: copy it to the box's speechforge.env now" -ForegroundColor Yellow
  & $py -m speechforge token new
  Write-Host "  On the box: ~/.config/local-chat/speechforge.env  ->  SPEECHFORGE_TOKEN=<the line above>" -ForegroundColor Yellow
}
icacls $envFile /inheritance:r /grant:r "${env:USERNAME}:F" | Out-Null

# ------------------------------------------------------------------ 8. firewall + crash dumps
if ($SkipFirewall) {
  Step 8 "firewall: skipped (-SkipFirewall); mini crash dumps"
  Note "add an inbound rule yourself: TCP $Port, local address $ts, remote = the box. doctor reports it missing until then."
} else {
  $remote = if ($BoxAddress) { $BoxAddress } else { $TailnetRange }
  Step 8 "firewall rule (TCP $Port on $ts from $remote) and mini crash dumps"
  if (-not $BoxAddress) {
    # The tailnet range is every device on the tailnet (and anything shared
    # into it), not just the one box; the token is then the only thing between a
    # stray tailnet peer and the port. Pass -BoxAddress to narrow it later.
    Note "no -BoxAddress: admitting the whole tailnet range $TailnetRange. That is weaker than one"
    Note "address (every tailnet device reaches the port; only the token stops them). Re-run with"
    Note "-BoxAddress <box tailnet IP> to narrow the rule."
  }
  Get-NetFirewallRule -Name SpeechForge -ErrorAction SilentlyContinue | Remove-NetFirewallRule
  New-NetFirewallRule -Name SpeechForge -DisplayName SpeechForge -Direction Inbound -Action Allow `
    -Protocol TCP -LocalPort $Port -LocalAddress $ts -RemoteAddress $remote -Profile Any | Out-Null
}
$wer = "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\python.exe"
New-Item -Force -Path $wer | Out-Null
New-ItemProperty -Force -Path $wer -Name DumpType -PropertyType DWord -Value 1 | Out-Null   # mini, no heap
New-ItemProperty -Force -Path $wer -Name DumpCount -PropertyType DWord -Value 3 | Out-Null
New-ItemProperty -Force -Path $wer -Name DumpFolder -PropertyType ExpandString `
  -Value (Join-Path $data "state\dumps") | Out-Null

# ------------------------------------------------------------------ 9. start at logon + doctor
if ($SkipStartup) {
  Step 9 "Startup-folder launcher: skipped (-SkipStartup); doctor"
} else {
  Step 9 "Startup-folder launcher and doctor"
  $startup = [Environment]::GetFolderPath("Startup")
@"
@echo off
start "SpeechForge" /min "$current\launchers\Start SpeechForge.bat"
"@ | Set-Content -Encoding ASCII (Join-Path $startup "SpeechForge.bat")
}
& $py -m speechforge doctor
Write-Host "`nInstalled $Tag from $Source. Start it now: `"$current\launchers\Start SpeechForge.bat`" (or log off/on)." -ForegroundColor Green
Write-Host "Updates: `"$current\launchers\Update SpeechForge.bat`" (same source) or `"$py`" -m speechforge update --check" -ForegroundColor Green
