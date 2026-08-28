# Build a distributable Windows folder: dist\Udarata\
# Usage (from project root):
#   powershell -ExecutionPolicy Bypass -File build_exe.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "=== Udarata executable build ===" -ForegroundColor Cyan
Write-Host "Root: $Root"

# Ensure packaging tools
python -m pip install -q --upgrade pip
python -m pip install -q "pyinstaller>=6.0" -r requirements.txt

# Ensure MediaPipe full model exists for offline testers
$model = Join-Path $Root "models\pose_landmarker_full.task"
if (-not (Test-Path $model)) {
    Write-Host "Downloading MediaPipe full pose model..." -ForegroundColor Yellow
    python -c "from core.pose_extractor import _ensure_model; _ensure_model(1)"
}

# Ensure WAVs exist so testers hear audio without ffmpeg on first run
Write-Host "Ensuring reference WAV audio tracks..." -ForegroundColor Yellow
python -c @"
import os, config
for style in config.list_styles():
    for step in config.list_steps(style['id']):
        sid = step['id']
        vp = config.resolve_display_video_path(sid)
        if os.path.isfile(vp):
            w = config.ensure_expert_audio(vp, force=False)
            print(sid, '->', os.path.basename(w) if w else 'NO WAV')
"@

Write-Host "Running PyInstaller..." -ForegroundColor Yellow
python -m PyInstaller --noconfirm --clean Udarata.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit $LASTEXITCODE" }

$Dist = Join-Path $Root "dist\Udarata"
if (-not (Test-Path (Join-Path $Dist "Udarata.exe"))) {
    throw "Build finished but Udarata.exe was not found in $Dist"
}

# Copy runtime media next to the exe (writable + easy to update)
foreach ($folder in @("assets", "data", "models")) {
    $src = Join-Path $Root $folder
    $dst = Join-Path $Dist $folder
    if (Test-Path $src) {
        Write-Host "Copying $folder -> dist\Udarata\$folder"
        if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
        Copy-Item $src $dst -Recurse -Force
    }
}

# Tester readme
$readme = @"
Sri Lankan Traditional Dance Coaching — Test Build
=================================================

1. Unzip this whole folder (keep assets, data, models next to Udarata.exe).
2. Double-click Udarata.exe
3. Allow camera access when Windows asks.
4. Choose a dance style, then pick a step and begin.

Notes
- First launch may take a few seconds while models load.
- Keep the console window open if something fails — copy the error text.
- Audio needs the .wav files inside assets\<style>\ (included in this build).

Folder layout
  Udarata.exe
  _internal\
  assets\
    udarata\
    sabaragamuwa\
  data\
    udarata\
    sabaragamuwa\
  models\
"@
Set-Content -Path (Join-Path $Dist "README_TESTERS.txt") -Value $readme -Encoding UTF8

Write-Host ""
Write-Host "Build OK:" -ForegroundColor Green
Write-Host "  $Dist\Udarata.exe"
Write-Host "Zip the 'dist\Udarata' folder and send it to testers."
