<#
.SYNOPSIS
    Build Timbrel for Windows with PyInstaller and zip it for a release.

.DESCRIPTION
    Produces dist\Timbrel\ (one-folder build: Qt's LGPL DLLs stay separate
    files that users can replace) and dist\Timbrel-v<version>-win64.zip.
    VB-Audio Virtual Cable is never bundled; users install it themselves.

.EXAMPLE
    powershell scripts/build.ps1
    powershell scripts/build.ps1 -Python .venv\Scripts\python.exe -Version 0.1.0
#>
param(
    [string]$Python = "",
    [string]$Version = ""
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Python) {
    $venv = Join-Path $root ".venv\Scripts\python.exe"
    $Python = if (Test-Path $venv) { $venv } else { "python" }
}
if (-not $Version) {
    $Version = (& $Python -c "import timbrel; print(timbrel.__version__)").Trim()
}
Write-Host "Building Timbrel $Version with $Python"

& $Python -m PyInstaller --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "PyInstaller missing: pip install -e .[build]" }

# PyInstaller resolves relative paths from --specpath, so use absolute ones.
$build = Join-Path $root "build"
$dist = Join-Path $root "dist"
$icon = Join-Path $build "timbrel.ico"
$presets = Join-Path $root "src\timbrel\presets\builtin"

Remove-Item -Recurse -Force (Join-Path $build "pyinstaller"), (Join-Path $dist "Timbrel") -ErrorAction SilentlyContinue
& $Python (Join-Path $root "scripts\make_icon.py") $icon
if ($LASTEXITCODE -ne 0) { throw "icon generation failed" }

& $Python -m PyInstaller `
    --noconfirm --clean --windowed `
    --name Timbrel `
    --icon $icon `
    --workpath (Join-Path $build "pyinstaller") `
    --specpath $build `
    --distpath $dist `
    --paths (Join-Path $root "src") `
    --add-data "$presets;timbrel\presets\builtin" `
    --exclude-module tkinter `
    (Join-Path $root "scripts\timbrel_entry.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$app = Join-Path $dist "Timbrel"
# Not needed by a widgets app: Qt's software OpenGL renderer (20 MB) and Qt's
# own UI translations (Timbrel isn't translated).
$qt = Join-Path $app "_internal\PySide6"
Remove-Item (Join-Path $qt "opengl32sw.dll"), (Join-Path $qt "translations") -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item (Join-Path $root "LICENSE"), (Join-Path $root "README.md"), (Join-Path $root "THIRD_PARTY_NOTICES.md") $app

# Compress-Archive in Windows PowerShell 5.1 fails on some files; use .NET.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = Join-Path $dist "Timbrel-v$Version-win64.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
[System.IO.Compression.ZipFile]::CreateFromDirectory($app, $zip, "Optimal", $true)
$size = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "Done: $zip ($size MB)"
