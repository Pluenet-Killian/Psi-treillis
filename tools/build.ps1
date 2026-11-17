# SPDX-License-Identifier: Apache-2.0
# Configures and builds Psi-treillis on Windows with MSVC 2022 (host compiler of nvcc) and Ninja, in build\.
#
#   .\tools\build.ps1                    the library, psit and the examples
#   .\tools\build.ps1 -Tools             also the measurement instruments of tools\ (solver, front end, tracers, viewer)
#   .\tools\build.ps1 -Clean             removes build\ first
#   .\tools\build.ps1 -Install folder    then installs the header, the library, psit and the CMake configuration
#   .\tools\build.ps1 -Dir build_tests   builds in another folder
#
# On Linux, call CMake directly (see README.md).
param([switch]$Tools, [switch]$Clean, [string]$Install = '', [string]$Dir = 'build')
$ErrorActionPreference = 'Stop'

$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -products * -version '[17.0,18.0)' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath |
    Select-Object -First 1
if (-not $vs) { throw 'No Visual Studio 2022 installation with the x64 C++ tools.' }

# Imports the vcvars64 environment into the current PowerShell session.
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] }
}

$root = Split-Path $PSScriptRoot -Parent
$build = Join-Path $root $Dir
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

$withTools = if ($Tools) { 'ON' } else { 'OFF' }
cmake -S $root -B $build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl "-DPSIT_BUILD_TOOLS=$withTools"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
cmake --build $build
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
if ($Install) {
    cmake --install $build --prefix $Install
    if ($LASTEXITCODE -ne 0) { throw 'Installation failed.' }
}
