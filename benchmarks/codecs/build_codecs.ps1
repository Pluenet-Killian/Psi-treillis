# SPDX-License-Identifier: Apache-2.0
# Builds the comparison compressors from their unmodified sources in third_party\ (git submodules pinned at the
# revisions that were measured) into build\codecs\. Windows, MSVC 2022, Ninja.
#
#   .\benchmarks\codecs\build_codecs.ps1 [-Only sz3,sperr,zfp,cuszp]
#
# What is built, and how each compressor is then called by benchmarks\codecs\comparison_codecs.py:
#   sz3     its command-line tool (sz3.exe)
#   sperr   its static library without OpenMP, then a DLL that exports its C interface (sperr_capi.dll). Its own
#           command-line tool does not compile with MSVC (error C2678) and its OpenMP loops use unsigned indices that
#           MSVC refuses (error C3016): SPERR therefore runs on one thread here.
#   zfp     its command-line tool (zfpcmd.exe), with OpenMP
#   cuszp   its own example program (examples/cuSZp.cpp), through the wrapper project cuszp_wrap\, because the original
#           CMakeLists does not configure with MSVC
param([string[]]$Only = @('sz3', 'sperr', 'zfp', 'cuszp'))
$ErrorActionPreference = 'Continue'
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -products * -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vs) { throw 'No Visual Studio 2022 installation with the x64 C++ tools.' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] } }

$jobs = @{
    sz3   = @{ Src = 'third_party\SZ3';   Args = @('-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTING=OFF', '-DSZ3_INSTALL=OFF') }
    sperr = @{ Src = 'third_party\SPERR'; Args = @('-DBUILD_SHARED_LIBS=OFF', '-DBUILD_UNIT_TESTS=OFF', '-DBUILD_CLI_UTILITIES=OFF', '-DUSE_OMP=OFF') }
    zfp   = @{ Src = 'third_party\zfp';   Args = @('-DBUILD_SHARED_LIBS=OFF', '-DBUILD_TESTING=OFF', '-DBUILD_UTILITIES=ON', '-DBUILD_EXAMPLES=OFF', '-DZFP_WITH_OPENMP=ON') }
    cuszp = @{ Src = 'benchmarks\codecs\cuszp_wrap'; Args = @("-DCUSZP_SRC=$($root.Path.Replace('\','/'))/third_party/cuSZp", '-DCMAKE_CUDA_ARCHITECTURES=native') }
}
$failed = 0
foreach ($name in $Only) {
    $j = $jobs[$name]
    $b = Join-Path $root "build\codecs\$name"
    New-Item -ItemType Directory -Force $b | Out-Null
    $log = Join-Path $b 'build_log.txt'
    $cfg = @('-S', (Join-Path $root $j.Src), '-B', $b, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=cl', '-DCMAKE_C_COMPILER=cl') + $j.Args
    & cmake @cfg *> $log
    $c = $LASTEXITCODE
    if ($c -eq 0) { & cmake --build $b *>> $log; $c = $LASTEXITCODE }
    if ($name -eq 'sperr' -and $c -eq 0) {
        & link /nologo /DLL /LTCG "/OUT:$b\sperr_capi.dll" "/DEF:$PSScriptRoot\sperr_capi.def" "/WHOLEARCHIVE:$b\src\SPERR.lib" *>> $log
        $c = $LASTEXITCODE
    }
    $built = (Get-ChildItem $b -Recurse -Include *.exe, *.dll -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch 'CMakeFiles' } | ForEach-Object { $_.Name }) -join ', '
    Write-Host ("{0,-6} exit code {1}; built: {2}" -f $name, $c, $built)
    if ($c -ne 0) { $failed++; Get-Content $log -Tail 12 | ForEach-Object { Write-Host "    $_" } }
}
exit $failed
