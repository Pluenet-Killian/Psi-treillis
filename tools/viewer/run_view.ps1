# SPDX-License-Identifier: Apache-2.0
# Launches the visual comparator psi_view (split screen, two complete chains on the same frames).
#
# Usage: .\tools\viewer\run_view.ps1 [-Size 256|512] [-Against zfp|raw|psi] [-Degree 2|3] [-View 1|2|3] [-Sort] [-Sync]
#                                    [-Tolerance 5] [-Snapshot image.png]
#
#   -Against zfp   on the left: cuZFP stream at 2 bits + half-precision texture (the compressed reference)
#   -Against raw   on the left: raw float32 frames read from disk (what a standard tool does)
#   -Against psi   on the left: the same Psi-treillis without an error bound (to be used with -Tolerance)
#   -Tolerance 5   on the right: Psi-treillis with an error bound, here 5 % of u' on the largest error (degree 3
#                  recommended)
#   -Degree 3      on the right: Psi-treillis of degree 3 (same field error as the reference, slower kernel)
#   -View          1 particles, 2 divergence slice, 3 error slice against the raw frame
#
# Needs build\ (tools\build.ps1 -Tools) and data\turb256.seq or data\turb512.seq. The files compared are kept in
# build\work\view\; the missing ones are produced at the first launch (psi_stream pack, psi_view packzfp). The script
# works from the root of the repository, wherever it is launched from: a relative -Snapshot path is taken from there.
param(
    [ValidateSet(256, 512)][int]$Size = 256,
    [ValidateSet('zfp', 'raw', 'psi')][string]$Against = 'zfp',
    [ValidateSet(2, 3)][int]$Degree = 2,
    [ValidateSet(1, 2, 3)][int]$View = 1,
    [switch]$Sort,
    [switch]$Sync,
    [double]$Tolerance = 0,
    [string]$Snapshot = '',
    [int]$Particles = 4000000
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
$workDir = 'build/work/view'
New-Item -ItemType Directory -Force $workDir | Out-Null
$seq = "data/turb$Size.seq"
if (-not (Test-Path $seq)) { throw "Sequence missing: $seq (see benchmarks/README.md)." }
# Same ranges of frames as the files already measured: 33 frames from frame 16 at 256^3, 64 from frame 0 at 512^3.
$start = if ($Size -eq 256) { 16 } else { 0 }
$frames = if ($Size -eq 256) { 33 } else { 64 }
$psi = if ($Degree -eq 3) { "$workDir/turb${Size}_d3.psi" } else { "$workDir/turb$Size.psi" }
$zfs = "$workDir/turb$Size.zfs"

if (-not (Test-Path $psi)) {
    & .\build\psi_stream.exe pack --data $seq --file $psi --degree $Degree --start $start --frames $frames
}
if ($Against -eq 'zfp' -and -not (Test-Path $zfs)) {
    & .\build\psi_view.exe packzfp --data $seq --file $zfs --rate 2 --start $start --frames $frames
}
# Right side: with -Tolerance, a file with an error bound (sparse fine level), produced at the first launch.
$right = $psi
if ($Tolerance -gt 0) {
    $inv = [Globalization.CultureInfo]::InvariantCulture
    $right = "$workDir/turb${Size}_d${Degree}_b$($Tolerance.ToString($inv).Replace('.', '_')).psi"
    if (-not (Test-Path $right)) {
        & .\build\psi_stream.exe pack --data $seq --file $right --degree $Degree --start $start --frames $frames --tolerance ($Tolerance / 100).ToString($inv)
    }
}
$left = if ($Against -eq 'zfp') { "zfp:$zfs" } elseif ($Against -eq 'raw') { "raw:$seq" } else {
    # Same encoder and same settings as the right side, without the bound: a separate file, because the older
    # turb*.psi files may have been written with another quantisation step.
    $ref = "$workDir/turb${Size}_d${Degree}_no_bound.psi"
    if (-not (Test-Path $ref)) {
        # Out-Host: the output of pack is displayed, but does not enter the value of this block.
        & .\build\psi_stream.exe pack --data $seq --file $ref --degree $Degree --start $start --frames $frames | Out-Host
    }
    "psi:$ref"
}
$argv = @('--left', $left, '--right', "psi:$right", '--data', $seq, '--view', "$View", '--particles', "$Particles", '--start', "$start", '--frames', "$frames")
if ($Sort) { $argv += '--sort' }
if ($Sync) { $argv += '--sync' }
if ($Snapshot) { $argv += @('--snapshot', $Snapshot) }
& .\build\psi_view.exe @argv
