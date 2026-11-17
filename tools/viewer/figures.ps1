# SPDX-License-Identifier: Apache-2.0
# Retakes the captures of the visual comparator published in README.md and docs/ (folder docs\figures\).
#
# Usage: .\tools\viewer\figures.ps1 [-Only particles_256,divergence_256] [-Force]
#
# Each figure is a capture of psi_view (build\, built by tools\build.ps1 -Tools) in synchronised mode: the two halves
# show the same frame. The files compared are re-encoded right here by the current encoder, in build\work\figures\: a
# figure therefore cannot show a file of an earlier version of the codec. Settings: degree 3, default values of pack;
# cuZFP at 2 bits per value; 4 million particles per side, sorted every 16 frames (--sort), as in benchmarks/playback.
# Needs data\turb256.seq and data\turb512.seq.
#
# Writes the captures in docs\figures\ and, in benchmarks\playback\figures.log, what psi_view printed about each one.
#
# The band of each capture displays times measured during the run. They are valid only if the GPU is free: the script
# stops if another compute program occupies it (-Force overrides this, and the times must then not be quoted).
param(
    [string[]]$Only = @(),
    [switch]$Force
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root

# name of the figure, size, left side, view (1 particles, 2 divergence, 3 error against the raw frame)
$figures = @(
    @{ Name = 'particles_256';  Size = 256; Left = 'zfp'; View = 1 },
    @{ Name = 'divergence_256'; Size = 256; Left = 'zfp'; View = 2 },
    @{ Name = 'error_256';      Size = 256; Left = 'zfp'; View = 3 },
    @{ Name = 'particles_512';  Size = 512; Left = 'zfp'; View = 1 },
    @{ Name = 'raw_512';        Size = 512; Left = 'raw'; View = 1 }
)
if ($Only.Count) { $figures = $figures | Where-Object { $Only -contains $_.Name } }
if (-not $figures) { throw "No figure of that name. Names: particles_256, divergence_256, error_256, particles_512, raw_512." }

$occupants = & nvidia-smi --query-compute-apps=process_name --format=csv,noheader |
    Where-Object { $_ -match '(psi_|tracers|dns|python|sz3|cuSZp|zfp)[^\\/]*$' }
if ($occupants -and -not $Force) {
    throw "The GPU is occupied by: $($occupants -join ', '). The times of the band would be wrong; run again later, or with -Force."
}

$workDir = 'build/work/figures'
$figureDir = 'docs/figures'
New-Item -ItemType Directory -Force $workDir, $figureDir | Out-Null
# What psi_view measures during each capture is kept next to the other logs of the comparator.
$logFile = 'benchmarks/playback/figures.log'
New-Item -ItemType Directory -Force (Split-Path $logFile) | Out-Null
"# .\tools\viewer\figures.ps1, $(Get-Date -Format 'yyyy-MM-dd'): output of psi_view for each capture of docs/figures/" | Set-Content $logFile
$pack = '.\build\psi_stream.exe'
$view = '.\build\psi_view.exe'

foreach ($size in ($figures | ForEach-Object { $_.Size } | Sort-Object -Unique)) {
    $seq = "data/turb$size.seq"
    if (-not (Test-Path $seq)) { throw "Sequence missing: $seq (see benchmarks/README.md)." }
    # Same ranges of frames as the reference files: 33 frames from frame 16 at 256^3, 64 from frame 0 at 512^3.
    $start = if ($size -eq 256) { 16 } else { 0 }
    $frames = if ($size -eq 256) { 33 } else { 64 }
    & $pack pack --data $seq --file "$workDir/turb${size}_d3.psi" --degree 3 --start $start --frames $frames | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pack failed at $size^3 (code $LASTEXITCODE)." }
    if (($figures | Where-Object { $_.Size -eq $size -and $_.Left -eq 'zfp' })) {
        & $view packzfp --data $seq --file "$workDir/turb$size.zfs" --rate 2 --start $start --frames $frames | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "packzfp failed at $size^3 (code $LASTEXITCODE)." }
    }
    foreach ($f in ($figures | Where-Object { $_.Size -eq $size })) {
        $left = if ($f.Left -eq 'zfp') { "zfp:$workDir/turb$size.zfs" } else { "raw:$seq" }
        $image = "$figureDir/$($f.Name).png"
        & $view --left $left --right "psi:$workDir/turb${size}_d3.psi" --data $seq --view $f.View --sync --sort --particles 4000000 `
            --start $start --frames $frames --width 1600 --height 900 --snapshot $image |
            Select-String -Pattern '^(Capture|Left|Right|  (cost|divergence|error))' | ForEach-Object { $_.Line } |
            Tee-Object -FilePath $logFile -Append
        if ($LASTEXITCODE -ne 0) { throw "psi_view failed for $($f.Name) (code $LASTEXITCODE)." }
    }
}
# psi_view writes uncompressed PNGs (4.2 MB): recompress them losslessly.
& python tools/viewer/png_compress.py "docs/figures/*_256.png" "docs/figures/*_512.png"
if ($LASTEXITCODE -ne 0) { Write-Warning 'Captures left uncompressed.' }
