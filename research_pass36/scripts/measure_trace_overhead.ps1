$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent | Split-Path -Parent
$out = Join-Path $root 'research_pass36\outputs\trace_overhead'
$exe = Join-Path $root 'research_pass35\outputs\binaries\reference\avifenc.exe'
$src = Join-Path $root 'hpt_data\googlewiki.png'
New-Item -ItemType Directory -Force $out | Out-Null
$encodeArgs = @('--codec aom --qcolor 5 --qalpha 0 --speed 6 --jobs 1 --depth 8 --yuv 420 --range full --cicp 1/13/1 --ignore-exif --ignore-xmp --ignore-profile', "`"$src`"", "`"$(Join-Path $out 'timed.avif')`"") -join ' '
$samples = [System.Collections.Generic.List[object]]::new()

for ($i = 0; $i -lt 28; $i++) {
  $watch = [System.Diagnostics.Stopwatch]::StartNew()
  $process = Start-Process -FilePath $exe -ArgumentList $encodeArgs -WorkingDirectory $out -PassThru -Wait `
    -RedirectStandardOutput (Join-Path $out 'stdout.txt') -RedirectStandardError (Join-Path $out 'stderr.txt')
  $watch.Stop()
  if ($process.ExitCode) { throw "avifenc failed: $($process.ExitCode)" }
  if ($i -ge 3) {
    $samples.Add([ordered]@{
      repetition = $i - 2
      wall_ms = [Math]::Round($watch.Elapsed.TotalMilliseconds, 3)
      process_cpu_ms = [Math]::Round($process.TotalProcessorTime.TotalMilliseconds, 3)
    })
  }
}

$sortedWall = @($samples | ForEach-Object wall_ms | Sort-Object)
$sortedCpu = @($samples | ForEach-Object process_cpu_ms | Sort-Object)
$result = [ordered]@{
  binary = $exe
  image = 'googlewiki.png'
  qcolor = 5
  speed = 6
  jobs = 1
  warmups = 3
  repetitions = $samples.Count
  samples = @($samples)
  median_wall_ms = $sortedWall[[int][Math]::Floor($sortedWall.Count / 2)]
  median_process_cpu_ms = $sortedCpu[[int][Math]::Floor($sortedCpu.Count / 2)]
}
$path = Join-Path $out 'unprofiled.json'
$json = $result | ConvertTo-Json -Depth 4
[IO.File]::WriteAllText($path, $json + "`n")
$json
