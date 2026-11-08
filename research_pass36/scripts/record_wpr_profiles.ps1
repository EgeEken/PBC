$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$out = Join-Path $root 'research_pass36\outputs\wpr'
$bin = Join-Path $root 'research_pass36\outputs\profiling\avifenc-symbol.exe'
$path = "$root\research_pass36\outputs\profiling;C:\Strawberry\c\bin;C:\Windows\System32;C:\Windows"
New-Item -ItemType Directory -Force $out | Out-Null
$runLog = Join-Path $out 'record.log'
"started $(Get-Date -Format o) pid=$PID" | Set-Content -LiteralPath $runLog
$env:PATH = $path
$proc = Get-Process -Id $PID
$affinity = [IntPtr]1
try { $proc.ProcessorAffinity = $affinity } catch { }

$cases = @(
  @{ image = 'googlewiki.png'; q = 5; speed = 6 },
  @{ image = 'mario64.png'; q = 50; speed = 6 },
  @{ image = 'cat.jpg'; q = 35; speed = 6 },
  @{ image = 'googlewiki.png'; q = 5; speed = 8 },
  @{ image = 'cat.jpg'; q = 35; speed = 8 }
)
$rows = [System.Collections.Generic.List[object]]::new()

foreach ($captureId in 'a','b') {
  $etl = Join-Path $out "profile_$captureId.etl"
  $captureStart = Get-Date
  "capture $captureId start $(Get-Date -Format o)" | Add-Content -LiteralPath $runLog
  wpr -start CPU.light -filemode *> (Join-Path $out "wpr_start_$captureId.log")
  if ($LASTEXITCODE) { "WPR start failed: $LASTEXITCODE" | Add-Content -LiteralPath $runLog; throw "WPR start failed: $LASTEXITCODE" }
  try {
    foreach ($case in $cases) {
      $src = Join-Path $root "hpt_data\$($case.image)"
      for ($repeat = 1; $repeat -le 6; $repeat++) {
        $tag = "${captureId}_$([IO.Path]::GetFileNameWithoutExtension($case.image))_q$($case.q)_s$($case.speed)_r$repeat"
        $dst = Join-Path $out "$tag.avif"
        $argumentLine = "--codec aom --qcolor $($case.q) --qalpha 0 --speed $($case.speed) --jobs 1 --depth 8 --yuv 420 --range full --cicp 1/13/1 --ignore-exif --ignore-xmp --ignore-profile `"$src`" `"$dst`""
        $process = Start-Process -FilePath $bin -ArgumentList $argumentLine -WorkingDirectory $out -PassThru -Wait `
          -RedirectStandardOutput (Join-Path $out "$tag.stdout.log") -RedirectStandardError (Join-Path $out "$tag.stderr.log")
        if ($process.ExitCode) { throw "avifenc failed: $tag, exit $($process.ExitCode)" }
        $rows.Add([ordered]@{
          profile = $captureId
          image = $case.image
          qcolor = $case.q
          speed = $case.speed
          repeat = $repeat
          exit_code = $process.ExitCode
        })
      }
    }
  } catch {
    wpr -cancel | Out-Null
    $_ | Out-String | Add-Content -LiteralPath $runLog
    throw
  }
  wpr -stop $etl "Pass36 optimized symbol build $captureId" -skipPdbGen *> (Join-Path $out "wpr_stop_$captureId.log")
  if ($LASTEXITCODE) { "WPR stop failed: $LASTEXITCODE" | Add-Content -LiteralPath $runLog; throw "WPR stop failed: $LASTEXITCODE" }
  $rows.Add([ordered]@{ profile = $captureId; capture_started = $captureStart.ToUniversalTime().ToString('o'); trace_bytes = (Get-Item $etl).Length })
  "capture $captureId done bytes=$((Get-Item $etl).Length) $(Get-Date -Format o)" | Add-Content -LiteralPath $runLog
}

$manifest = [ordered]@{
  profiler = 'Windows Performance Recorder CPU.light'
  sampling_interval_ms = 1
  encoder = $bin
  encoder_threads = 1
  repetitions_per_case_per_profile = 6
  affinity_mask = '0x1 on capture PowerShell process; child inheritance checked from sample ProcessorNumber distribution'
  runs = @($rows)
}
[IO.File]::WriteAllText((Join-Path $out 'profile_manifest.json'), (($manifest | ConvertTo-Json -Depth 5) + "`n"))
Write-Output "Pass36 WPR captures complete: $out"

