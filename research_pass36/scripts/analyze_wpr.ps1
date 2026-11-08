param(
  [Parameter(Mandatory=$true)][string]$EtlPath,
  [Parameter(Mandatory=$true)][string]$SymbolExe,
  [Parameter(Mandatory=$true)][string]$Image,
  [Parameter(Mandatory=$true)][int]$QColor,
  [Parameter(Mandatory=$true)][int]$Speed,
  [Parameter(Mandatory=$true)][string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$traceAssembly = 'C:\Program Files\HP\Overlay\Microsoft.Diagnostics.Tracing.TraceEvent.dll'
[void][Reflection.Assembly]::LoadFrom($traceAssembly)
$options = [Microsoft.Diagnostics.Tracing.Etlx.TraceLogOptions]::new()
$options.KeepAllEvents = $false
$options.LocalSymbolsOnly = $true
$trace = [Microsoft.Diagnostics.Tracing.Etlx.TraceLog]::OpenOrConvert($EtlPath, $options)
$processes = @($trace.Processes | Where-Object {
  $_.Name -eq 'avifenc' -and $_.CommandLine.Contains($Image) -and
  $_.CommandLine.Contains("--qcolor $QColor") -and $_.CommandLine.Contains("--speed $Speed")
})
$allAddresses = [System.Collections.Generic.HashSet[string]]::new()
$sampleStacks = [System.Collections.Generic.List[object]]::new()
$processCpuMs = 0.0
$sampledProcessCount = 0
$missingStacks = 0
$sampleIntervalMs = $trace.SampleProfileInterval.TotalMilliseconds

foreach ($process in $processes) {
  $processCpuMs += $process.CPUMSec
  foreach ($event in $process.EventsInProcess) {
    if ($event.EventName -notmatch 'PerfInfo/Sample') { continue }
    $sampledProcessCount++
    $frames = [System.Collections.Generic.List[string]]::new()
    $frame = $trace.GetCallStackForEvent($event)
    $depth = 0
    while ($frame -and $depth -lt 100) {
      $address = [uint64]$frame.CodeAddress.Address
      $module = $process.LoadedModules.GetModuleContainingAddress($address, $event.TimeStampRelativeMSec)
      if ($module -and $module.Name -eq 'avifenc') {
        $peAddress = [uint64]0x140000000 + ($address - [uint64]$module.ImageBase)
        $hexAddress = '0x{0:X}' -f $peAddress
        [void]$allAddresses.Add($hexAddress)
        $frames.Add($hexAddress)
      }
      $frame = $frame.Caller
      $depth++
    }
    if ($frames.Count) {
      $sampleStacks.Add(@{ process_id = $process.ProcessID; addresses = @($frames); leaf = $frames[0] })
    } else {
      $missingStacks++
    }
  }
}

$addresses = @($allAddresses | Sort-Object)
$symbols = @{}
$addr2line = 'C:\Strawberry\c\bin\addr2line.exe'
for ($start = 0; $start -lt $addresses.Count; $start += 250) {
  $end = [Math]::Min($start + 249, $addresses.Count - 1)
  $batch = @($addresses[$start..$end])
  $resolved = @(& $addr2line -f -C -e $SymbolExe $batch)
  for ($i = 0; $i -lt $batch.Count; $i++) {
    $symbols[$batch[$i]] = @{ function = [string]$resolved[$i * 2]; source = [string]$resolved[$i * 2 + 1] }
  }
}

$counts = @{}
$symbolizedSamples = 0
foreach ($stack in $sampleStacks) {
  $seen = [System.Collections.Generic.HashSet[string]]::new()
  foreach ($address in $stack.addresses) {
    $symbol = $symbols[$address]
    if ($symbol.function -and $symbol.function -ne '??') {
      if ($seen.Add($symbol.function)) {
        if (-not $counts.ContainsKey($symbol.function)) {
          $counts[$symbol.function] = @{ function = $symbol.function; source = $symbol.source; inclusive = 0; exclusive = 0 }
        }
        $counts[$symbol.function].inclusive++
      }
    }
  }
  if ($seen.Count) { $symbolizedSamples++ }
  $leaf = $symbols[$stack.leaf]
  if ($leaf.function -and $leaf.function -ne '??') {
    if (-not $counts.ContainsKey($leaf.function)) {
      $counts[$leaf.function] = @{ function = $leaf.function; source = $leaf.source; inclusive = 0; exclusive = 0 }
    }
    $counts[$leaf.function].exclusive++
  }
}

$hotspots = @($counts.Values | ForEach-Object {
  [ordered]@{
    function = $_.function
    source = $_.source
    inclusive_samples = $_.inclusive
    inclusive_cpu_ms = [Math]::Round($_.inclusive * $sampleIntervalMs, 3)
    inclusive_percent = [Math]::Round(100.0 * $_.inclusive / [Math]::Max(1, $sampledProcessCount), 2)
    exclusive_samples = $_.exclusive
    exclusive_cpu_ms = [Math]::Round($_.exclusive * $sampleIntervalMs, 3)
    exclusive_percent = [Math]::Round(100.0 * $_.exclusive / [Math]::Max(1, $sampledProcessCount), 2)
  }
} | Sort-Object { $_.inclusive_samples } -Descending)

$result = [ordered]@{
  etl = [IO.Path]::GetFullPath($EtlPath)
  image = $Image
  qcolor = $QColor
  speed = $Speed
  trace_event_assembly = $traceAssembly
  sampling_interval_ms = $sampleIntervalMs
  process_count = $processes.Count
  process_cpu_ms = [Math]::Round($processCpuMs, 3)
  process_sample_count = $sampledProcessCount
  sampled_cpu_ms = [Math]::Round($sampledProcessCount * $sampleIntervalMs, 3)
  samples_with_encoder_stack = $sampleStacks.Count
  samples_without_encoder_stack = $missingStacks
  samples_with_symbols = $symbolizedSamples
  unique_encoder_addresses = $addresses.Count
  hotspots = $hotspots
}
$json = $result | ConvertTo-Json -Depth 5
[IO.File]::WriteAllText($OutputPath, $json + "`n")
$json
