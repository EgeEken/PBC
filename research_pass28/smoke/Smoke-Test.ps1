param(
    [Parameter(Mandatory=$true)][string]$CombinedArchive,
    [Parameter(Mandatory=$true)][string]$DecoderArchive,
    [Parameter(Mandatory=$true)][string]$OneFile,
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Report = (Join-Path (Get-Location) 'PBC32-smoke-results.json')
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$fixtures = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'fixtures.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$declared = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$rows = New-Object System.Collections.Generic.List[object]
$root = Join-Path ([IO.Path]::GetTempPath()) ('PBC28 ' + [char]0x00F6 + ' ' + [char]0x65E5 + ' ' + [guid]::NewGuid())
[IO.Directory]::CreateDirectory($root) | Out-Null
$old = @{}
foreach ($key in @('PATH','PYTHONHOME','PYTHONPATH','VIRTUAL_ENV','NUMBA_CACHE_DIR','TEMP','TMP','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')) {
    $old[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
}
function Hash-Bytes([byte[]]$Data) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($algorithm.ComputeHash($Data))).Replace('-','').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
function Invoke-Pbc([string]$Executable, [string[]]$Arguments, [bool]$ExpectSuccess = $true) {
    $before = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $stdout = & $Executable @Arguments 2> (Join-Path $root 'stderr.txt')
        $code = $LASTEXITCODE
    } finally { $ErrorActionPreference = $before }
    $stderr = [IO.File]::ReadAllText((Join-Path $root 'stderr.txt'))
    if ($ExpectSuccess -and $code -ne 0) { throw "Command failed ($code): $Arguments $stderr" }
    if (-not $ExpectSuccess -and ($code -eq 0 -or $stderr -match 'Traceback')) { throw "Invalid failure response: $Arguments $code $stderr" }
    return @{ code=$code; stdout=($stdout -join "`n"); stderr=$stderr }
}
function Check-Pixels([string]$Path, $Expected) {
    $bitmap = New-Object Drawing.Bitmap($Path)
    try {
        if ($bitmap.Width -ne $Expected.size[0] -or $bitmap.Height -ne $Expected.size[1]) { throw "Wrong dimensions: $Path" }
        $channels = $Expected.mode.Length
        $bytes = New-Object byte[] ($bitmap.Width*$bitmap.Height*$channels)
        $i = 0
        for ($y=0; $y -lt $bitmap.Height; $y++) {
            for ($x=0; $x -lt $bitmap.Width; $x++) {
                $c = $bitmap.GetPixel($x,$y)
                $bytes[$i++]=$c.R; $bytes[$i++]=$c.G; $bytes[$i++]=$c.B
                if ($channels -eq 4) { $bytes[$i++]=$c.A }
            }
        }
        $hash = Hash-Bytes $bytes
        if ($hash -ne $Expected.pixel_sha256) { throw "Pixel hash mismatch: $Path $hash" }
        return $hash
    } finally { $bitmap.Dispose() }
}
$status = 'FAIL'
$failure = $null
try {
    foreach ($item in @(@('combined',$CombinedArchive),@('decoder',$DecoderArchive),@('decoder_onefile',$OneFile))) {
        $entry = @($declared.artifacts | Where-Object { $_.kind -eq $item[0] })
        if ($entry.Count -ne 1) { throw "Missing checksum: $($item[0])" }
        $hash = (Get-FileHash -LiteralPath $item[1] -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $entry[0].sha256) { throw "Archive SHA-256 mismatch: $($item[1])" }
        $rows.Add(@{test='archive_sha256';kind=$item[0];sha256=$hash;pass=$true})
    }
    Expand-Archive -LiteralPath $CombinedArchive -DestinationPath (Join-Path $root 'combined')
    Expand-Archive -LiteralPath $DecoderArchive -DestinationPath (Join-Path $root 'decoder')
    [IO.Directory]::CreateDirectory((Join-Path $root 'onefile')) | Out-Null
    Copy-Item -LiteralPath $OneFile -Destination (Join-Path $root 'onefile/pbc-decode.exe')
    [IO.Directory]::CreateDirectory((Join-Path $root 'temp')) | Out-Null
    $env:PATH = Join-Path $env:SystemRoot 'System32'
    $env:PYTHONHOME=$null; $env:PYTHONPATH=$null; $env:VIRTUAL_ENV=$null
    $env:NUMBA_CACHE_DIR=Join-Path $root 'cache'
    $env:TEMP=Join-Path $root 'temp'; $env:TMP=$env:TEMP
    foreach ($key in @('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')) { [Environment]::SetEnvironmentVariable($key,'1','Process') }
    foreach ($name in @('python.exe','gcc.exe','cl.exe')) {
        if (Get-Command $name -CommandType Application -ErrorAction SilentlyContinue) { throw "Development tool still on PATH: $name" }
    }
    $apps = [ordered]@{
        combined = Join-Path $root 'combined/pbc/pbc.exe'
        decoder = Join-Path $root 'decoder/pbc-decode/pbc-decode.exe'
        decoder_onefile = Join-Path $root 'onefile/pbc-decode.exe'
    }
    foreach ($name in $apps.Keys) {
        $version = Invoke-Pbc $apps[$name] @('--version')
        if (-not $declared.version -or $version.stdout -ne ('pbc '+$declared.version+'; stream v0')) { throw 'Unexpected version' }
        foreach ($fixture in $fixtures.streams) {
            $input = Join-Path $root ($fixture.name+'.pbc')
            $output = Join-Path $root ($name+'-'+$fixture.name+'.png')
            [IO.File]::WriteAllBytes($input,[Convert]::FromBase64String($fixture.data))
            $arguments = @($input,$output)
            if ($name -eq 'combined') { $arguments = @('decode')+$arguments }
            $null = Invoke-Pbc $apps[$name] $arguments
            $hash = Check-Pixels $output $fixture
            $beforeHash = (Get-FileHash -LiteralPath $output).Hash
            $null = Invoke-Pbc $apps[$name] $arguments $false
            if ((Get-FileHash -LiteralPath $output).Hash -ne $beforeHash) { throw 'Overwrite protection failed' }
            $rows.Add(@{test='known_decode_pixels_overwrite';app=$name;fixture=$fixture.name;sha256=$hash;pass=$true})
        }
        foreach ($fixture in $fixtures.invalid) {
            $input = Join-Path $root $fixture.name
            [IO.File]::WriteAllBytes($input,[Convert]::FromBase64String($fixture.data))
            $output = Join-Path $root 'invalid.png'
            $arguments = @($input,$output)
            if ($name -eq 'combined') { $arguments = @('decode')+$arguments }
            $null = Invoke-Pbc $apps[$name] $arguments $false
            if (Test-Path -LiteralPath $output) { throw 'Malformed stream wrote an output' }
            $rows.Add(@{test='malformed_rejection';app=$name;fixture=$fixture.name;pass=$true})
        }
    }
    foreach ($fixture in $fixtures.inputs) {
        $input = Join-Path $root ($fixture.name+'.png')
        [IO.File]::WriteAllBytes($input,[Convert]::FromBase64String($fixture.png))
        foreach ($preset in @('compression','balanced','quality','high_quality')) {
            $output = Join-Path $root ($fixture.name+'-'+$preset+'.pbc')
            $null = Invoke-Pbc $apps.combined @('encode',$input,$output,'--preset',$preset)
            $firstHash = (Get-FileHash -LiteralPath $output).Hash
            $null = Invoke-Pbc $apps.combined @('encode',$input,$output,'--preset',$preset) $false
            if ((Get-FileHash -LiteralPath $output).Hash -ne $firstHash) { throw 'Encode overwrite protection failed' }
            $null = Invoke-Pbc $apps.combined @('encode',$input,$output,'--preset',$preset,'--force')
            if ((Get-FileHash -LiteralPath $output).Hash -ne $firstHash) { throw 'Encoding is not deterministic in this environment' }
            $decoded = Join-Path $root ($fixture.name+'-'+$preset+'.png')
            $null = Invoke-Pbc $apps.decoder @($output,$decoded)
            $expected = $fixture.expected.$preset
            $hash = Check-Pixels $decoded $expected
            $rows.Add(@{test='encode_determinism_pixels';mode=$fixture.mode;preset=$preset;pixel_sha256=$hash;stream_sha256=$firstHash.ToLowerInvariant();matches_recorded_windows_stream=($firstHash.ToLowerInvariant() -eq $expected.stream_sha256);pass=$true})
        }
    }
    if (-not (Get-ChildItem -LiteralPath (Join-Path $root 'cache') -Recurse -Filter '*.nbc')) { throw 'Persistent cache was not populated' }
    if (Get-ChildItem -LiteralPath $env:TEMP -Directory -Filter '_MEI*') { throw 'One-file temporary extraction was not cleaned' }
    $rows.Add(@{test='persistent_cache_and_onefile_cleanup';pass=$true})
    $status='PASS'
} catch {
    $failure=$_.Exception.Message
} finally {
    foreach ($key in $old.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],'Process') }
    $result=[ordered]@{status=$status;failure=$failure;utc=[DateTime]::UtcNow.ToString('o');os=[Environment]::OSVersion.VersionString;powershell=$PSVersionTable.PSVersion.ToString();process_bits=[IntPtr]::Size*8;admin=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator);path_isolation=$true;clean_consumer_machine_proven=$false;working_directory=$root;checks=$rows.ToArray()}
    $text=$result | ConvertTo-Json -Depth 10
    [IO.File]::WriteAllText([IO.Path]::GetFullPath($Report),$text,(New-Object Text.UTF8Encoding($false)))
    Write-Output "$status; $($rows.Count) checks; report: $Report; work files: $root"
    if ($failure) { Write-Output $failure }
}
if ($status -ne 'PASS') { exit 1 }
