$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Pass = Join-Path $Root 'research_pass40'
$P35 = Join-Path $Root 'research_pass35'
$P39 = Join-Path $Root 'research_pass39'
$Source = Join-Path $Pass 'outputs\sources\libaom'
$BuildRoot = Join-Path $Pass 'outputs\build'
$InstallRoot = Join-Path $Pass 'outputs\install'
$AomCommit = '10aece4157eb79315da205f39e19bf6ab3ee30d0'

if (-not (Test-Path $Source)) {
    git clone --shared --no-checkout (Join-Path $P35 'sources\libaom') $Source
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone the local pinned libaom source.' }
    git -C $Source checkout --detach $AomCommit
    if ($LASTEXITCODE -ne 0) { throw 'Could not check out pinned libaom.' }
}
if ((git -C $Source rev-parse HEAD).Trim() -ne $AomCommit) { throw 'Wrong libaom revision.' }
$QMapPatch = Join-Path $P39 'patches\qindex-map.patch'
$SpatialPatch = Join-Path $Pass 'patches\spatial-rebalance.patch'
git -C $Source apply --check $QMapPatch
if ($LASTEXITCODE -eq 0) {
    git -C $Source apply $QMapPatch
} else {
    git -C $Source apply --reverse --check $QMapPatch
    if ($LASTEXITCODE -ne 0) { throw 'Pass 39 qmap patch does not match the source.' }
}
git -C $Source apply --check $SpatialPatch
if ($LASTEXITCODE -eq 0) {
    git -C $Source apply $SpatialPatch
} else {
    git -C $Source apply --reverse --check $SpatialPatch
    if ($LASTEXITCODE -ne 0) { throw 'Pass 40 intervention patch does not match the source.' }
}

function Build-Aom($Name, $Defines) {
    $Build = Join-Path $BuildRoot $Name
    $Install = Join-Path $InstallRoot $Name
    New-Item -ItemType Directory -Force $BuildRoot, $InstallRoot | Out-Null
    $Flags = if ($Defines) { '-D' + ($Defines -join ' -D') } else { '' }
    cmake -S $Source -B $Build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_ASM_NASM_COMPILER=nasm '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG' '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG' '-DCMAKE_ASM_NASM_FLAGS=-DWIN64' -DAOM_TARGET_CPU=x86_64 -DENABLE_NASM=ON -DENABLE_TOOLS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_TESTS=OFF -DCONFIG_AV1_ENCODER=1 -DCONFIG_AV1_DECODER=1 -DBUILD_SHARED_LIBS=OFF ("-DCMAKE_C_FLAGS=$Flags")
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $Name" }
    cmake --build $Build --parallel 6
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed: $Name" }
    cmake --install $Build --prefix $Install
    if ($LASTEXITCODE -ne 0) { throw "CMake install failed: $Name" }
}

Build-Aom 'control' @()
Build-Aom 'experiment' @('PBC_SPATIAL_REBALANCE')
Build-Aom 'experiment-qmap' @('PBC_SPATIAL_REBALANCE', 'PBC_DUMP_QMAP')
