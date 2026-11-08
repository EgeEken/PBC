$ErrorActionPreference = 'Stop'
$Pass = Split-Path $PSScriptRoot -Parent
$Repo = Split-Path $Pass -Parent
$Src = Join-Path $Pass 'sources'
$Bld = Join-Path $Pass 'build'
$Ins = Join-Path $Pass 'install'
$Out = Join-Path $Pass 'outputs\binaries'
$Patch = Join-Path $Pass 'patches\libaom-profile-counters.patch'
$AomCommit = '10aece4157eb79315da205f39e19bf6ab3ee30d0'
$AvifCommit = '1aadfad932c98c069a1204261b1856f81f3bc199'
New-Item -ItemType Directory -Force $Src,$Bld,$Ins,$Out | Out-Null

function Get-PinnedSource($Url,$Path,$Commit,$Tag) {
    if (-not (Test-Path $Path)) {
        git clone --no-checkout $Url $Path
        git -C $Path fetch --depth 1 origin $Tag
        git -C $Path checkout --detach $Commit
    }
    if ((git -C $Path rev-parse HEAD).Trim() -ne $Commit) { throw "Wrong source revision: $Path" }
}
function Build-Aom($Source,$Name,$Prefix,$Profile) {
    $Build = Join-Path $Bld "aom-$Name"
    $CmakeArgs = @('-S',$Source,'-B',$Build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=gcc','-DCMAKE_CXX_COMPILER=g++','-DCMAKE_ASM_NASM_COMPILER=nasm','-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG','-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG','-DCMAKE_ASM_NASM_FLAGS=-DWIN64','-DAOM_TARGET_CPU=x86_64','-DENABLE_NASM=ON','-DENABLE_TOOLS=OFF','-DENABLE_EXAMPLES=OFF','-DENABLE_TESTS=OFF','-DCONFIG_AV1_ENCODER=1','-DCONFIG_AV1_DECODER=1','-DBUILD_SHARED_LIBS=OFF')
    if ($Profile) { $CmakeArgs += @('-DCMAKE_C_FLAGS=-DPBC_PROFILE','-DCONFIG_COLLECT_PARTITION_STATS=2','-DCONFIG_COLLECT_COMPONENT_TIMING=1') }
    cmake @Args
    cmake --build $Build --parallel 6
    cmake --install $Build --prefix $Prefix
}
function Link-Avif($AomPrefix,$Name) {
    $Build = Join-Path $Bld 'libavif-reproduction'
    $CmakeArgs = @('-S',(Join-Path $Src 'libavif'),'-B',$Build,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=gcc','-DCMAKE_CXX_COMPILER=g++','-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG','-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG','-DBUILD_SHARED_LIBS=OFF','-DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=TRUE','-DAVIF_CODEC_AOM=SYSTEM','-DAVIF_CODEC_AOM_ENCODE=ON','-DAVIF_CODEC_AOM_DECODE=ON',('-DAOM_INCLUDE_DIR='+(Join-Path $AomPrefix 'include')),('-DAOM_LIBRARY='+(Join-Path $AomPrefix 'lib\libaom.a')),'-DAVIF_LIBYUV=LOCAL','-DAVIF_LIBSHARPYUV=LOCAL','-DAVIF_JPEG=LOCAL','-DAVIF_ZLIBPNG=LOCAL','-DAVIF_BUILD_APPS=ON','-DAVIF_BUILD_TESTS=OFF')
    cmake @Args
    cmake --build $Build --target avifenc avifdec --parallel 6
    $Dest = Join-Path $Out $Name
    New-Item -ItemType Directory -Force $Dest | Out-Null
    Copy-Item (Join-Path $Build 'avifenc.exe') $Dest -Force
    Copy-Item (Join-Path $Build 'avifdec.exe') $Dest -Force
    Copy-Item 'C:\Strawberry\c\bin\libgcc_s_seh-1.dll','C:\Strawberry\c\bin\libwinpthread-1.dll' $Dest -Force
}

$AomRef = Join-Path $Src 'libaom-reference'
$AomResearch = Join-Path $Src 'libaom-research'
Get-PinnedSource 'https://aomedia.googlesource.com/aom' $AomRef $AomCommit 'v3.12.1'
Get-PinnedSource 'https://aomedia.googlesource.com/aom' $AomResearch $AomCommit 'v3.12.1'
Get-PinnedSource 'https://github.com/AOMediaCodec/libavif.git' (Join-Path $Src 'libavif') $AvifCommit 'v1.3.0'
git -C $AomResearch apply --check $Patch 2>$null
if ($LASTEXITCODE -eq 0) { git -C $AomResearch apply $Patch }
else {
    git -C $AomResearch apply --reverse --check $Patch 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Research instrumentation patch does not match the pinned source.' }
}

$RefPrefix = Join-Path $Ins 'reference'
$OffPrefix = Join-Path $Ins 'research-off'
$ProfilePrefix = Join-Path $Ins 'profile'
Build-Aom $AomRef 'reference' $RefPrefix $false
Build-Aom $AomResearch 'research-off' $OffPrefix $false
Build-Aom $AomResearch 'profile' $ProfilePrefix $true
Link-Avif $RefPrefix 'reference'
Copy-Item (Join-Path $Out 'reference\avifenc.exe') (Join-Path $Out 'reference-avifenc.exe') -Force
Copy-Item (Join-Path $Out 'reference\avifdec.exe') (Join-Path $Out 'reference-avifdec.exe') -Force
Link-Avif $OffPrefix 'research-off'
Copy-Item (Join-Path $Out 'research-off\avifenc.exe') (Join-Path $Out 'research-off-avifenc.exe') -Force
Copy-Item (Join-Path $Out 'research-off\avifdec.exe') (Join-Path $Out 'research-off-avifdec.exe') -Force
Link-Avif $ProfilePrefix 'profile'
Copy-Item (Join-Path $Out 'profile\avifenc.exe') (Join-Path $Out 'profile-avifenc.exe') -Force
Copy-Item (Join-Path $Out 'profile\avifdec.exe') (Join-Path $Out 'profile-avifdec.exe') -Force
Write-Host 'Builds complete. Run the Python measurement scripts from the repository root.'

