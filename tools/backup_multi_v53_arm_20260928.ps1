param(
    [Parameter(Mandatory=$true)][ValidateSet('p010','distancing','p016','p014','p012')][string]$CaseId,
    [Parameter(Mandatory=$true)][ValidateSet('on','off')][string]$Arm
)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Key = Join-Path $Project 'outofmemory.pem'
$HostName = 'outofmemory@123.37.28.167'
$RemoteRoot = "/data/multipolicy_v53_20260928/$CaseId"
$RemoteArm = "$RemoteRoot/$Arm"
$Cdir = Join-Path 'C:\Users\Administrator\Documents\kw26_a100_recovery_20260926\multipolicy_v53_20260928' "$CaseId\$Arm"
$Gdir = Join-Path $Project "output\recovery_20260928\multipolicy_v53\$CaseId\$Arm"

function Invoke-Remote([string]$Command) {
    $result = & ssh -i $Key -p 10022 -o BatchMode=yes -o ConnectTimeout=10 $HostName $Command
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $Command" }
    return ($result -join "`n")
}
function Copy-Remote([string]$RemoteFile, [string]$LocalFile) {
    & scp -i $Key -P 10022 -o BatchMode=yes -o ConnectTimeout=10 "${HostName}:$RemoteFile" $LocalFile
    if ($LASTEXITCODE -ne 0) { throw "SCP failed: $RemoteFile" }
}
function Confirm-Hash([string]$Path, [string]$Expected) {
    $got = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($got -ne $Expected.ToLowerInvariant()) { throw "SHA256 mismatch: $Path" }
}
function Read-Sums([string]$Path) {
    $map = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^([0-9a-f]{64})\s+(.+)$') { $map[$Matches[2].Trim()] = $Matches[1] }
    }
    return $map
}

$exitCode = Invoke-Remote "cat '$RemoteArm/preserve.exitcode'"
if ($exitCode.Trim() -ne '0') { throw "Preserve step incomplete: $exitCode" }
Invoke-Remote "cd '$RemoteArm/graph_backup' && sha256sum -c SHA256SUMS" | Out-Null
Invoke-Remote "cd '$RemoteRoot' && sha256sum -c '${Arm}_artifacts.sha256'" | Out-Null
New-Item -ItemType Directory -Force -Path $Cdir, $Gdir | Out-Null

$graphNames = @('neo4j.dump','system.dump','config_plugins.tar.gz','neo4j_version.txt')
$remoteManifest = "$RemoteArm/graph_backup/SHA256SUMS"
$cManifest = Join-Path $Cdir 'graph_SHA256SUMS'
Copy-Remote $remoteManifest $cManifest
$graphSums = Read-Sums $cManifest
foreach ($name in $graphNames) {
    if (-not $graphSums.ContainsKey($name)) { throw "Missing graph SHA: $name" }
    $cfile = Join-Path $Cdir $name
    Copy-Remote "$RemoteArm/graph_backup/$name" $cfile
    Confirm-Hash $cfile $graphSums[$name]
}

$archive = "${Arm}_artifacts.tar.gz"
$shaFile = "${Arm}_artifacts.sha256"
$cArchiveSha = Join-Path $Cdir $shaFile
Copy-Remote "$RemoteRoot/$shaFile" $cArchiveSha
$archiveSums = Read-Sums $cArchiveSha
if (-not $archiveSums.ContainsKey($archive)) { throw "Missing archive SHA: $archive" }
$cArchive = Join-Path $Cdir $archive
Copy-Remote "$RemoteRoot/$archive" $cArchive
Confirm-Hash $cArchive $archiveSums[$archive]

$cOutputs = Join-Path $Cdir 'outputs.sha256'
Copy-Remote "$RemoteArm/outputs.sha256" $cOutputs
$outputsHash = (Invoke-Remote "sha256sum '$RemoteArm/outputs.sha256'").Split(' ')[0]
Confirm-Hash $cOutputs $outputsHash

$names = @('graph_SHA256SUMS') + $graphNames + @($shaFile,$archive,'outputs.sha256')
foreach ($name in $names) {
    $cfile = Join-Path $Cdir $name
    $gfile = Join-Path $Gdir $name
    Copy-Item -LiteralPath $cfile -Destination $gfile -Force
    $expected = (Get-FileHash -LiteralPath $cfile -Algorithm SHA256).Hash.ToLowerInvariant()
    Confirm-Hash $gfile $expected
}

# The archive is the immutable backup. Extract report inputs to G: and check
# each source hash so the local scorer can read the original ledger bytes.
$gArchive = Join-Path $Gdir $archive
& tar -xzf $gArchive -C $Gdir
if ($LASTEXITCODE -ne 0) { throw "Archive extraction failed: $gArchive" }
foreach ($line in Get-Content -LiteralPath $cOutputs) {
    if ($line -notmatch '^([0-9a-f]{64})\s+(.+)$') { throw "Invalid outputs SHA line: $line" }
    $expected = $Matches[1]
    $leaf = [System.IO.Path]::GetFileName($Matches[2])
    $extracted = Join-Path (Join-Path $Gdir $Arm) $leaf
    Confirm-Hash $extracted $expected
}

$marker = "server, C:, and G: copies verified by SHA256 at $(Get-Date -Format o); archive=$($archiveSums[$archive])"
Invoke-Remote "printf '%s\n' '$marker' > '$RemoteArm/external_copy_verified.txt'" | Out-Null
Write-Output "VERIFIED $CaseId/$Arm graph and complete output on server/C:/G:; marker written"
