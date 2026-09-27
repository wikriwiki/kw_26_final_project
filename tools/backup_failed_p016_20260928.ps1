$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Key = Join-Path $Project 'outofmemory.pem'
$HostName = 'outofmemory@123.37.28.167'
$RemoteRoot = '/data/multipolicy_v53_20260928/p016'
$Cdir = 'C:\Users\Administrator\Documents\kw26_a100_recovery_20260926\multipolicy_v53_20260928\p016\failed_prepatch'
$Gdir = Join-Path $Project 'output\recovery_20260928\multipolicy_v53\p016\failed_prepatch'

function Invoke-Remote([string]$Command) {
    $result = & ssh -i $Key -p 10022 -o BatchMode=yes -o ConnectTimeout=10 $HostName $Command
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $Command" }
    return ($result -join "`n")
}
function Copy-Remote([string]$Remote, [string]$Local) {
    & scp -i $Key -P 10022 -o BatchMode=yes -o ConnectTimeout=10 "${HostName}:$Remote" $Local
    if ($LASTEXITCODE -ne 0) { throw "SCP failed: $Remote" }
}
function Confirm-Hash([string]$Path, [string]$Expected) {
    $got = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($got -ne $Expected.ToLowerInvariant()) { throw "SHA256 mismatch: $Path" }
}
function Read-Sums([string]$Path) {
    $map = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -notmatch '^([0-9a-f]{64})\s+(.+)$') { throw "Invalid checksum line: $line" }
        $map[$Matches[2].Trim()] = $Matches[1]
    }
    return $map
}

if ((Invoke-Remote "cat '$RemoteRoot/failed_prepatch_snapshot.exitcode'").Trim() -ne '0') {
    throw 'Failed-arm snapshot is incomplete'
}
Invoke-Remote "cd '$RemoteRoot/failed_prepatch_graph_backup' && sha256sum -c SHA256SUMS" | Out-Null
Invoke-Remote "cd '$RemoteRoot' && sha256sum -c failed_prepatch_artifacts.sha256 failed_prepatch_snapshot.sha256" | Out-Null
New-Item -ItemType Directory -Force -Path $Cdir, $Gdir | Out-Null
$manifest = Join-Path $Cdir 'graph_SHA256SUMS'
Copy-Remote "$RemoteRoot/failed_prepatch_graph_backup/SHA256SUMS" $manifest
$graphSums = Read-Sums $manifest
$names = @('neo4j.dump','system.dump','config_plugins.tar.gz','neo4j_version.txt')
foreach ($name in $names) {
    if (-not $graphSums.ContainsKey($name)) { throw "Missing graph SHA: $name" }
    $file = Join-Path $Cdir $name
    Copy-Remote "$RemoteRoot/failed_prepatch_graph_backup/$name" $file
    Confirm-Hash $file $graphSums[$name]
}
$other = @('failed_prepatch_artifacts.sha256','failed_prepatch_artifacts.tar.gz',
           'failed_prepatch_snapshot.sha256','failed_prepatch_snapshot.json')
foreach ($name in $other) {
    Copy-Remote "$RemoteRoot/$name" (Join-Path $Cdir $name)
}
$artifactSums = Read-Sums (Join-Path $Cdir 'failed_prepatch_artifacts.sha256')
$snapshotSums = Read-Sums (Join-Path $Cdir 'failed_prepatch_snapshot.sha256')
Confirm-Hash (Join-Path $Cdir 'failed_prepatch_artifacts.tar.gz') $artifactSums['failed_prepatch_artifacts.tar.gz']
Confirm-Hash (Join-Path $Cdir 'failed_prepatch_snapshot.json') $snapshotSums['failed_prepatch_snapshot.json']
$allNames = @('graph_SHA256SUMS') + $names + $other
foreach ($name in $allNames) {
    $source = Join-Path $Cdir $name
    $dest = Join-Path $Gdir $name
    Copy-Item -LiteralPath $source -Destination $dest -Force
    Confirm-Hash $dest (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
}
$marker = "failed P016 prepatch graph and artifacts verified on server/C:/G: at $(Get-Date -Format o)"
Invoke-Remote "printf '%s\n' '$marker' > '$RemoteRoot/failed_prepatch_external_copy_verified.txt'" | Out-Null
Write-Output 'VERIFIED failed P016 prepatch graph and artifacts on server/C:/G:'
