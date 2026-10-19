param(
    [Parameter(Mandatory=$true)][ValidateSet('on','off')][string]$Arm
)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Key = Join-Path $Project 'outofmemory.pem'
$HostName = 'outofmemory@123.37.28.167'
$RemoteRoot = '/data/multipolicy_v53_20260928/distancing'
$RemoteArm = "$RemoteRoot/$Arm"
$Cdir = Join-Path 'C:\Users\Administrator\Documents\kw26_a100_recovery_20260926\multipolicy_v53_20260928' "distancing\$Arm"
$Gdir = Join-Path $Project "output\recovery_20260928\multipolicy_v53\distancing\$Arm"

function Invoke-Remote([string]$Command) {
    $result = & ssh -i $Key -p 10022 -o BatchMode=yes -o ConnectTimeout=10 $HostName $Command
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $Command" }
    return ($result -join "`n")
}
function Confirm-Hash([string]$Path, [string]$Expected) {
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) { throw "SHA256 mismatch: $Path" }
}

$exitCode = Invoke-Remote "cat '$RemoteArm/launch.exitcode'"
if ($exitCode.Trim() -ne '0') { throw "Distancing $Arm is not complete" }
Invoke-Remote "test -s '$RemoteArm/summary.json' && test -s '$RemoteArm/graph_restored.marker' && test ! -e '$RemoteArm/graph_backup/SHA256SUMS'" | Out-Null

# The exporter reads canonical receipts and current graph POI/home coordinates.
# Its one-day smoke was validated on real P010 receipts before any DIST arm.
$export = "cd /data/pilot_repo_20260927 && source /data/venv/bin/activate && source <(grep '^export NEO4J_URI=' tools/run_p013_ruler.sh | head -1) && export PYTHONPATH=/data/pilot_repo_20260927 && python scripts/report/export_receipt_coordinates.py --roster '$RemoteRoot/roster.json' --metrics-dir '$RemoteArm/metrics' --start 2020-11-24 --end 2020-11-26 --out '$RemoteArm/receipt_coordinates.jsonl' --coordinates-json '$RemoteArm/poi_coordinates.json'"
Invoke-Remote $export | Write-Output
Invoke-Remote "cd '$RemoteArm' && sha256sum receipt_coordinates.jsonl receipt_coordinates.jsonl.manifest.json poi_coordinates.json > coordinates_prebackup.sha256" | Out-Null

New-Item -ItemType Directory -Force -Path $Cdir, $Gdir | Out-Null
$manifest = Join-Path $Cdir 'coordinates_prebackup.sha256'
& scp -i $Key -P 10022 -o BatchMode=yes -o ConnectTimeout=10 "${HostName}:$RemoteArm/coordinates_prebackup.sha256" $manifest
if ($LASTEXITCODE -ne 0) { throw 'SCP coordinate manifest failed' }
$names = @('receipt_coordinates.jsonl','receipt_coordinates.jsonl.manifest.json','poi_coordinates.json')
$hashes = @{}
foreach ($line in Get-Content -LiteralPath $manifest) {
    if ($line -notmatch '^([0-9a-f]{64})\s+(.+)$') { throw "Invalid SHA line: $line" }
    $hashes[$Matches[2].Trim()] = $Matches[1]
}
foreach ($name in $names) {
    if (-not $hashes.ContainsKey($name)) { throw "Missing SHA for $name" }
    $cfile = Join-Path $Cdir $name
    & scp -i $Key -P 10022 -o BatchMode=yes -o ConnectTimeout=10 "${HostName}:$RemoteArm/$name" $cfile
    if ($LASTEXITCODE -ne 0) { throw "SCP failed: $name" }
    Confirm-Hash $cfile $hashes[$name]
    $gfile = Join-Path $Gdir $name
    Copy-Item -LiteralPath $cfile -Destination $gfile -Force
    Confirm-Hash $gfile $hashes[$name]
}
$gManifest = Join-Path $Gdir 'coordinates_prebackup.sha256'
Copy-Item -LiteralPath $manifest -Destination $gManifest -Force
Confirm-Hash $gManifest (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash
$serverManifestHash = (Invoke-Remote "sha256sum '$RemoteArm/coordinates_prebackup.sha256'").Split(' ')[0]
Confirm-Hash $manifest $serverManifestHash
$marker = "coordinate ledger/manifest/POI map copied to C: and G: with SHA256 at $(Get-Date -Format o)"
Invoke-Remote "printf '%s\n' '$marker' > '$RemoteArm/coordinates_external_copy_verified.txt'" | Out-Null
Write-Output "VERIFIED distancing/$Arm receipt coordinates on server/C:/G: before graph dump"
