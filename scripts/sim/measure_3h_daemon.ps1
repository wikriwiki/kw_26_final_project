$Project = "G:\내 드라이브\Kw\final_project"
$Out = "$env:USERPROFILE\sim_output"
$log = Join-Path $Project "logs\measure_3h.log"
function Log($m) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $log -Value "[$ts] $m" -Encoding utf8
}
Log "measure daemon armed — every 3h"
while ($true) {
    Start-Sleep -Seconds 10800
    Log "=== 3h tick ==="
    foreach ($d in 18..24) {
        $f = Join-Path $Out "metrics\day_2026-05-$d.jsonl"
        if (Test-Path $f) {
            $n = (Get-Content $f | Measure-Object -Line).Lines
            $pct = [math]::Round($n / 14560 * 100, 1)
            Log ("  day_2026-05-{0}: {1}/14,560 ({2}%)" -f $d, $n, $pct)
        }
    }
    # vLLM ping
    try {
        $vstat = Invoke-RestMethod -Uri "http://localhost:8000/v1/models" -TimeoutSec 5 -EA Stop
        Log "  vLLM OK (models=$(($vstat.data | ForEach-Object id) -join ','))"
    } catch { Log "  vLLM ping FAIL" }
}