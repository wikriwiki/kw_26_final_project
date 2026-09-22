param(
    [string]$Out = "$env:USERPROFILE\sim_output",
    [string]$Project = "G:\내 드라이브\Kw\final_project"
)

$ErrorActionPreference = "Continue"
$logDir = Join-Path $Project "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$log = Join-Path $logDir "auto_finalize_v3.log"

function Log($m) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $m`r`n"
    for ($i = 0; $i -lt 8; $i++) {
        try {
            [System.IO.File]::AppendAllText($log, $line, [System.Text.Encoding]::UTF8)
            return
        } catch { Start-Sleep -Milliseconds 250 }
    }
}

Log "daemon armed v3 (5-day) — waiting for day_2026-05-23.jsonl (signal that 5/22 Night2 finished)"

$signalFile = Join-Path $Out "metrics\day_2026-05-23.jsonl"
$d22File    = Join-Path $Out "metrics\day_2026-05-22.jsonl"

while ($true) {
    if (Test-Path $signalFile) {
        $sizeNext = (Get-Item $signalFile).Length
        if ($sizeNext -gt 0) {
            $sizeLast = if (Test-Path $d22File) { (Get-Item $d22File).Length } else { 0 }
            Log "5/23 jsonl detected (size=$sizeNext) — 5/22 size=$sizeLast. Killing sim."
            Get-Process python3.13 -EA SilentlyContinue | Stop-Process -Force
            Start-Sleep -Seconds 30
            break
        }
    }
    Start-Sleep -Seconds 60
}

Set-Location $Project
Log "=== build viz ==="

# 0.5) KNOWS_POI 5/20 보정 (visited Memory 누락분 단골화 반영) — sim kill 후라 충돌 없음
Log "step0.5 KNOWS_POI 5/20 보정"
try {
    $h = @{ Authorization = "Basic " + [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("neo4j:neo4j_poc_2026")); "Content-Type" = "application/json" }
    $cyFix = Get-Content (Join-Path $Project "scripts\sim\fix_knows_poi_0520.cypher") -Encoding utf8 -Raw
    # 주석/세미콜론 제거 (단일 statement)
    $cyFix = ($cyFix -split "`n" | Where-Object { $_ -notmatch '^\s*//' }) -join "`n"
    $cyFix = $cyFix.TrimEnd().TrimEnd(';')
    $body = @{ statements = @(@{ statement = $cyFix }) } | ConvertTo-Json -Depth 5
    $r = Invoke-RestMethod -Uri "http://localhost:7474/db/neo4j/tx/commit" -Method POST -Headers $h -Body $body -TimeoutSec 300
    if ($r.errors.Count -gt 0) { Log ("step0.5 ERROR: " + $r.errors[0].message) }
    else { Log ("step0.5 done — KNOWS_POI updated=" + $r.results[0].data[0].row[0]) }
} catch { Log ("step0.5 EXCEPTION: " + $_) }

# 1) export_visualization
Log "step1 export_visualization"
$out1 = Join-Path $logDir "export_viz.log"
& python scripts/sim/export_visualization.py --start 2026-05-18 --days 5 *>&1 | Out-File $out1 -Encoding utf8
Log "step1 done (exit=$LASTEXITCODE)"

# 2) build_standalone_html
Log "step2 build_standalone_html"
$out2 = Join-Path $logDir "build_html.log"
& python scripts/sim/build_standalone_html.py *>&1 | Out-File $out2 -Encoding utf8
Log "step2 done (exit=$LASTEXITCODE)"

# 3) final report (markdown + charts, skip-interview)
Log "step3 generate_final_report (skip-interview)"
$out3 = Join-Path $logDir "final_report.log"
& python scripts/sim/generate_final_report.py --start 2026-05-18 --days 5 --policy-from 2026-05-20 --out output/sim/report/FINAL_REPORT_5D.md --skip-interview *>&1 | Out-File $out3 -Encoding utf8
Log "step3 done (exit=$LASTEXITCODE)"

# 4) precise L2 spillover DID (NEW)
Log "step4 precise_l2_did"
$out4 = Join-Path $logDir "precise_l2.log"
& python scripts/sim/precise_l2_did.py --start 2026-05-18 --days 5 --policy-from 2026-05-20 --baseline-days "2026-05-19" --out output/sim/report/L2_PRECISE_5D.md *>&1 | Out-File $out4 -Encoding utf8
Log "step4 done (exit=$LASTEXITCODE)"

# 5) caveats append (NEW) — CAVEATS_5D.md 본문을 FINAL_REPORT_5D.md 뒤에 append
Log "step5 append caveats"
$caveatsPath = Join-Path $Project "output\sim\report\CAVEATS_5D.md"
$reportPath  = Join-Path $Project "output\sim\report\FINAL_REPORT_5D.md"
if ((Test-Path $reportPath) -and (Test-Path $caveatsPath)) {
    $sep = "`r`n`r`n---`r`n`r`n# 부록 A. 시뮬레이션 데이터 한계 및 해석 주의사항`r`n`r`n"
    $caveatsBody = Get-Content $caveatsPath -Encoding utf8 -Raw
    # 첫 H1 줄 제거하고 본문만 append
    $caveatsBody = $caveatsBody -replace '^# [^\r\n]*[\r\n]+', ''
    Add-Content -Path $reportPath -Value ($sep + $caveatsBody) -Encoding utf8
    Log "step5 caveats appended to FINAL_REPORT_5D.md"
} else {
    Log "step5 SKIP — report=$reportPath exists=$(Test-Path $reportPath) caveats=$caveatsPath exists=$(Test-Path $caveatsPath)"
}

# 6) precise_l2 merge to FINAL_REPORT_5D (NEW) — L2_PRECISE_5D.md 본문도 append
Log "step6 append precise_l2"
$precisePath = Join-Path $Project "output\sim\report\L2_PRECISE_5D.md"
if ((Test-Path $reportPath) -and (Test-Path $precisePath)) {
    $sep2 = "`r`n`r`n---`r`n`r`n# 부록 B. L2 spillover 정밀 측정 (4-case 분리)`r`n`r`n"
    $preciseBody = Get-Content $precisePath -Encoding utf8 -Raw
    $preciseBody = $preciseBody -replace '^# [^\r\n]*[\r\n]+', ''
    Add-Content -Path $reportPath -Value ($sep2 + $preciseBody) -Encoding utf8
    Log "step6 precise_l2 appended"
} else {
    Log "step6 SKIP — precise=$precisePath exists=$(Test-Path $precisePath)"
}

# 7) full report with interview (vLLM-dependent — vLLM 죽었으면 skip 가능)
Log "step7 generate_final_report (with interview)"
$out7 = Join-Path $logDir "final_report_full.log"
$env:LLM_MODE = "qwen14b"
& python scripts/sim/generate_final_report.py --start 2026-05-18 --days 5 --policy-from 2026-05-20 --out output/sim/report/FINAL_REPORT_5D_FULL.md *>&1 | Out-File $out7 -Encoding utf8
Log "step7 done (exit=$LASTEXITCODE)"

# 7-b) full report에도 caveats + precise 머지
$reportFullPath = Join-Path $Project "output\sim\report\FINAL_REPORT_5D_FULL.md"
if ((Test-Path $reportFullPath) -and (Test-Path $caveatsPath) -and (Test-Path $precisePath)) {
    $sep3 = "`r`n`r`n---`r`n`r`n# 부록 A. 시뮬레이션 데이터 한계 및 해석 주의사항`r`n`r`n"
    $caveatsBody2 = (Get-Content $caveatsPath -Encoding utf8 -Raw) -replace '^# [^\r\n]*[\r\n]+', ''
    $sep4 = "`r`n`r`n---`r`n`r`n# 부록 B. L2 spillover 정밀 측정 (4-case 분리)`r`n`r`n"
    $preciseBody2 = (Get-Content $precisePath -Encoding utf8 -Raw) -replace '^# [^\r\n]*[\r\n]+', ''
    Add-Content -Path $reportFullPath -Value ($sep3 + $caveatsBody2 + $sep4 + $preciseBody2) -Encoding utf8
    Log "step7b appended caveats+precise to FULL"
}

# 8) ChatGPT-ready 통합 페이지 요약 (간단 요약)
Log "=== ALL DONE ==="
Log "outputs:"
Log "  - output/sim/report/FINAL_REPORT_5D.md (+html, charts in .d/)"
Log "  - output/sim/report/FINAL_REPORT_5D_FULL.md (+인터뷰)"
Log "  - output/sim/report/L2_PRECISE_5D.md (+.json)"
Log "  - output/sim/report/CAVEATS_5D.md"
Log "  - output/sim/viz/* (export_visualization 출력)"
Log "  - output/sim/sim_standalone.html (build_standalone_html 출력)"
