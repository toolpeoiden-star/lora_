# ============================================================
# AI Hub 한국 음식 이미지 일괄 압축 해제 (7-Zip 버전)
# 한글 파일명 안전. 중첩 zip 자동 처리. 원본 zip 삭제.
# ============================================================
$ErrorActionPreference = "Continue"
$root = "C:\Users\user\Desktop\한국 음식 이미지\kfood"
$sevenZip = "C:\Program Files\7-Zip\7z.exe"

if (-not (Test-Path $sevenZip)) {
    Write-Host "❌ 7-Zip이 설치되지 않았습니다. 먼저 실행하세요:" -ForegroundColor Red
    Write-Host "   winget install 7zip.7zip" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $root)) {
    Write-Host "❌ 경로 없음: $root" -ForegroundColor Red
    exit 1
}

Set-Location $root
Write-Host "📂 작업 경로: $root" -ForegroundColor Cyan
Write-Host "🗜  도구: 7-Zip`n" -ForegroundColor Cyan

$passNum = 0
do {
    $passNum++
    # zip + 7z + tar + gz 모두 처리
    $archives = Get-ChildItem -Path $root -Include *.zip,*.7z,*.tar,*.tar.gz,*.tgz -Recurse -File
    if (-not $archives) {
        Write-Host "✅ 더 이상 압축 파일 없음. 완료." -ForegroundColor Green
        break
    }

    Write-Host "── Pass $passNum : $($archives.Count)개 압축 파일 ──" -ForegroundColor Yellow

    foreach ($arc in $archives) {
        $target = $arc.DirectoryName
        $sizeMB = [math]::Round($arc.Length / 1MB, 1)
        Write-Host "  [$sizeMB MB] $($arc.Name)" -ForegroundColor White

        # 7-Zip 압축 해제 — 한글 파일명 안전, -y는 모든 prompt에 yes
        & $sevenZip x $arc.FullName "-o$target" -y -bso0 -bsp1 | Out-Null

        if ($LASTEXITCODE -eq 0) {
            Remove-Item -LiteralPath $arc.FullName -Force
        } else {
            Write-Host "    ⚠️  실패 (exit $LASTEXITCODE) — 원본 zip 유지" -ForegroundColor Red
        }
    }

    Write-Host ""
} while ($passNum -lt 10)

# 결과 요약
Write-Host "`n=== 결과 요약 (이미지 폴더별 개수) ===" -ForegroundColor Cyan
$summary = Get-ChildItem -Path $root -Directory -Recurse | ForEach-Object {
    $imgCount = (Get-ChildItem -LiteralPath $_.FullName -Include *.jpg,*.jpeg,*.png -File -ErrorAction SilentlyContinue).Count
    if ($imgCount -gt 0) {
        [PSCustomObject]@{ Folder = $_.Name; Count = $imgCount }
    }
} | Sort-Object Count -Descending

$summary | Format-Table -AutoSize | Out-Host
$total = ($summary | Measure-Object -Property Count -Sum).Sum
Write-Host "총 이미지: $total 장 / 폴더(클래스) 수: $($summary.Count)" -ForegroundColor Green

Write-Host "`n✅ 압축 해제 완료. 다음: prepare_data_vision.py" -ForegroundColor Green
