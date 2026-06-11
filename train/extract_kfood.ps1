# ============================================================
# AI Hub 한국 음식 이미지 데이터셋 일괄 압축 해제
# 중첩 zip(zip 안에 또 zip) 자동 처리. 원본 zip 삭제.
# ============================================================
$ErrorActionPreference = "Continue"
$root = "C:\Users\user\Desktop\한국 음식 이미지\kfood"

if (-not (Test-Path $root)) {
    Write-Host "❌ 경로 없음: $root" -ForegroundColor Red
    exit 1
}

Set-Location $root
Write-Host "📂 작업 경로: $root`n" -ForegroundColor Cyan

$passNum = 0
do {
    $passNum++
    $zips = Get-ChildItem -Path $root -Filter "*.zip" -Recurse -File
    if (-not $zips) {
        Write-Host "✅ 더 이상 압축 파일 없음. 완료." -ForegroundColor Green
        break
    }

    Write-Host "── Pass $passNum : $($zips.Count)개 압축 파일 발견 ──" -ForegroundColor Yellow

    foreach ($zip in $zips) {
        $target = $zip.DirectoryName
        $sizeMB = [math]::Round($zip.Length / 1MB, 1)
        Write-Host "  [$sizeMB MB] $($zip.Name)" -ForegroundColor White

        try {
            Expand-Archive -LiteralPath $zip.FullName -DestinationPath $target -Force -ErrorAction Stop
            Remove-Item -LiteralPath $zip.FullName -Force
        } catch {
            Write-Host "    ⚠️  실패: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "    (한글 파일명/Windows 표준 zip 미지원 가능 — 7-Zip 권장)" -ForegroundColor DarkYellow
        }
    }

    Write-Host ""
} while ($passNum -lt 10)  # 무한루프 방지 (10단계 중첩까지)

# 최종 확인: 폴더별 이미지 수
Write-Host "`n=== 결과 요약 (이미지 폴더별 개수) ===" -ForegroundColor Cyan
Get-ChildItem -Path $root -Directory -Recurse | ForEach-Object {
    $imgCount = (Get-ChildItem -LiteralPath $_.FullName -Include *.jpg,*.jpeg,*.png -File -ErrorAction SilentlyContinue).Count
    if ($imgCount -gt 0) {
        "{0,-50} {1,6}장" -f $_.Name, $imgCount
    }
} | Sort-Object | Select-Object -First 50

Write-Host "`n(상위 50개만 표시. 전체 보려면 위 코드에서 Select-Object 제거)" -ForegroundColor DarkGray
Write-Host "`n✅ 압축 해제 완료. 다음: prepare_data_vision.py" -ForegroundColor Green
