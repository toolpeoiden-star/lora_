# ============================================================
# Ollama에 LoRA 학습 모델 등록 + LAN 노출 + Open WebUI 기동
# ============================================================
$ErrorActionPreference = "Stop"

$MODEL_NAME = "company-qwen"
$GGUF_DIR = "$PSScriptRoot\gguf"
$MODELFILE = "$PSScriptRoot\Modelfile"
$MODELFILE_TMP = "$PSScriptRoot\Modelfile.tmp"

# --- 1. safetensors 폴더 확인 (Ollama가 자체 변환) ---
$safetensors = Get-ChildItem $GGUF_DIR -Filter "*.safetensors" -ErrorAction SilentlyContinue
if (-not $safetensors) {
    Write-Host "❌ safetensors 파일이 없습니다. 먼저 export_gguf.py 실행." -ForegroundColor Red
    exit 1
}
Write-Host "✓ safetensors 발견: $($safetensors.Count)개 파일" -ForegroundColor Green

# --- 2. Modelfile 그대로 사용 (FROM ./gguf) ---
Copy-Item $MODELFILE $MODELFILE_TMP -Force

# --- 3. Ollama 등록 ---
Write-Host "`n[1/3] Ollama 모델 등록..." -ForegroundColor Cyan
Set-Location $PSScriptRoot
ollama create $MODEL_NAME -f $MODELFILE_TMP
Remove-Item $MODELFILE_TMP

# --- 4. Ollama를 LAN에 노출 ---
Write-Host "`n[2/3] Ollama를 0.0.0.0:11434로 노출 (사내망 공유)..." -ForegroundColor Cyan
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "0.0.0.0:11434", "User")
[System.Environment]::SetEnvironmentVariable("OLLAMA_ORIGINS", "*", "User")
Write-Host "  → Ollama 재시작 필요: 시스템 트레이 Ollama 아이콘 → Quit → 다시 실행" -ForegroundColor Yellow

# 방화벽 인바운드 허용 (관리자 권한 필요)
Write-Host "`n  방화벽 규칙 추가 시도 (관리자 권한 필요)..."
try {
    New-NetFirewallRule -DisplayName "Ollama 11434" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow -ErrorAction Stop | Out-Null
    Write-Host "  ✓ 방화벽 11434 인바운드 허용" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  방화벽 규칙 추가 실패 — 관리자 권한 PowerShell에서 수동 실행:" -ForegroundColor Yellow
    Write-Host '     New-NetFirewallRule -DisplayName "Ollama 11434" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow' -ForegroundColor Gray
}

# --- 5. Open WebUI (Docker) ---
Write-Host "`n[3/3] Open WebUI 기동 (Docker)..." -ForegroundColor Cyan
$dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerInstalled) {
    Write-Host "  ⚠️  Docker Desktop이 설치되어 있지 않습니다." -ForegroundColor Yellow
    Write-Host "     https://www.docker.com/products/docker-desktop/ 에서 설치 후 재실행." -ForegroundColor Gray
} else {
    docker rm -f open-webui 2>$null | Out-Null
    docker run -d `
        --name open-webui `
        -p 3000:8080 `
        -e OLLAMA_BASE_URL=http://host.docker.internal:11434 `
        -v open-webui:/app/backend/data `
        --restart always `
        ghcr.io/open-webui/open-webui:main

    try {
        New-NetFirewallRule -DisplayName "Open WebUI 3000" -Direction Inbound -Protocol TCP -LocalPort 3000 -Action Allow -ErrorAction Stop | Out-Null
    } catch {}
}

# --- 안내 ---
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.InterfaceAlias -notmatch "Loopback|vEthernet"} | Select-Object -First 1).IPAddress
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "✅ 배포 완료" -ForegroundColor Green
Write-Host "============================================================"
Write-Host "모델 이름      : $MODEL_NAME"
Write-Host "Ollama API     : http://${ip}:11434"
Write-Host "Open WebUI     : http://${ip}:3000  ← 임직원에게 공유"
Write-Host "테스트         : ollama run $MODEL_NAME"
Write-Host "`n⚠️  Ollama 트레이 아이콘에서 Quit 후 재실행해야 LAN 노출이 적용됩니다." -ForegroundColor Yellow
