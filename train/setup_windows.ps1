# ============================================================
# 5080 PC 최초 1회 세팅 스크립트
# RTX 5080(Blackwell, sm_120)은 CUDA 12.8 + PyTorch nightly 필요
# 소요시간: 30분 ~ 1시간 (네트워크 속도에 따라)
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "[1/6] Python 가상환경 생성..." -ForegroundColor Cyan
$venvPath = "$PSScriptRoot\..\venv"
if (-not (Test-Path $venvPath)) {
    python -m venv $venvPath
}
& "$venvPath\Scripts\Activate.ps1"

Write-Host "[2/6] pip 업그레이드..." -ForegroundColor Cyan
python -m pip install --upgrade pip wheel setuptools

Write-Host "[3/6] PyTorch nightly (CUDA 12.8) 설치 — Blackwell GPU 필수..." -ForegroundColor Cyan
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

Write-Host "[4/6] Unsloth + 학습 의존성 설치..." -ForegroundColor Cyan
pip install "unsloth[cu128-torch-nightly] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets accelerate peft trl bitsandbytes
pip install sentencepiece protobuf

Write-Host "[5/6] llama.cpp 빌드 (GGUF 변환용)..." -ForegroundColor Cyan
$llamaPath = "$PSScriptRoot\..\llama.cpp"
if (-not (Test-Path $llamaPath)) {
    git clone https://github.com/ggerganov/llama.cpp.git $llamaPath
}
Set-Location $llamaPath
pip install -r requirements.txt
Set-Location $PSScriptRoot

Write-Host "[6/6] GPU 인식 확인..." -ForegroundColor Cyan
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB'); print(f'Compute capability: {torch.cuda.get_device_capability(0)}')"

Write-Host "`n✅ 세팅 완료. 다음 단계: python train\prepare_data.py" -ForegroundColor Green
