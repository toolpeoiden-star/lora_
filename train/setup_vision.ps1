# ============================================================
# 비전 학습 전용 의존성 추가 설치 스크립트
# setup_windows.ps1 실행 후 한 번만 추가로 실행
# ============================================================
$ErrorActionPreference = "Stop"

Write-Host "[1/3] 가상환경 활성화..." -ForegroundColor Cyan
$venvPath = "$PSScriptRoot\..\venv"
& "$venvPath\Scripts\Activate.ps1"

Write-Host "[2/3] 비전 LoRA 추가 패키지 설치..." -ForegroundColor Cyan
# gemma4 비전 지원에 필요한 최신 transformers + 이미지 처리
pip install --upgrade "transformers>=4.50"
pip install --upgrade "peft>=0.13"
pip install pillow torchvision

Write-Host "[3/3] gemma4 모델 ID 확인..." -ForegroundColor Cyan
Write-Host @"

⚠️ 중요: 환경변수 BASE_MODEL로 실제 Hugging Face ID 지정 필요.
gemma4 12b instruct 모델은 출시 직후라 정확한 ID 확인 후 사용:

  https://huggingface.co/google 에서 검색 → 정확한 ID 복사

예시 (실제 ID는 다를 수 있음):
  `$env:BASE_MODEL='google/gemma-4-12B-it'
  python train\train_lora_vision.py

또는 train_lora_vision.py 의 MODEL_ID 상수를 직접 수정.

"@ -ForegroundColor Yellow

Write-Host "✅ 비전 학습 준비 완료" -ForegroundColor Green
