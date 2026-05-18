# 5080 PC LoRA 학습/배포 가이드

## 사전 요구사항
- Windows 11
- RTX 5080 (드라이버 최신)
- Python 3.10 ~ 3.11
- Ollama 설치됨
- (Open WebUI용) Docker Desktop

## 실행 순서

### 최초 1회
```powershell
git clone https://github.com/tlstjdgh21/loRA_mcp.git
cd loRA_mcp
.\train\setup_windows.ps1
```
PyTorch nightly(CUDA 12.8) + Unsloth + llama.cpp 설치. 30분~1시간.

마지막에 출력되는 GPU 정보에서 `CUDA available: True`, `GPU: NVIDIA GeForce RTX 5080`, `Compute capability: (12, 0)` 확인.

### 학습 사이클 (데이터 추가될 때마다)
```powershell
.\venv\Scripts\Activate.ps1
git pull                          # 최신 dataset 받기
python train\prepare_data.py      # 정제 + 분할
python train\train_lora.py        # 학습 (100건 기준 10~15분)
python train\export_gguf.py       # GGUF 변환 (10~20분)
.\train\deploy.ps1                # Ollama 등록 + LAN 공유
```

### 임직원 접속
- **Open WebUI**: `http://{5080_PC_IP}:3000`
- **Ollama API 직접**: `http://{5080_PC_IP}:11434`

5080 PC IP 확인: `ipconfig` → IPv4 주소

## 트러블슈팅

### "CUDA error: no kernel image available"
5080은 Blackwell(sm_120)이라 일반 PyTorch가 못 잡습니다. `setup_windows.ps1`이 nightly + cu128을 설치하는지 확인.

### bitsandbytes가 5080을 못 잡음
2026년 1월 기준 bitsandbytes Blackwell 지원은 v0.45+ 필요. 안 되면 8bit optimizer 대신 `optim="adamw_torch"`로 train_lora.py 수정.

### OOM (Out of Memory)
- `per_device_train_batch_size=1` + `gradient_accumulation_steps=8`로 조정
- `max_seq_length`를 1024로 축소

### Ollama가 LAN에서 안 보임
1. 시스템 트레이 Ollama → Quit → 다시 실행 (환경변수 반영)
2. 방화벽 11434/3000 인바운드 확인
3. 공유기 격리 설정 확인

## 데이터셋 적재 워크플로
1. 이 PC(또는 직원 PC)에서 Claude Code 사용 중 좋은 대화가 나오면 `/학습 [태그]` 입력
2. 자동으로 `dataset/`에 JSONL 추가 + GitHub 푸시
3. 5080 PC에서 주기적으로 `git pull` → 위 학습 사이클 재실행
