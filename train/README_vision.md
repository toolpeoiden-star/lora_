# 🍱 calorie-ai 비전 LoRA 학습 가이드

`train_lora.py` (Qwen2.5 텍스트)와 별개로 **gemma4:12b 비전 모델**을
한국 음식 이미지 인식용으로 파인튜닝하는 파이프라인.

## ⚠️ 호환성 주의

- **gemma4는 2026-06-03 출시**. Unsloth/transformers의 비전 지원이 출시 초기라 불안정 가능.
- 정확한 Hugging Face 모델 ID 확인 필요 (`google/gemma-4-12b-it` 추정).
- `train_lora_vision.py` 의 `target_modules` 가 모델 실제 구조와 안 맞으면
  `model.print_trainable_parameters()` 결과 확인 후 조정.

## 디렉토리 구조

```
lora_mcp/
├── dataset/
│   ├── aihub/                    # AI Hub 한국 음식 (aihubshell로 받음)
│   │   ├── Training/
│   │   │   ├── 김치찌개/*.jpg
│   │   │   ├── 비빔밥/*.jpg
│   │   │   └── ...
│   │   └── Validation/...        # AI Hub 자체 검증셋 (학습엔 안 씀)
│   ├── calorie_ai/               # 앱 운영 누적 데이터 (출시 후)
│   │   ├── training/2026-06.jsonl
│   │   └── uploads/m_xxx.jpg
│   └── eval/                     # 우리 검증셋 (직접 라벨링, 200장+)
│       └── manual_eval.jsonl
├── train/
│   ├── prepare_data_vision.py    # 데이터 수집 → train/val jsonl
│   ├── train_lora_vision.py      # QLoRA 학습
│   ├── eval_lora_vision.py       # 검증셋 정확도 측정
│   ├── setup_vision.ps1          # 비전 의존성 추가 설치
│   └── data_vision/
│       ├── train.jsonl
│       └── val.jsonl
└── train/lora_out_vision/final/  # 학습 결과 LoRA adapter
```

## 사용 흐름

### 0. 최초 1회 (setup_windows.ps1 실행 이후)

```powershell
.\venv\Scripts\Activate.ps1
.\train\setup_vision.ps1
```

### 1. 검증셋 미리 만들기 (학습 시작 전 필수!)

본인 식사 사진 200~300장을 직접 라벨링.
`dataset/eval/manual_eval.jsonl` 에 다음 형식:

```json
{"image_path":"dataset/eval/images/img001.jpg","answer":{"items":[{"dish_name_ko":"김치찌개","grams_typical":350,"confidence":"high"}],"not_food":false}}
{"image_path":"dataset/eval/images/img002.jpg","answer":{"items":[{"dish_name_ko":"비빔밥","grams_typical":400,"confidence":"high"}],"not_food":false}}
```

이 데이터는 **절대 학습에 안 들어감**. 베이스 모델 vs LoRA 후 정확도 비교용.

### 2. AI Hub 데이터 받기 (aihubshell)

```powershell
# 1) 데이터셋 구조 확인
aihubshell -mode l -datasetkey {음식_데이터셋_번호}

# 2) 받기 (특정 카테고리만 / 전체)
aihubshell -mode d -datasetkey {번호} -aihubapikey '키' -filekey 1,2,3
```

`dataset/aihub/` 에 풀려야 함.

### 3. 학습 데이터 준비

```powershell
python train\prepare_data_vision.py
```

`train/data_vision/train.jsonl`, `val.jsonl` 생성.

### 4. 베이스라인 측정 (학습 전)

```powershell
python train\eval_lora_vision.py --base
```

검증셋 정확도 = "X%" 기록. LoRA 후와 비교용.

### 5. 학습 실행

```powershell
$env:BASE_MODEL='google/gemma-4-12b-it'  # 실제 ID 확인 후 사용
python train\train_lora_vision.py
```

소요 시간 (5080, 데이터 3만 장 기준): 약 45~90분.

### 6. 학습 후 정확도 측정

```powershell
python train\eval_lora_vision.py
```

베이스라인 대비 향상 확인. 만족스러우면 다음 단계.

### 7. GGUF 변환 + Ollama 등록

```powershell
python train\export_gguf.py
.\train\deploy.ps1
```

(텍스트 LoRA용 export_gguf.py 와 동일하지만 비전 모델 변환은 llama.cpp가 지원하는지 확인 필요)

### 8. calorie-ai 앱 재설정

서버 PC의 `calorie-ai/.env.local` 업데이트:

```
OLLAMA_VISION_MODEL=gemma4-food:v1
```

dev 서버 재시작.

## 트러블슈팅

### `target_modules` 오류
모델 구조가 예상과 다른 경우. 실행 후 다음 코드로 실제 모듈명 확인:

```python
from transformers import AutoModelForImageTextToText
m = AutoModelForImageTextToText.from_pretrained("google/gemma-4-12b-it")
for n, _ in m.named_modules():
    if "proj" in n or "linear" in n:
        print(n)
```

출력에서 자주 등장하는 끝 이름(e.g., `q_proj`, `k_proj`)을 `target_modules`에 사용.

### OOM (Out of Memory)
- `MAX_IMAGE_SIZE=336` (또는 224)으로 축소
- `gradient_accumulation_steps=8` 로 증가
- `max_length=512` 로 축소

### 학습이 너무 느림 (한 step에 분 단위)
- `dataloader_num_workers=0` 로 설정 (Windows에서 worker 이슈)
- 이미지 미리 리사이즈 → 디스크 캐시 (별도 스크립트)

### gemma4 비전 지원이 transformers에 없음
- transformers 최신 버전 확인: `pip install --upgrade transformers`
- 안 되면 gemma3 비전으로 임시 학습 후 gemma4 지원 추가될 때 재학습
- 또는 LLaMA-Factory, Axolotl 등 별도 도구 검토

## 다음 단계

1. 첫 학습 후 정확도 만족 → calorie-ai 모델 교체
2. 부족하면 AI Hub 데이터 추가 다운로드 → 재학습 (누적)
3. 사용자 데이터 누적 → 정기 재학습
