"""
LoRA 어댑터를 베이스 모델에 병합 → GGUF Q4_K_M으로 변환.
출력: train/gguf/qwen-lora-q4_k_m.gguf (약 4.5GB)
"""
from pathlib import Path
from unsloth import FastLanguageModel

ROOT = Path(__file__).parent.parent
LORA_DIR = ROOT / "train" / "lora_out" / "final"
GGUF_DIR = ROOT / "train" / "gguf"
GGUF_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048

print("[1/2] 학습된 LoRA 어댑터 로드...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(LORA_DIR),
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

print("[2/2] GGUF Q4_K_M 변환 (10~20분 소요)...")
# Unsloth가 내부적으로 llama.cpp의 convert_hf_to_gguf.py + quantize 실행
model.save_pretrained_gguf(
    str(GGUF_DIR),
    tokenizer,
    quantization_method="q4_k_m",
)

# 생성된 .gguf 파일 찾기
gguf_files = list(GGUF_DIR.glob("*.gguf"))
if gguf_files:
    print(f"\n✅ GGUF 생성 완료:")
    for f in gguf_files:
        size_gb = f.stat().st_size / 1e9
        print(f"   {f.name} ({size_gb:.2f} GB)")
    print(f"\n다음 단계: .\\train\\deploy.ps1")
else:
    print("⚠️  GGUF 파일이 생성되지 않았습니다. 로그를 확인하세요.")
