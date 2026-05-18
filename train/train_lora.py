"""
Qwen2.5-7B-Instruct QLoRA 학습 (Unsloth 사용).
RTX 5080(16GB) 기준 batch_size=2, grad_accum=4로 약 8~12GB VRAM 사용.

학습 시간 가이드(데이터 100건, 3 epoch):
  - 5080: 약 10~15분
  - 4090: 약 7분
"""
import json
from pathlib import Path
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "train" / "data"
OUTPUT_DIR = ROOT / "train" / "lora_out"

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN = 2048

# --- 1. 모델 로드 (4bit 양자화) ---
print("[1/4] 모델 로드 중...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,           # 자동 (bf16 on Blackwell)
    load_in_4bit=True,
)

# --- 2. LoRA 어댑터 부착 ---
print("[2/4] LoRA 어댑터 설정...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# --- 3. 데이터셋 로드 + 포매팅 ---
print("[3/4] 데이터 로드...")
def formatting(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

ds = load_dataset(
    "json",
    data_files={
        "train": str(DATA_DIR / "train.jsonl"),
        "val":   str(DATA_DIR / "val.jsonl"),
    },
)
ds = ds.map(formatting)
print(f"  train: {len(ds['train'])}개, val: {len(ds['val'])}개")

# --- 4. 학습 ---
print("[4/4] 학습 시작...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    args=TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=42,
        report_to="none",
    ),
)

trainer.train()

# --- 5. LoRA 어댑터 저장 ---
print("\n저장 중...")
model.save_pretrained(str(OUTPUT_DIR / "final"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

print(f"\n✅ 학습 완료: {OUTPUT_DIR / 'final'}")
print("다음 단계: python train\\export_gguf.py")
