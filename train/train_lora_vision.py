"""
gemma4:12b QLoRA 비전 학습 스크립트.

⚠️ 주의: gemma4는 2026-06-03 출시. Unsloth/transformers의 비전 지원은
출시 초기라 불안정할 수 있음. 실행 전 의존성 호환 확인:
  - transformers >= 4.50 (gemma4 비전 지원 시점)
  - peft >= 0.13
  - 또는 Unsloth multimodal (gemma4 추가 시점부터)

VRAM (RTX 5080 16GB) 기준 설정:
  - 4bit 양자화 + LoRA rank 16
  - batch_size=1, gradient_accumulation=4 (effective batch 4)
  - 이미지 해상도 448x448 또는 모델 기본

학습 시간 가이드 (3만 장, 1 epoch):
  - 5080: 약 45~90분
"""
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "train" / "data_vision"
OUTPUT_DIR = ROOT / "train" / "lora_out_vision"

# gemma4 정확한 모델 ID는 Hugging Face에서 확인 후 교체.
# 출시 직후 후보 이름 (실제 ID는 확인 필요):
#   "google/gemma-4-12B-it"  ← instruct 버전
#   "google/gemma-4-12b"     ← base
MODEL_ID = os.environ.get("BASE_MODEL", "google/gemma-4-12B-it")
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "448"))


# ============================================================
# Dataset
# ============================================================
class VisionSFTDataset(Dataset):
    """jsonl 파일을 읽어 (image, prompt, answer) 3종으로 반환."""

    def __init__(self, jsonl_path: Path, processor):
        self.records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self.records.append(rec)
        self.processor = processor
        self.root = ROOT

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        # 절대경로면 그대로, 상대경로면 ROOT 기준으로 해석
        raw = rec["image_path"]
        img_path = Path(raw) if Path(raw).is_absolute() else self.root / raw
        # 이미지 로드 + 리사이즈 (메모리 절약)
        img = Image.open(img_path).convert("RGB")
        if max(img.size) > MAX_IMAGE_SIZE:
            img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.LANCZOS)

        # gemma4 chat template — multimodal messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": rec["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rec["answer"]}],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=text,
            images=img,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        )
        # batch dim 제거
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        # 패딩 토큰 위치는 loss에서 제외
        if "attention_mask" in inputs:
            inputs["labels"][inputs["attention_mask"] == 0] = -100
        return inputs


# ============================================================
# 학습 메인
# ============================================================
def main():
    print(f"[1/5] 모델 로드: {MODEL_ID}")
    print(f"      MAX_IMAGE_SIZE={MAX_IMAGE_SIZE}, output={OUTPUT_DIR}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    print("[2/5] LoRA 어댑터 부착...")
    # 비전 통합 모델은 target_modules가 모델마다 다름. 일반적인 attention/MLP 우선.
    # 실행 후 오류 나면 model.named_modules() 출력해서 정확한 이름 확인.
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[3/5] 데이터셋 로드...")
    train_ds = VisionSFTDataset(DATA_DIR / "train.jsonl", processor)
    val_ds = VisionSFTDataset(DATA_DIR / "val.jsonl", processor)
    print(f"      train={len(train_ds)}, val={len(val_ds)}")

    print("[4/5] 학습 시작...")
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    print("\n[5/5] LoRA 어댑터 저장...")
    final_dir = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))

    print(f"\n✅ 학습 완료: {final_dir}")
    print("다음 단계: python train\\eval_lora_vision.py")


if __name__ == "__main__":
    main()
