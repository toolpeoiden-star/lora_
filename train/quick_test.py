"""
빠른 추론 테스트 — 사진 1장 또는 폴더의 모든 사진에 대해 LoRA 모델 추론.

사용:
  python train\quick_test.py 사진경로.jpg            # 단일 사진
  python train\quick_test.py 폴더경로\               # 폴더 안 모든 사진
  python train\quick_test.py --base 사진경로.jpg     # LoRA 없이 베이스 모델만 (비교용)

출력:
  사진별로 "AI 답" + "추론 시간(ms)" 표시.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).parent.parent
LORA_DIR = ROOT / "train" / "lora_out_vision" / "final"
MODEL_ID = "google/gemma-4-12B-it"

PROMPT = """당신은 한국 음식 영양 분석 전문가입니다. 사진 속 음식을 분석하세요.

규칙:
- 사진에서 보이는 음식만 분석. 식기·수저·젓가락·접시·그릇·테이블 등은 절대 items에 넣지 마라.
- 사진 전체가 완성된 요리(피자/햄버거/김밥/비빔밥/라면/떡볶이 등)면 요리 이름 1개로 답.
- 단일 식재료(토마토/치즈/밥/김/오이/대파/무)만 답하지 말고 요리명으로 답해라.
- 도시락·정식은 칸별로 분리 (밥/고기/반찬/국).
- 각 항목: dish_name_ko, cooking_method, grams_typical, confidence(high/medium/low).
- 음식이 전혀 아니면 not_food=true, items 빈 배열.

JSON으로만 답하라:
{"items":[{"dish_name_ko":"...","cooking_method":"...","grams_typical":...,"confidence":"..."}],"not_food":false}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="사진 파일 또는 폴더 경로")
    ap.add_argument("--base", action="store_true", help="LoRA 없이 베이스 모델만")
    ap.add_argument("--limit", type=int, default=10, help="폴더일 때 처리할 최대 사진 수")
    args = ap.parse_args()

    # 사진 목록 수집
    inp = Path(args.input)
    if inp.is_dir():
        imgs = sorted([p for p in inp.rglob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])[:args.limit]
    elif inp.is_file():
        imgs = [inp]
    else:
        print(f"❌ 파일/폴더 없음: {inp}")
        sys.exit(1)

    if not imgs:
        print("❌ 이미지 없음")
        sys.exit(1)
    print(f"테스트할 사진: {len(imgs)}장\n")

    # 모델 로드 (학습과 동일한 양자화 설정)
    print(f"모델 로드: {MODEL_ID}{' + LoRA' if not args.base else ' (베이스만)'}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=[
            "embed_vision", "vision_tower",
            "patch_dense", "patch_ln1", "patch_ln2",
            "multi_modal_projector", "vision_projector",
            "lm_head", "embed_tokens",
        ],
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    if not args.base:
        if not LORA_DIR.exists():
            print(f"❌ LoRA 폴더 없음: {LORA_DIR}")
            sys.exit(1)
        model = PeftModel.from_pretrained(model, str(LORA_DIR))
    model.eval()

    print("=" * 60)

    for img_path in imgs:
        print(f"\n📷 {img_path.name}")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((448, 448), Image.LANCZOS)

        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)
        if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.uint8:
            inputs["pixel_values"] = (inputs["pixel_values"].float() / 255.0).to(torch.bfloat16)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
            )
        wall_ms = int((time.time() - t0) * 1000)
        gen_text = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        print(f"   응답 ({wall_ms}ms):")
        # JSON 파싱 시도
        try:
            m = re.search(r"\{[\s\S]*\}", gen_text)
            obj = json.loads(m.group(0)) if m else None
            if obj:
                items = obj.get("items", [])
                for it in items:
                    name = it.get("dish_name_ko", "?")
                    g = it.get("grams_typical", "?")
                    cm = it.get("cooking_method", "")
                    c = it.get("confidence", "")
                    print(f"     - {name} | {g}g | {cm} | {c}")
                if obj.get("not_food"):
                    print("     (not_food=true)")
            else:
                print(f"     JSON 못 찾음: {gen_text[:200]}")
        except Exception as e:
            print(f"     JSON 파싱 실패 ({e}): {gen_text[:200]}")


if __name__ == "__main__":
    main()
