"""
학습한 LoRA의 정확도 측정 스크립트.

검증셋 (학습에 안 들어간 사진):
  - dataset/eval/*.jsonl
  - 각 줄: {"image_path": "...", "answer": {"items":[{"dish_name_ko":"..."}]}}

평가 메트릭:
  - JSON 파싱율 (응답이 valid JSON인 비율)
  - dish_name 정답 일치율 (top-1)
  - dish_name 포함율 (부분 일치)
  - 평균 응답 시간

베이스 모델 vs LoRA 후 비교:
  python eval_lora_vision.py --base   # 순정 gemma4
  python eval_lora_vision.py          # LoRA 적용 후 (기본값)
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).parent.parent
EVAL_DIR = ROOT / "dataset" / "eval"
LORA_DIR = ROOT / "train" / "lora_out_vision" / "final"
MODEL_ID = "google/gemma-4-12b-it"


def extract_dish_names(text: str) -> list:
    """모델 응답 텍스트에서 dish_name_ko 추출. JSON 깨져도 정규식 fallback."""
    try:
        # JSON 파싱 시도
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            obj = json.loads(m.group(0))
            items = obj.get("items", [])
            return [it.get("dish_name_ko", "").strip() for it in items if it.get("dish_name_ko")]
    except json.JSONDecodeError:
        pass
    # JSON 깨진 경우 정규식으로 dish_name_ko 추출
    return re.findall(r'"dish_name_ko"\s*:\s*"([^"]+)"', text)


def normalize_name(s: str) -> str:
    """음식 이름 비교용 정규화 — 공백·괄호 제거."""
    return re.sub(r"[\s()（）·,，]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="LoRA 없이 베이스 모델 평가 (비교용)")
    ap.add_argument("--limit", type=int, default=0, help="평가 샘플 수 제한 (디버깅용)")
    args = ap.parse_args()

    # 평가 데이터 로드
    records = []
    for jsonl in sorted(EVAL_DIR.glob("*.jsonl")):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"❌ 검증셋 없음: {EVAL_DIR}/*.jsonl")
    print(f"검증셋 {len(records)}개 로드\n")

    # 모델 로드 (학습과 동일하게 4bit)
    print(f"모델 로드: {MODEL_ID}{' + LoRA' if not args.base else ' (baseline)'}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    if not args.base:
        if not LORA_DIR.exists():
            raise SystemExit(f"❌ LoRA 폴더 없음: {LORA_DIR}")
        model = PeftModel.from_pretrained(model, str(LORA_DIR))
    model.eval()

    # 평가 실행
    stats = {
        "total": 0,
        "json_ok": 0,
        "exact_match": 0,    # 정답 dish_name 1개 이상 정확 일치
        "partial_match": 0,  # 정답에 정답이 포함됨 ("파김치" vs "대파김치")
        "wall_ms_sum": 0,
    }
    samples_log = []

    for rec in records:
        raw = rec["image_path"]
        img_p = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        img = Image.open(img_p).convert("RGB")
        if max(img.size) > 448:
            img.thumbnail((448, 448), Image.LANCZOS)
        # 정답 파싱
        gold = rec.get("answer", {})
        if isinstance(gold, str):
            try:
                gold = json.loads(gold)
            except Exception:
                gold = {}
        gold_names = [normalize_name(it.get("dish_name_ko", ""))
                      for it in gold.get("items", [])]
        gold_names = [g for g in gold_names if g]

        # 추론
        from prepare_data_vision import PROMPT_TEMPLATE
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT_TEMPLATE},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0)
        wall_ms = int((time.time() - t0) * 1000)
        gen_text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        pred_names = [normalize_name(n) for n in extract_dish_names(gen_text)]
        pred_names = [p for p in pred_names if p]

        # 메트릭 업데이트
        stats["total"] += 1
        stats["wall_ms_sum"] += wall_ms
        json_ok = bool(extract_dish_names(gen_text))
        if json_ok:
            stats["json_ok"] += 1
        # exact = 정답 dish 중 하나라도 예측 dish와 정확히 일치
        if any(g in pred_names for g in gold_names):
            stats["exact_match"] += 1
        # partial = 정답 dish 중 하나라도 예측 dish의 substring이거나 vice versa
        elif any(any(g in p or p in g for p in pred_names) for g in gold_names):
            stats["partial_match"] += 1

        samples_log.append({
            "image": rec["image_path"],
            "gold": gold_names,
            "pred": pred_names,
            "raw": gen_text[:200],
            "wall_ms": wall_ms,
        })

    # 리포트
    n = stats["total"]
    print(f"\n=== 평가 결과 ({n}건) ===")
    print(f"JSON 파싱 성공률 : {stats['json_ok']}/{n} = {stats['json_ok']/n:.1%}")
    print(f"정확 일치율      : {stats['exact_match']}/{n} = {stats['exact_match']/n:.1%}")
    print(f"부분 일치율      : {stats['partial_match']}/{n} = {stats['partial_match']/n:.1%}")
    print(f"종합 일치율      : {(stats['exact_match']+stats['partial_match'])}/{n} = {(stats['exact_match']+stats['partial_match'])/n:.1%}")
    print(f"평균 추론 시간   : {stats['wall_ms_sum']/n:.0f}ms")

    # 상세 로그 저장
    log_path = ROOT / "train" / f"eval_{'base' if args.base else 'lora'}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "samples": samples_log}, f, ensure_ascii=False, indent=2)
    print(f"\n상세 로그: {log_path}")


if __name__ == "__main__":
    main()
