"""
비전 LoRA 학습용 데이터 준비 스크립트.

입력 소스 2종:
  1) AI Hub 한국 음식 데이터셋
     - dataset/aihub/image/{class_name}/*.jpg (또는 .png)
     - dataset/aihub/label/{class_name}/*.json  (선택)
     클래스명이 폴더명 = 정답 음식명.

  2) calorie-ai 사용자 데이터
     - dataset/calorie_ai/training/*.jsonl  (앱이 적재한 학습 로그)
     - dataset/calorie_ai/uploads/*.jpg     (사진 파일들)
     event=confirm 줄에서 userFinal을 정답으로 사용.

출력:
  - train/data_vision/train.jsonl
  - train/data_vision/val.jsonl
각 줄: {"image_path": "...", "prompt": "...", "answer": "..."}

검증셋(우리가 직접 라벨링한 것)은 학습에서 분리:
  - dataset/eval/*.jsonl  → 학습에 안 들어감, eval_lora_vision.py만 씀
"""

import json
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "train" / "data_vision"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 데이터셋 위치를 환경변수로 덮어쓰기 가능 (대용량 데이터를 별도 경로에 둘 때)
#   $env:AIHUB_DIR='C:\Users\user\Desktop\한국 음식 이미지\kfood'
#   $env:CALORIE_AI_DIR='C:\path\to\calorie_ai'
AIHUB_DIR = Path(os.environ.get("AIHUB_DIR", str(DATA_DIR / "aihub")))
CALORIE_AI_DIR = Path(os.environ.get("CALORIE_AI_DIR", str(DATA_DIR / "calorie_ai")))

# 학습용 프롬프트 — calorie-ai 앱과 동일하게 유지 (테스트 환경 일치)
PROMPT_TEMPLATE = """당신은 한국 음식 영양 분석 전문가입니다. 사진 속 음식을 분석하세요.

규칙:
- 사진에서 보이는 음식만 분석. 식기·수저·젓가락·접시·그릇·테이블 등은 절대 items에 넣지 마라.
- 사진 전체가 완성된 요리(피자/햄버거/김밥/비빔밥/라면/떡볶이 등)면 요리 이름 1개로 답.
- 단일 식재료(토마토/치즈/밥/김/오이/대파/무)만 답하지 말고 요리명으로 답해라.
- 도시락·정식은 칸별로 분리 (밥/고기/반찬/국).
- 각 항목: dish_name_ko, cooking_method, grams_typical, confidence(high/medium/low).
- 음식이 전혀 아니면 not_food=true, items 빈 배열.

JSON으로만 답하라:
{"items":[{"dish_name_ko":"...","cooking_method":"...","grams_typical":...,"confidence":"..."}],"not_food":false}"""


def make_answer_from_class(class_name: str, default_grams: int = 150) -> str:
    """AI Hub 단일 클래스 라벨 → 학습용 정답 JSON."""
    return json.dumps(
        {
            "items": [
                {
                    "dish_name_ko": class_name,
                    "cooking_method": "",
                    "grams_typical": default_grams,
                    "confidence": "high",
                }
            ],
            "not_food": False,
        },
        ensure_ascii=False,
    )


def make_answer_from_user_final(user_final: list) -> str:
    """사용자가 정정한 최종 결과 → 학습용 정답 JSON."""
    items = []
    for f in user_final:
        items.append(
            {
                "dish_name_ko": f.get("name", ""),
                "cooking_method": "",
                "grams_typical": int(f.get("grams", 150)),
                "confidence": "high",
            }
        )
    return json.dumps({"items": items, "not_food": False}, ensure_ascii=False)


def collect_aihub() -> list:
    """AI Hub 폴더에서 이미지 재귀 탐색. 부모 폴더명 = 클래스명.
    실제 구조 예: kfood/구이/구이/갈비구이/*.jpg → 클래스 '갈비구이'."""
    records = []
    aihub_root = AIHUB_DIR
    if not aihub_root.exists():
        print(f"  [skip] AI Hub 폴더 없음: {aihub_root}")
        return records
    print(f"  [aihub] 탐색 시작: {aihub_root}")

    # 폴더 구조가 데이터셋마다 다를 수 있어 광범위 탐색
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for img_path in aihub_root.rglob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue
        # 부모 폴더명을 클래스명으로 추정 (보통 폴더명 = 음식 클래스)
        class_name = img_path.parent.name
        # 너무 일반적인 폴더명은 제외 (training/image/raw 등)
        if class_name.lower() in {
            "training", "validation", "test", "image", "images", "raw", "label", "labels"
        }:
            continue
        # 클래스명에 숫자 prefix가 있으면 제거: "001_김치찌개" -> "김치찌개"
        class_name = re.sub(r"^\d+[_-\s]*", "", class_name).strip()
        if not class_name:
            continue
        records.append({
            "image_path": str(img_path).replace("\\", "/"),  # 절대경로 (ROOT 밖에 있을 수 있음)
            "prompt": PROMPT_TEMPLATE,
            "answer": make_answer_from_class(class_name),
            "source": "aihub",
            "class": class_name,
        })
    print(f"  [aihub] {len(records)}개 이미지, "
          f"{len(set(r['class'] for r in records))}개 클래스")
    return records


def collect_calorie_ai() -> list:
    """calorie-ai 앱의 training-log + uploads."""
    records = []
    ca_root = CALORIE_AI_DIR
    if not ca_root.exists():
        print(f"  [skip] calorie-ai 폴더 없음: {ca_root}")
        return records

    for jsonl in (ca_root / "training").glob("*.jsonl"):
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 사용자 정정 결과만 학습 (event=confirm + userFinal 있음)
                if rec.get("event") != "confirm":
                    continue
                user_final = rec.get("userFinal") or []
                if not user_final:
                    continue
                # photoUrl: "/uploads/m_xxx.jpg" -> dataset/calorie_ai/uploads/m_xxx.jpg
                photo_url = rec.get("photoUrl", "")
                if not photo_url.startswith("/uploads/"):
                    continue
                img_rel = photo_url.replace("/uploads/", "uploads/")
                img_full = ca_root / img_rel
                if not img_full.exists():
                    continue
                records.append({
                    "image_path": str(img_full).replace("\\", "/"),  # 절대경로
                    "prompt": PROMPT_TEMPLATE,
                    "answer": make_answer_from_user_final(user_final),
                    "source": "calorie_ai",
                    "class": "_user",
                })
    print(f"  [calorie-ai] {len(records)}개 정정 샘플")
    return records


def balance_by_class(records: list, max_per_class: int = 500) -> list:
    """클래스당 이미지 수 균형. 너무 많은 클래스는 max_per_class로 잘라냄."""
    from collections import defaultdict
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r)
    balanced = []
    for cls, recs in by_class.items():
        random.shuffle(recs)
        balanced.extend(recs[:max_per_class])
    print(f"  [balance] {len(records)} -> {len(balanced)}개 "
          f"(클래스당 최대 {max_per_class}개)")
    return balanced


def main():
    random.seed(42)
    print("=== 비전 학습 데이터 준비 ===\n")

    records = []
    print("[1/3] AI Hub 데이터 수집...")
    records.extend(collect_aihub())

    print("\n[2/3] calorie-ai 사용자 데이터 수집...")
    records.extend(collect_calorie_ai())

    if not records:
        print("\n❌ 수집된 데이터 0개. dataset/ 폴더 확인.")
        return

    print(f"\n총 {len(records)}개 샘플\n")

    print("[3/3] 클래스 균형 + train/val 분할...")
    records = balance_by_class(records, max_per_class=500)
    random.shuffle(records)

    split = max(1, int(len(records) * 0.05))
    val_records = records[:split]
    train_records = records[split:]

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    write_jsonl(OUT_DIR / "train.jsonl", train_records)
    write_jsonl(OUT_DIR / "val.jsonl", val_records)

    print(f"\n✅ train: {len(train_records)}개 -> {OUT_DIR / 'train.jsonl'}")
    print(f"✅ val:   {len(val_records)}개 -> {OUT_DIR / 'val.jsonl'}")
    print(f"\n다음 단계: python train\\train_lora_vision.py")


if __name__ == "__main__":
    main()
