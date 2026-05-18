"""
dataset/*.jsonl 파일들을 모아 학습용 train.jsonl / val.jsonl 생성.
- // meta: 주석 줄 제거
- 빈 줄 제거
- JSON 유효성 검증
- 9:1 train/val 분할
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "train" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

records = []
skipped = 0

for jsonl_file in sorted(DATASET_DIR.glob("*.jsonl")):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
                if "messages" not in obj or not isinstance(obj["messages"], list):
                    skipped += 1
                    continue
                if len(obj["messages"]) < 2:
                    skipped += 1
                    continue
                records.append(obj)
            except json.JSONDecodeError as e:
                print(f"  [skip] {jsonl_file.name}:{lineno} - {e}")
                skipped += 1

print(f"\n총 {len(records)}개 샘플 로드, {skipped}개 스킵")

if len(records) < 10:
    print(f"⚠️  샘플이 너무 적습니다 ({len(records)}개). 최소 50개 이상 권장.")

random.shuffle(records)
split = max(1, int(len(records) * 0.1))
val_records = records[:split]
train_records = records[split:]

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

write_jsonl(OUT_DIR / "train.jsonl", train_records)
write_jsonl(OUT_DIR / "val.jsonl", val_records)

print(f"✅ train: {len(train_records)}개 → {OUT_DIR / 'train.jsonl'}")
print(f"✅ val:   {len(val_records)}개 → {OUT_DIR / 'val.jsonl'}")
print(f"\n다음 단계: python train\\train_lora.py")
