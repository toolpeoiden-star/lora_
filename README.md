# lora_mcp

Qwen 2.5 LoRA 학습용 대화 데이터셋 저장소.

Claude Code 전역 슬래시 커맨드 `/학습`으로 직원이 선별한 대화를 ChatML JSONL 포맷으로 자동 변환·커밋·푸시합니다.

## 구조
- `dataset/` — 학습용 JSONL 파일. 파일명: `YYYYMMDD-HHMMSS-{slug}.jsonl`
- 각 파일은 frontmatter 메타데이터(품질점수, 도메인 태그, 작성자)를 첫 줄 주석으로 포함

## 포맷 (Qwen ChatML)
```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```
