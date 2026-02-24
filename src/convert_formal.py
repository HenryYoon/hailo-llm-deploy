"""
JSONL 답변 습니다체 변환 스크립트
==================================
raft_builder.py 실행 후 JSONL의 output 필드를 습니다체로 변환.

Usage:
    python src/convert_formal.py
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "processed"
TARGETS = ["train_dataset.jsonl", "val_dataset.jsonl", "test_dataset.jsonl"]

SYSTEM_PROMPT = (
    "당신은 한국어 문체 변환 전문가입니다. "
    "주어진 텍스트의 해라체/하다체 문장 종결을 습니다체(합니다/입니다/됩니다/있습니다 등)로 바꿔주세요.\n"
    "규칙:\n"
    "- 문장 종결 어미만 바꾸고, 내용·어순·법률 용어·인용문·참고문헌은 절대 수정하지 마세요.\n"
    '- 법조문 인용 안의 원문(「」, "" 안)은 그대로 두세요.\n'
    "- 이미 습니다체인 문장은 그대로 두세요.\n"
    "- 변환된 전체 텍스트만 출력하세요. 설명 없이."
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


def convert_to_formal(text: str) -> str:
    """Convert a single text to 습니다체 via LLM."""
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def process_file(filepath: Path):
    """Process a single JSONL file."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    converted = 0
    for row in tqdm(rows, desc=filepath.name):
        original = row["output"]
        row["output"] = convert_to_formal(original)
        if row["output"] != original:
            converted += 1
        time.sleep(0.1)  # rate limit

    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  {filepath.name}: {converted}/{len(rows)} converted")


def main():
    print("=" * 50)
    print("JSONL 습니다체 변환")
    print("=" * 50)

    for name in TARGETS:
        filepath = DATA_DIR / name
        if not filepath.exists():
            print(f"  [SKIP] {name} not found")
            continue
        process_file(filepath)

    print("\nDone!")


if __name__ == "__main__":
    main()
