"""Post-processing: convert JSONL answer outputs from informal to formal Korean style."""

import json
import logging
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "당신은 한국어 문체 변환 전문가입니다. "
    "주어진 텍스트의 해라체/하다체 문장 종결을 습니다체(합니다/입니다/됩니다/있습니다 등)로 바꿔주세요.\n"
    "규칙:\n"
    "- 문장 종결 어미만 바꾸고, 내용·어순·법률 용어·인용문·참고문헌은 절대 수정하지 마세요.\n"
    '- 법조문 인용 안의 원문(「」, "" 안)은 그대로 두세요.\n'
    "- 이미 습니다체인 문장은 그대로 두세요.\n"
    "- 변환된 전체 텍스트만 출력하세요. 설명 없이."
)


class FormalConverter:
    """Convert answer text style via LLM (informal -> formal Korean)."""

    def __init__(
        self,
        data_dir: Path,
        targets: list[str],
        api_key: str | None = None,
        model: str = "openai/gpt-4o-mini",
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    ):
        self.data_dir = data_dir
        self.targets = targets
        self.model = model
        self.system_prompt = system_prompt

        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set.")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)

    def convert_to_formal(self, text: str) -> str:
        """Convert a single text to formal style via LLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def process_file(self, filepath: Path):
        """Process a single JSONL file, writing atomically."""
        rows = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

        converted = 0
        for row in tqdm(rows, desc=filepath.name):
            original = row["output"]
            row["output"] = self.convert_to_formal(original)
            if row["output"] != original:
                converted += 1
            time.sleep(0.1)

        tmp_path = filepath.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, filepath)

        logger.info("%s: %d/%d converted", filepath.name, converted, len(rows))

    def run(self):
        """Convert all target JSONL files."""
        for name in self.targets:
            filepath = self.data_dir / name
            if not filepath.exists():
                logger.warning("Skipping %s (not found)", name)
                continue
            self.process_file(filepath)
