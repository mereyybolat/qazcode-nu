from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a clinical coding assistant. "
    "Return ONLY valid JSON with this schema: "
    '{"diagnoses":[{"rank":1,"diagnosis":"...","icd10_code":"...","explanation":"..."}]}. '
    "No markdown."
)


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("QAZCODE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key. Set QAZCODE_API_KEY.")

        self.base_url = base_url or os.getenv("QAZCODE_HUB_URL", "https://hub.qazcode.ai")
        self.model = model or os.getenv("QAZCODE_MODEL", "oss-120b")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def rank_diagnoses(
        self,
        symptoms: str,
        retrieved_context: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        prompt_obj = {
            "task": "Rank likely diagnoses using provided clinical protocol context.",
            "symptoms": symptoms,
            "top_k": top_k,
            "context": [
                {
                    "protocol_id": item.get("protocol_id"),
                    "title": item.get("title", ""),
                    "icd_codes": item.get("icd_codes", []),
                    "text": item.get("text", "")[:1500],
                }
                for item in retrieved_context
            ],
        }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt_obj, ensure_ascii=False)},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        parsed = self._safe_json_extract(content)
        diagnoses = parsed.get("diagnoses", [])
        if not isinstance(diagnoses, list):
            return []

        cleaned: list[dict[str, Any]] = []
        for i, d in enumerate(diagnoses[:top_k], start=1):
            if not isinstance(d, dict):
                continue
            cleaned.append(
                {
                    "rank": int(d.get("rank", i)),
                    "diagnosis": str(d.get("diagnosis", "")).strip(),
                    "icd10_code": str(d.get("icd10_code", "")).strip(),
                    "explanation": str(d.get("explanation", "")).strip(),
                }
            )
        return cleaned

    @staticmethod
    def _safe_json_extract(text: str) -> dict[str, Any]:
        if not text:
            return {}
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                obj = json.loads(candidate)
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}
