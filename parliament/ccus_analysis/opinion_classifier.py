import json
from typing import Protocol

from django.conf import settings

from parliament.summaries.llm import get_llm_response, llms

from .hansard_fetcher import strip_html
from .models import Argument, Opinion, PoliticalActor


OPINION_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {
            "type": "string",
            "enum": ["support", "oppose", "neutral", "mixed"],
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "arguments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "economic",
                            "environmental",
                            "ethical",
                            "social",
                            "technical",
                            "jurisdictional",
                        ],
                    },
                    "text": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["type", "text", "quote"],
            },
        },
    },
    "required": ["stance", "confidence", "arguments"],
}

OPINION_INSTRUCTIONS = """You are an expert analyst of Canadian parliamentary debates.

Your task is to classify a politician's stance on Carbon Capture, Utilization and Storage (CCUS)
based on their speeches in Parliament. Analyse both:
1. Their stance on CCUS as a technology (does it work / is it viable?)
2. Their stance on CCUS as a policy tool (should the government support / fund it?)

Classify overall stance as one of:
- support: generally favourable, advocates for CCUS development or funding
- oppose: generally critical, argues against CCUS investment or effectiveness
- neutral: no clear position expressed
- mixed: acknowledges merits and drawbacks, or position is nuanced/conditional

For each distinct argument made, identify:
- type: one of economic | environmental | ethical | social | technical | jurisdictional
- text: a concise summary of the argument (1-2 sentences)
- quote: the most relevant verbatim excerpt from the speeches (keep it under 200 characters)

Respond in JSON matching the schema provided.
"""


class OpinionClassifier(Protocol):
    def classify(self, actor: PoliticalActor, bill: dict) -> Opinion: ...


class LLMOpinionClassifier:
    """Classifies politician stance on CCUS using a configurable LLM (default: Ollama)."""

    def __init__(self, model: str | None = None):
        self.model = model or getattr(settings, "CCUS_LLM_MODEL", llms.OLLAMA)

    def classify(self, actor: PoliticalActor, bill: dict) -> Opinion:
        bill_name = (bill.get("name") or {}).get("en") or bill.get("url", "Unknown bill")
        combined_text = self._combine_speeches(actor.speeches)

        prompt = f"Bill: {bill_name}\nPolitician: {actor.name}\n\n{combined_text}"

        response_text, _ = get_llm_response(
            OPINION_INSTRUCTIONS,
            prompt,
            model=self.model,
            json=OPINION_SCHEMA,
        )

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            data = {"stance": "neutral", "confidence": "low", "arguments": []}

        arguments = [
            Argument(
                type=arg.get("type", "economic"),
                text=arg.get("text", ""),
                quote=arg.get("quote", ""),
            )
            for arg in data.get("arguments", [])
        ]

        return Opinion(
            actor=actor,
            stance=data.get("stance", "neutral"),
            arguments=arguments,
            confidence=data.get("confidence", "low"),
        )

    def _combine_speeches(self, speeches: list[dict]) -> str:
        parts = []
        for speech in speeches:
            # Prefer already-stripped text; fall back to stripping HTML
            content_text = speech.get("content_text", {})
            if isinstance(content_text, dict):
                text = content_text.get("en") or content_text.get("fr") or ""
            else:
                content = speech.get("content", {})
                if isinstance(content, dict):
                    text = strip_html(content.get("en") or content.get("fr") or "")
                else:
                    text = ""
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)


# Backward compatibility
GeminiOpinionClassifier = LLMOpinionClassifier
