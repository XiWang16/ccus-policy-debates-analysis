from typing import Protocol

from .models import Jurisdiction


class JurisdictionExtractor(Protocol):
    def extract(self, text: str) -> list[Jurisdiction]: ...


class SpacyJurisdictionExtractor:
    def __init__(self):
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

    def extract(self, text: str) -> list[Jurisdiction]:
        doc = self.nlp(text)
        results: list[Jurisdiction] = []
        seen: set[tuple[str, str]] = set()

        for ent in doc.ents:
            if ent.label_ not in ("GPE", "ORG"):
                continue
            key = (ent.text.lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)

            context = self._get_sentence(ent)
            results.append(Jurisdiction(
                entity=ent.text,
                label=ent.label_,
                context=context,
            ))

        return results

    def _get_sentence(self, ent) -> str:
        try:
            return ent.sent.text.strip()
        except Exception:
            # spaCy sentence boundaries not available; fall back to a window
            start = max(0, ent.start_char - 100)
            end = min(len(ent.doc.text), ent.end_char + 100)
            return ent.doc.text[start:end]
