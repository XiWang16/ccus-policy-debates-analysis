from .models import PoliticalActor


class ActorExtractor:
    def extract(self, speeches: list[dict]) -> list[PoliticalActor]:
        """Group speeches by politician_url; one PoliticalActor per unique politician."""
        actors: dict[str, PoliticalActor] = {}
        anonymous: list[dict] = []

        for speech in speeches:
            politician_url = speech.get("politician_url") or speech.get("politician")
            if not politician_url:
                anonymous.append(speech)
                continue

            if politician_url not in actors:
                name = self._resolve_name(speech)
                party = self._resolve_party(speech)
                actors[politician_url] = PoliticalActor(
                    name=name,
                    politician_url=politician_url,
                    party=party,
                    speeches=[],
                )
            actors[politician_url].speeches.append(speech)

        result = list(actors.values())

        if anonymous:
            result.append(PoliticalActor(
                name="Unknown",
                politician_url=None,
                party=None,
                speeches=anonymous,
            ))

        return result

    def _resolve_name(self, speech: dict) -> str:
        attribution = speech.get("attribution", {})
        if isinstance(attribution, dict):
            return attribution.get("en") or attribution.get("fr") or "Unknown"
        return str(attribution) if attribution else "Unknown"

    def _resolve_party(self, speech: dict) -> str | None:
        # Party info isn't directly in the speech response; return None for now.
        return None
