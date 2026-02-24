import re

from lxml.html.clean import Cleaner

from .api_client import OpenParliamentClient


_html_cleaner = Cleaner(safe_attrs_only=True, remove_tags=["a", "span", "em", "strong", "b", "i"])


def strip_html(html: str) -> str:
    """Return plain text from an HTML string."""
    if not html:
        return ""
    try:
        import lxml.html
        doc = lxml.html.fromstring(html)
        return doc.text_content()
    except Exception:
        # Fallback: naive tag strip
        return re.sub(r"<[^>]+>", " ", html)


class HansardFetcher:
    def __init__(self, client: OpenParliamentClient):
        self.client = client

    def get_speeches(self, bill: dict) -> list[dict]:
        """Return all speeches for a bill with HTML stripped from content."""
        bill_url = bill.get("url", "")
        speeches = []
        for speech in self.client.get_speeches(bill_url):
            speech = dict(speech)
            content = speech.get("content", {})
            if isinstance(content, dict):
                speech["content_text"] = {
                    lang: strip_html(html) for lang, html in content.items()
                }
            speeches.append(speech)
        return speeches
