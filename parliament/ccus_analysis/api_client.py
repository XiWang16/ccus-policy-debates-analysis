from collections.abc import Iterator
from urllib.parse import urljoin

import requests


class OpenParliamentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get_json(self, url: str, **params) -> dict:
        params.setdefault("format", "json")
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _paginate(self, url: str, **params) -> Iterator[dict]:
        """Yield individual objects from a paginated list endpoint."""
        params.setdefault("format", "json")
        next_url: str | None = url
        while next_url:
            data = self._get_json(next_url, **params)
            params = {}  # subsequent pages use the full next_url already
            for obj in data.get("objects", []):
                yield obj
            pagination = data.get("pagination", {})
            next_url = pagination.get("next_url")
            if next_url:
                # next_url may be a relative path
                next_url = urljoin(self.base_url, next_url)

    def get_bills(self, **params) -> Iterator[dict]:
        url = f"{self.base_url}/bills/"
        yield from self._paginate(url, **params)

    def get_speeches(self, bill_url: str) -> Iterator[dict]:
        """Fetch all speeches for a bill. bill_url is the bill's API URL path."""
        url = f"{self.base_url}/speeches/"
        yield from self._paginate(url, bill_debated=bill_url)

    def get_bill_detail(self, bill_url: str) -> dict:
        """Fetch a single bill by its URL (relative or absolute)."""
        url = urljoin(self.base_url, bill_url)
        return self._get_json(url)
