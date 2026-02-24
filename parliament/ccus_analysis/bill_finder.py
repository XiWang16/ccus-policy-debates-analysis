from .api_client import OpenParliamentClient
from .keywords import KeywordProvider


class CCUSBillFinder:
    def __init__(self, client: OpenParliamentClient, keyword_provider: KeywordProvider):
        self.client = client
        self.keyword_provider = keyword_provider

    def find_bills(self) -> list[dict]:
        keywords = [kw.lower() for kw in self.keyword_provider.get_keywords()]
        matched = []
        for bill in self.client.get_bills():
            if self._matches(bill, keywords):
                matched.append(bill)
        return matched

    def _matches(self, bill: dict, keywords: list[str]) -> bool:
        name_en = (bill.get("name", {}).get("en") or "").lower()
        name_fr = (bill.get("name", {}).get("fr") or "").lower()
        for kw in keywords:
            if kw in name_en or kw in name_fr:
                return True
        return False
