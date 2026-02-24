from typing import Protocol

# TODO: We need to expand this list to include more keywords.

CCUS_KEYWORDS = [
    "carbon capture",
    "carbon capture and storage",
    "CCS",
    "CCUS",
    "carbon sequestration",
    "CO2 storage",
    "geological storage",
    "carbon utilization",
    "direct air capture",
    "DAC",
    "underground storage",
    "enhanced oil recovery",
    "captage du carbone",  # French
]


class KeywordProvider(Protocol):
    def get_keywords(self) -> list[str]: ...


class StaticCCUSKeywordProvider:
    def get_keywords(self) -> list[str]:
        return CCUS_KEYWORDS
