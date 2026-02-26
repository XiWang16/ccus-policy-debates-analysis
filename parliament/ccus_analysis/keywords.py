from typing import Protocol

CCUS_KEYWORDS = [
    # Core CCUS/CCS terms
    "carbon capture",
    "carbon capture and storage",
    "carbon capture, utilization, and storage",
    "carbon capture and utilization",
    "CCS",
    "CCUS",

    # Carbon sequestration / storage
    "carbon sequestration",
    "carbon storage",
    "CO2 storage",
    "CO2 capture",
    "carbon dioxide capture",
    "carbon dioxide storage",

    # Geological / underground storage
    "geological storage",
    "geological sequestration",
    "underground storage",
    "subsurface storage",
    "saline aquifer storage",
    "deep saline formation",
    "offshore carbon storage",

    # Direct air capture
    "direct air capture",
    "DAC",
    "atmospheric carbon removal",
    "engineered carbon removal",
    "point source capture",

    # Utilization
    "carbon utilization",
    "carbon dioxide utilization",
    "CO2 utilization",
    "carbon mineralization",

    # Enhanced oil/gas recovery
    "enhanced oil recovery",
    "EOR",
    "CO2-EOR",
    "enhanced gas recovery",

    # Hydrogen (CCUS-related)
    "blue hydrogen",
    "low-carbon hydrogen",
    "clean hydrogen",
    "hydrogen with CCS",

    # Infrastructure / industrial
    "CO2 pipeline",
    "carbon dioxide pipeline",
    "carbon transport infrastructure",
    "carbon storage hub",
    "industrial carbon capture",
    "carbon capture facility",
    "capture and sequestration",

    # Tax credits / investment policy
    "carbon capture tax credit",
    "investment tax credit for carbon capture",
    "CCUS investment tax credit",
    "carbon capture credit",

    # Specific programs / legislation context
    "emissions reduction fund",
    "carbon offset",
    "net-zero emissions",

    # French terms
    "captage du carbone",
    "captage et stockage du carbone",
    "captage, utilisation et stockage du carbone",
    "séquestration du carbone",
    "stockage géologique",
    "captage direct dans l'air",
    "récupération assistée du pétrole",
    "hydrogène bleu",
    "crédit d'impôt pour le captage",
    "stockage souterrain du carbone",
    "utilisation du carbone",
    "stockage du dioxyde de carbone",
    "captage industriel du carbone",
]


class KeywordProvider(Protocol):
    def get_keywords(self) -> list[str]: ...


class StaticCCUSKeywordProvider:
    def get_keywords(self) -> list[str]:
        return CCUS_KEYWORDS
