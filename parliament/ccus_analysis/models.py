from dataclasses import dataclass, field


@dataclass
class PoliticalActor:
    name: str
    politician_url: str | None
    party: str | None
    speeches: list[dict] = field(default_factory=list)


@dataclass
class Argument:
    type: str   # economic | environmental | ethical | social | technical | jurisdictional
    text: str
    quote: str


@dataclass
class Opinion:
    actor: PoliticalActor
    stance: str               # support | oppose | neutral | mixed
    arguments: list[Argument]
    confidence: str           # high | medium | low


@dataclass
class Jurisdiction:
    entity: str
    label: str                # GPE | ORG
    context: str              # surrounding sentence


@dataclass
class BillAnalysis:
    bill: dict
    speeches: list[dict]
    actors: list[PoliticalActor]
    opinions: list[Opinion]
    jurisdictions: list[Jurisdiction]
    # How this bill was identified as CCUS-related:
    #   "keyword"  — matched a CCUS keyword in the title or full text
    #   "manual"   — explicitly listed in MANUAL_CCUS_BILL_NUMBERS
    match_reason: str = "keyword"


@dataclass
class CCUSAnalysisResult:
    bills: list[BillAnalysis]
    generated_at: str
