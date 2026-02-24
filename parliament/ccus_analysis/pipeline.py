from datetime import datetime, timezone

from .actor_extractor import ActorExtractor
from .api_client import OpenParliamentClient
from .bill_finder import CCUSBillFinder
from .hansard_fetcher import HansardFetcher, strip_html
from .jurisdiction_extractor import JurisdictionExtractor, SpacyJurisdictionExtractor
from .keywords import KeywordProvider, StaticCCUSKeywordProvider
from .models import BillAnalysis, CCUSAnalysisResult
from .opinion_classifier import LLMOpinionClassifier, OpinionClassifier


class CCUSAnalysisPipeline:
    def __init__(
        self,
        client: OpenParliamentClient,
        keyword_provider: KeywordProvider,
        bill_finder: CCUSBillFinder,
        hansard_fetcher: HansardFetcher,
        actor_extractor: ActorExtractor,
        opinion_classifier: OpinionClassifier,
        jurisdiction_extractor: JurisdictionExtractor,
    ):
        self.client = client
        self.keyword_provider = keyword_provider
        self.bill_finder = bill_finder
        self.hansard_fetcher = hansard_fetcher
        self.actor_extractor = actor_extractor
        self.opinion_classifier = opinion_classifier
        self.jurisdiction_extractor = jurisdiction_extractor

    def run(self) -> CCUSAnalysisResult:
        bills = self.bill_finder.find_bills()
        bill_analyses: list[BillAnalysis] = []

        for bill in bills:
            speeches = self.hansard_fetcher.get_speeches(bill)
            actors = self.actor_extractor.extract(speeches)
            opinions = [self.opinion_classifier.classify(actor, bill) for actor in actors]

            all_text = " ".join(
                (speech.get("content_text", {}) or {}).get("en", "")
                or strip_html((speech.get("content", {}) or {}).get("en", ""))
                for speech in speeches
            )
            jurisdictions = self.jurisdiction_extractor.extract(all_text)

            bill_analyses.append(BillAnalysis(
                bill=bill,
                speeches=speeches,
                actors=actors,
                opinions=opinions,
                jurisdictions=jurisdictions,
            ))

        generated_at = datetime.now(tz=timezone.utc).isoformat()
        return CCUSAnalysisResult(bills=bill_analyses, generated_at=generated_at)

    @classmethod
    def create_default(cls, base_url: str = "http://localhost:8000") -> "CCUSAnalysisPipeline":
        client = OpenParliamentClient(base_url=base_url)
        keyword_provider = StaticCCUSKeywordProvider()
        bill_finder = CCUSBillFinder(client=client, keyword_provider=keyword_provider)
        hansard_fetcher = HansardFetcher(client=client)
        actor_extractor = ActorExtractor()
        opinion_classifier = LLMOpinionClassifier()
        jurisdiction_extractor = SpacyJurisdictionExtractor()
        return cls(
            client=client,
            keyword_provider=keyword_provider,
            bill_finder=bill_finder,
            hansard_fetcher=hansard_fetcher,
            actor_extractor=actor_extractor,
            opinion_classifier=opinion_classifier,
            jurisdiction_extractor=jurisdiction_extractor,
        )
