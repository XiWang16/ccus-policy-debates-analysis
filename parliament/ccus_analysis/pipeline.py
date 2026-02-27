from datetime import datetime, timezone

from .config import API_BASE_URL
from .step1_bill_finder import CCUSBillFinder
from .step2_hansard_fetcher import HansardFetcher
from .step3_actor_extractor import ActorExtractor
from .step4_opinion_classifier import LLMOpinionClassifier, OpinionClassifier
from .keywords import KeywordProvider, StaticCCUSKeywordProvider
from .manual_bills import get_manual_bill_entries, parse_bill_entry
from .models import BillAnalysis, CCUSAnalysisResult
from .api_client import OpenParliamentClient


class CCUSAnalysisPipeline:
    """
    Deployment pipeline: analyses only the bills listed in manual_bills.py.

    For each bill entry (optionally session-pinned, e.g. "C-50/39-2") the pipeline:
      1. Looks up the matching session(s) via the bills API.
      2. Fetches Hansard speeches for each matched session.
      3. Validates that the bill contains at least one CCUS keyword (in its full
         text or in the speeches). Prints a WARNING if none are found.
      4. If keywords are confirmed, runs full analysis:
           - actor extraction (who spoke on this bill, procedural speeches skipped)
           - LLM opinion classification (stance + arguments per actor)
      5. Bills without keyword confirmation are still included in output (for
         traceability) but with empty actors / opinions.

    The `CCUSBillFinder.find_bills()` keyword-scan utility remains available for
    exploratory use but is **not** invoked by this pipeline.
    """

    def __init__(
        self,
        client: OpenParliamentClient,
        keyword_provider: KeywordProvider,
        bill_finder: CCUSBillFinder,
        hansard_fetcher: HansardFetcher,
        actor_extractor: ActorExtractor,
        opinion_classifier: OpinionClassifier,
        manual_bill_numbers: list[str] | None = None,
    ):
        self.client = client
        self.keyword_provider = keyword_provider
        self.bill_finder = bill_finder
        self.hansard_fetcher = hansard_fetcher
        self.actor_extractor = actor_extractor
        self.opinion_classifier = opinion_classifier
        # Parse each entry into (number, session_or_None). Entries may be plain
        # bill numbers ("C-50") or session-pinned ("C-50/39-2").
        raw_entries: list[str] = (
            manual_bill_numbers if manual_bill_numbers is not None
            else [f"{n}/{s}" if s else n for n, s in get_manual_bill_entries()]
        )
        self.manual_bill_entries: list[tuple[str, str | None]] = [
            parse_bill_entry(e) for e in raw_entries
        ]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> CCUSAnalysisResult:
        """Process all bills in the manual list and return analysis results."""
        keywords = self.keyword_provider.get_keywords()
        bill_analyses: list[BillAnalysis] = []

        print(
            f"[Pipeline] Starting analysis of {len(self.manual_bill_entries)} "
            "manual bill entry/entries...",
            flush=True,
        )

        for number, pinned_session in self.manual_bill_entries:
            found = self.bill_finder.find_by_number(number)
            if not found:
                print(f"[Pipeline] WARNING: No bills found for '{number}' in the API", flush=True)
                continue

            # Filter to the pinned session if one was specified.
            if pinned_session:
                found = [b for b in found if b.get("session") == pinned_session]
                if not found:
                    print(
                        f"[Pipeline] WARNING: No bill found for '{number}' "
                        f"in session '{pinned_session}'",
                        flush=True,
                    )
                    continue

            label = f"{number}/{pinned_session}" if pinned_session else number
            print(
                f"[Pipeline] '{label}' matched {len(found)} session(s) in the API",
                flush=True,
            )

            for bill in found:
                bill_name = (bill.get("name") or {}).get("en", "")
                session = bill.get("session", "")
                print(
                    f"[Pipeline]   Fetching speeches for {number} ({session}): "
                    f"{bill_name[:70]}",
                    flush=True,
                )
                speeches = self.hansard_fetcher.get_speeches(bill)
                print(
                    f"[Pipeline]   {number} ({session}): {len(speeches)} speech(es) fetched",
                    flush=True,
                )

                # Keyword validation — checks bill full text + Hansard speeches.
                has_kw = (
                    self.bill_finder.bill_contains_keywords(bill, keywords)
                    or self.bill_finder.speeches_contain_keywords(speeches, keywords)
                )

                if not has_kw:
                    print(
                        f"[Pipeline]   WARNING: No CCUS keywords found for {number} "
                        f"('{bill_name}', session {session}). "
                        "Included in output but skipping LLM analysis.",
                        flush=True,
                    )
                    bill_analyses.append(BillAnalysis(
                        bill=bill,
                        speeches=speeches,
                        actors=[],
                        opinions=[],
                        jurisdictions=[],
                        match_reason="manual",
                    ))
                    continue

                print(
                    f"[Pipeline]   Keywords confirmed for {number} ({session}). "
                    "Running full analysis...",
                    flush=True,
                )
                bill_analyses.append(
                    self._analyze_bill_with_speeches(bill, speeches, match_reason="manual")
                )

        generated_at = datetime.now(tz=timezone.utc).isoformat()
        print(
            f"[Pipeline] Complete. {len(bill_analyses)} bill record(s) produced.",
            flush=True,
        )
        return CCUSAnalysisResult(bills=bill_analyses, generated_at=generated_at)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_bill_with_speeches(
        self, bill: dict, speeches: list[dict], match_reason: str
    ) -> BillAnalysis:
        actors = self.actor_extractor.extract(speeches)
        print(
            f"[Pipeline]     {len(actors)} actor(s) found, classifying opinions...",
            flush=True,
        )
        opinions = []
        for i, actor in enumerate(actors, 1):
            print(
                f"[Pipeline]     Classifying actor {i}/{len(actors)}: {actor.name}",
                flush=True,
            )
            opinions.append(self.opinion_classifier.classify(actor, bill))

        return BillAnalysis(
            bill=bill,
            speeches=speeches,
            actors=actors,
            opinions=opinions,
            jurisdictions=[],
            match_reason=match_reason,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create_default(cls, base_url: str = API_BASE_URL) -> "CCUSAnalysisPipeline":
        client = OpenParliamentClient(base_url=base_url)
        keyword_provider = StaticCCUSKeywordProvider()
        # search_full_text=False: the per-bill detail fetch is only used for
        # keyword *validation* on manual bills, not for mass discovery.
        bill_finder = CCUSBillFinder(
            client=client,
            keyword_provider=keyword_provider,
            search_full_text=False,
        )
        hansard_fetcher = HansardFetcher(client=client)
        actor_extractor = ActorExtractor()
        opinion_classifier = LLMOpinionClassifier()
        return cls(
            client=client,
            keyword_provider=keyword_provider,
            bill_finder=bill_finder,
            hansard_fetcher=hansard_fetcher,
            actor_extractor=actor_extractor,
            opinion_classifier=opinion_classifier,
        )
