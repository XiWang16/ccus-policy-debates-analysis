"""
Tests for parliament.ccus_analysis.

Run with:
  DATABASE_URL=sqlite:///db.sqlite3 .venv/bin/python manage.py test parliament.ccus_analysis
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from django.test import TestCase

from .actor_extractor import ActorExtractor
from .api_client import OpenParliamentClient
from .bill_finder import CCUSBillFinder
from .hansard_fetcher import HansardFetcher, strip_html
from .jurisdiction_extractor import SpacyJurisdictionExtractor
from .keywords import StaticCCUSKeywordProvider
from .models import (
    Argument,
    BillAnalysis,
    CCUSAnalysisResult,
    Jurisdiction,
    Opinion,
    PoliticalActor,
)
from .opinion_classifier import GeminiOpinionClassifier
from .output import JSONOutputWriter
from .pipeline import CCUSAnalysisPipeline


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------

class StripHtmlTest(TestCase):
    def test_strips_tags(self):
        result = strip_html("<p>Hello <strong>world</strong></p>")
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertNotIn("<p>", result)

    def test_empty_string(self):
        self.assertEqual(strip_html(""), "")

    def test_plain_text_unchanged(self):
        self.assertEqual(strip_html("no tags here"), "no tags here")


# ---------------------------------------------------------------------------
# KeywordProvider
# ---------------------------------------------------------------------------

class KeywordProviderTest(TestCase):
    def test_returns_list(self):
        provider = StaticCCUSKeywordProvider()
        kws = provider.get_keywords()
        self.assertIsInstance(kws, list)
        self.assertGreater(len(kws), 0)

    def test_contains_ccus(self):
        provider = StaticCCUSKeywordProvider()
        kws = [k.lower() for k in provider.get_keywords()]
        self.assertIn("ccus", kws)

    def test_contains_french(self):
        provider = StaticCCUSKeywordProvider()
        kws = [k.lower() for k in provider.get_keywords()]
        self.assertTrue(any("captage" in k for k in kws))


# ---------------------------------------------------------------------------
# OpenParliamentClient (mocked HTTP)
# ---------------------------------------------------------------------------

class OpenParliamentClientTest(TestCase):
    def _make_client(self):
        return OpenParliamentClient(base_url="http://localhost:8000")

    @patch("parliament.ccus_analysis.api_client.requests.Session")
    def test_get_bills_paginates(self, MockSession):
        page1 = {
            "objects": [{"url": "/bills/44-1/C-1/"}, {"url": "/bills/44-1/C-2/"}],
            "pagination": {"next_url": "/bills/?page=2"},
        }
        page2 = {
            "objects": [{"url": "/bills/44-1/C-3/"}],
            "pagination": {"next_url": None},
        }
        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = page1
        mock_resp1.raise_for_status = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = page2
        mock_resp2.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.side_effect = [mock_resp1, mock_resp2]
        mock_session.headers = {}
        MockSession.return_value = mock_session

        client = self._make_client()
        bills = list(client.get_bills())
        self.assertEqual(len(bills), 3)

    @patch("parliament.ccus_analysis.api_client.requests.Session")
    def test_get_bill_detail(self, MockSession):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"url": "/bills/44-1/C-50/", "name": {"en": "A Bill"}}
        mock_resp.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_session.headers = {}
        MockSession.return_value = mock_session

        client = self._make_client()
        detail = client.get_bill_detail("/bills/44-1/C-50/")
        self.assertEqual(detail["name"]["en"], "A Bill")


# ---------------------------------------------------------------------------
# CCUSBillFinder
# ---------------------------------------------------------------------------

class BillFinderTest(TestCase):
    def _make_finder(self, bills):
        client = MagicMock()
        client.get_bills.return_value = iter(bills)
        provider = StaticCCUSKeywordProvider()
        return CCUSBillFinder(client=client, keyword_provider=provider)

    def test_finds_matching_bill(self):
        bills = [
            {"url": "/bills/44-1/C-1/", "name": {"en": "An Act respecting carbon capture", "fr": ""}},
            {"url": "/bills/44-1/C-2/", "name": {"en": "An Act about potatoes", "fr": ""}},
        ]
        finder = self._make_finder(bills)
        result = finder.find_bills()
        self.assertEqual(len(result), 1)
        self.assertIn("carbon capture", result[0]["name"]["en"])

    def test_no_match(self):
        bills = [
            {"url": "/bills/44-1/C-99/", "name": {"en": "Something unrelated", "fr": ""}},
        ]
        finder = self._make_finder(bills)
        self.assertEqual(finder.find_bills(), [])

    def test_case_insensitive(self):
        bills = [
            {"url": "/bills/44-1/C-5/", "name": {"en": "About CCUS Technology", "fr": ""}},
        ]
        finder = self._make_finder(bills)
        self.assertEqual(len(finder.find_bills()), 1)

    def test_french_keyword(self):
        bills = [
            {"url": "/bills/44-1/C-7/", "name": {"en": "", "fr": "Loi sur le captage du carbone"}},
        ]
        finder = self._make_finder(bills)
        self.assertEqual(len(finder.find_bills()), 1)


# ---------------------------------------------------------------------------
# HansardFetcher
# ---------------------------------------------------------------------------

class HansardFetcherTest(TestCase):
    def test_strips_html_from_content(self):
        speech = {
            "url": "/speeches/1/",
            "politician_url": "/politicians/1/",
            "content": {"en": "<p>We support <em>carbon capture</em> strongly.</p>"},
            "attribution": {"en": "Mr. Smith"},
        }
        client = MagicMock()
        client.get_speeches.return_value = iter([speech])
        fetcher = HansardFetcher(client=client)
        bill = {"url": "/bills/44-1/C-1/"}
        speeches = fetcher.get_speeches(bill)
        self.assertEqual(len(speeches), 1)
        self.assertIn("carbon capture", speeches[0]["content_text"]["en"])
        self.assertNotIn("<p>", speeches[0]["content_text"]["en"])

    def test_empty_content(self):
        speech = {
            "url": "/speeches/2/",
            "politician_url": None,
            "content": {},
            "attribution": {},
        }
        client = MagicMock()
        client.get_speeches.return_value = iter([speech])
        fetcher = HansardFetcher(client=client)
        speeches = fetcher.get_speeches({"url": "/bills/44-1/C-1/"})
        self.assertEqual(speeches[0]["content_text"], {})


# ---------------------------------------------------------------------------
# ActorExtractor
# ---------------------------------------------------------------------------

class ActorExtractorTest(TestCase):
    def test_groups_by_politician_url(self):
        speeches = [
            {"politician_url": "/politicians/1/", "attribution": {"en": "Alice"}, "content": {}},
            {"politician_url": "/politicians/1/", "attribution": {"en": "Alice"}, "content": {}},
            {"politician_url": "/politicians/2/", "attribution": {"en": "Bob"}, "content": {}},
        ]
        extractor = ActorExtractor()
        actors = extractor.extract(speeches)
        self.assertEqual(len(actors), 2)
        names = {a.name for a in actors}
        self.assertIn("Alice", names)
        self.assertIn("Bob", names)

    def test_anonymous_speeches_grouped(self):
        speeches = [
            {"politician_url": None, "attribution": {}, "content": {}},
            {"politician_url": None, "attribution": {}, "content": {}},
        ]
        extractor = ActorExtractor()
        actors = extractor.extract(speeches)
        self.assertEqual(len(actors), 1)
        self.assertIsNone(actors[0].politician_url)
        self.assertEqual(len(actors[0].speeches), 2)

    def test_each_actor_has_correct_speech_count(self):
        speeches = [
            {"politician_url": "/politicians/5/", "attribution": {"en": "Carol"}, "content": {}},
            {"politician_url": "/politicians/5/", "attribution": {"en": "Carol"}, "content": {}},
            {"politician_url": "/politicians/5/", "attribution": {"en": "Carol"}, "content": {}},
        ]
        extractor = ActorExtractor()
        actors = extractor.extract(speeches)
        self.assertEqual(len(actors), 1)
        self.assertEqual(len(actors[0].speeches), 3)


# ---------------------------------------------------------------------------
# OpinionClassifier
# ---------------------------------------------------------------------------

class GeminiOpinionClassifierTest(TestCase):
    @patch("parliament.ccus_analysis.opinion_classifier.get_llm_response")
    def test_classify_support(self, mock_llm):
        response_json = json.dumps({
            "stance": "support",
            "confidence": "high",
            "arguments": [
                {
                    "type": "economic",
                    "text": "CCUS creates jobs.",
                    "quote": "CCUS will create thousands of jobs.",
                }
            ],
        })
        mock_llm.return_value = (response_json, {})

        actor = PoliticalActor(
            name="Alice",
            politician_url="/politicians/1/",
            party="Green",
            speeches=[{
                "content_text": {"en": "I support CCUS wholeheartedly."},
            }],
        )
        bill = {"url": "/bills/44-1/C-1/", "name": {"en": "CCUS Act"}}
        classifier = GeminiOpinionClassifier()
        opinion = classifier.classify(actor, bill)
        self.assertEqual(opinion.stance, "support")
        self.assertEqual(opinion.confidence, "high")
        self.assertEqual(len(opinion.arguments), 1)
        self.assertEqual(opinion.arguments[0].type, "economic")

    @patch("parliament.ccus_analysis.opinion_classifier.get_llm_response")
    def test_classify_handles_bad_json(self, mock_llm):
        mock_llm.return_value = ("not valid json {{", {})
        actor = PoliticalActor(name="Bob", politician_url=None, party=None, speeches=[])
        bill = {"url": "/bills/44-1/C-2/"}
        classifier = GeminiOpinionClassifier()
        opinion = classifier.classify(actor, bill)
        self.assertEqual(opinion.stance, "neutral")
        self.assertEqual(opinion.confidence, "low")


# ---------------------------------------------------------------------------
# JurisdictionExtractor
# ---------------------------------------------------------------------------

class SpacyJurisdictionExtractorTest(TestCase):
    def setUp(self):
        try:
            self.extractor = SpacyJurisdictionExtractor()
        except RuntimeError:
            self.skipTest("spaCy model en_core_web_sm not installed")

    def test_extracts_gpe(self):
        text = "Canada and Alberta have significant interest in carbon capture projects."
        results = self.extractor.extract(text)
        entities = {j.entity for j in results}
        # At least one of the GPEs should be detected
        self.assertTrue(
            entities & {"Canada", "Alberta"},
            f"Expected GPE entities, got: {entities}",
        )

    def test_deduplicates(self):
        text = "Canada is great. Canada supports CCS. Canada will invest."
        results = self.extractor.extract(text)
        canada_hits = [j for j in results if j.entity == "Canada"]
        self.assertEqual(len(canada_hits), 1)

    def test_empty_text(self):
        results = self.extractor.extract("")
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# JSONOutputWriter
# ---------------------------------------------------------------------------

class JSONOutputWriterTest(TestCase):
    def _make_result(self) -> CCUSAnalysisResult:
        actor = PoliticalActor(name="Alice", politician_url="/politicians/1/", party="NDP", speeches=[])
        argument = Argument(type="economic", text="Creates jobs", quote="Jobs, jobs, jobs")
        opinion = Opinion(actor=actor, stance="support", arguments=[argument], confidence="high")
        jurisdiction = Jurisdiction(entity="Canada", label="GPE", context="Canada will invest.")
        bill_analysis = BillAnalysis(
            bill={"url": "/bills/44-1/C-1/", "name": {"en": "CCUS Act", "fr": ""}},
            speeches=[],
            actors=[actor],
            opinions=[opinion],
            jurisdictions=[jurisdiction],
        )
        return CCUSAnalysisResult(bills=[bill_analysis], generated_at="2026-02-21T00:00:00+00:00")

    def test_writes_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONOutputWriter(Path(tmpdir))
            result = self._make_result()
            writer.write(result)

            expected_files = [
                "ccus_bills.json",
                "ccus_speeches.json",
                "ccus_opinions.json",
                "ccus_jurisdictions.json",
                "ccus_summary.json",
            ]
            for fname in expected_files:
                path = Path(tmpdir) / fname
                self.assertTrue(path.exists(), f"Missing: {fname}")

    def test_summary_contains_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONOutputWriter(Path(tmpdir))
            result = self._make_result()
            writer.write(result)

            with open(Path(tmpdir) / "ccus_summary.json") as f:
                summary = json.load(f)
            self.assertEqual(summary["bill_count"], 1)
            self.assertEqual(summary["total_opinions"], 1)
            self.assertIn("support", summary["stance_breakdown"])

    def test_opinions_file_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONOutputWriter(Path(tmpdir))
            result = self._make_result()
            writer.write(result)

            with open(Path(tmpdir) / "ccus_opinions.json") as f:
                opinions = json.load(f)
            self.assertEqual(len(opinions), 1)
            self.assertEqual(opinions[0]["opinions"][0]["stance"], "support")
            self.assertEqual(opinions[0]["opinions"][0]["actor"]["name"], "Alice")


# ---------------------------------------------------------------------------
# Pipeline (integration-style with mocks)
# ---------------------------------------------------------------------------

class CCUSAnalysisPipelineTest(TestCase):
    def test_run_produces_result(self):
        bill = {"url": "/bills/44-1/C-1/", "name": {"en": "CCUS Act", "fr": ""}}
        speech = {
            "politician_url": "/politicians/1/",
            "attribution": {"en": "Alice"},
            "content": {"en": "<p>I support CCUS.</p>"},
            "content_text": {"en": "I support CCUS."},
        }
        actor = PoliticalActor(name="Alice", politician_url="/politicians/1/", party=None, speeches=[speech])
        opinion = Opinion(actor=actor, stance="support", arguments=[], confidence="high")

        mock_bill_finder = MagicMock()
        mock_bill_finder.find_bills.return_value = [bill]

        mock_hansard = MagicMock()
        mock_hansard.get_speeches.return_value = [speech]

        mock_actor_extractor = MagicMock()
        mock_actor_extractor.extract.return_value = [actor]

        mock_opinion_classifier = MagicMock()
        mock_opinion_classifier.classify.return_value = opinion

        mock_jurisdiction_extractor = MagicMock()
        mock_jurisdiction_extractor.extract.return_value = [
            Jurisdiction(entity="Canada", label="GPE", context="Canada will invest.")
        ]

        pipeline = CCUSAnalysisPipeline(
            client=MagicMock(),
            keyword_provider=MagicMock(),
            bill_finder=mock_bill_finder,
            hansard_fetcher=mock_hansard,
            actor_extractor=mock_actor_extractor,
            opinion_classifier=mock_opinion_classifier,
            jurisdiction_extractor=mock_jurisdiction_extractor,
        )

        result = pipeline.run()
        self.assertIsInstance(result, CCUSAnalysisResult)
        self.assertEqual(len(result.bills), 1)
        self.assertEqual(result.bills[0].opinions[0].stance, "support")
        self.assertEqual(result.bills[0].jurisdictions[0].entity, "Canada")
