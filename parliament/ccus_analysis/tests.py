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
from .manual_bills import get_manual_bill_numbers, is_valid_bill_number
from .models import (
    Argument,
    BillAnalysis,
    CCUSAnalysisResult,
    Jurisdiction,
    Opinion,
    PoliticalActor,
)
from .opinion_classifier import GeminiOpinionClassifier
from .output import CSVOutputWriter, JSONOutputWriter
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

    def test_expanded_keywords_present(self):
        provider = StaticCCUSKeywordProvider()
        kws = [k.lower() for k in provider.get_keywords()]
        for expected in ("direct air capture", "blue hydrogen", "enhanced oil recovery"):
            self.assertIn(expected, kws, f"Expected keyword missing: {expected}")


# ---------------------------------------------------------------------------
# Manual bills validation
# ---------------------------------------------------------------------------

class ManualBillsTest(TestCase):
    def test_valid_house_bill(self):
        self.assertTrue(is_valid_bill_number("C-50"))

    def test_valid_senate_bill(self):
        self.assertTrue(is_valid_bill_number("S-243"))

    def test_invalid_lowercase(self):
        self.assertFalse(is_valid_bill_number("c-50"))

    def test_invalid_no_dash(self):
        self.assertFalse(is_valid_bill_number("C50"))

    def test_invalid_wrong_prefix(self):
        self.assertFalse(is_valid_bill_number("B-10"))

    def test_invalid_empty(self):
        self.assertFalse(is_valid_bill_number(""))

    def test_get_manual_bill_numbers_returns_list(self):
        numbers = get_manual_bill_numbers()
        self.assertIsInstance(numbers, list)
        # All returned numbers must be valid
        for n in numbers:
            self.assertTrue(is_valid_bill_number(n), f"Invalid number in list: {n}")

    def test_manual_list_contains_expected_bills(self):
        numbers = get_manual_bill_numbers()
        for expected in ("S-243", "C-50", "C-59"):
            self.assertIn(expected, numbers)


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
    def _make_finder(self, bills, search_full_text=False):
        """Build a finder with a mocked client. Full-text disabled by default
        so tests don't need to mock the detail endpoint."""
        client = MagicMock()
        client.get_bills.return_value = iter(bills)
        provider = StaticCCUSKeywordProvider()
        return CCUSBillFinder(
            client=client,
            keyword_provider=provider,
            search_full_text=search_full_text,
        )

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

    def test_find_by_number_calls_api_with_filter(self):
        client = MagicMock()
        client.get_bills.return_value = iter([
            {"url": "/bills/44-1/C-50/", "number": "C-50", "name": {"en": "Sustainable Jobs Act"}},
        ])
        provider = StaticCCUSKeywordProvider()
        finder = CCUSBillFinder(client=client, keyword_provider=provider, search_full_text=False)
        results = finder.find_by_number("C-50")
        client.get_bills.assert_called_once_with(number="C-50")
        self.assertEqual(len(results), 1)

    def test_speeches_contain_keywords_true(self):
        speeches = [
            {"content_text": {"en": "We strongly support carbon capture technology."}},
        ]
        client = MagicMock()
        finder = CCUSBillFinder(client=client, keyword_provider=StaticCCUSKeywordProvider(), search_full_text=False)
        self.assertTrue(finder.speeches_contain_keywords(speeches))

    def test_speeches_contain_keywords_false(self):
        speeches = [
            {"content_text": {"en": "This is about agriculture subsidies."}},
        ]
        client = MagicMock()
        finder = CCUSBillFinder(client=client, keyword_provider=StaticCCUSKeywordProvider(), search_full_text=False)
        self.assertFalse(finder.speeches_contain_keywords(speeches))

    def test_bill_contains_keywords_via_full_text(self):
        client = MagicMock()
        client.get_bill_detail.return_value = {
            "full_text": {"en": "This bill establishes a framework for carbon capture and storage.", "fr": ""},
            "short_title": {"en": "", "fr": ""},
        }
        finder = CCUSBillFinder(client=client, keyword_provider=StaticCCUSKeywordProvider(), search_full_text=False)
        bill = {"url": "/bills/44-1/C-99/"}
        self.assertTrue(finder.bill_contains_keywords(bill))

    def test_bill_contains_keywords_no_full_text(self):
        client = MagicMock()
        client.get_bill_detail.return_value = {"full_text": None, "short_title": {"en": "", "fr": ""}}
        finder = CCUSBillFinder(client=client, keyword_provider=StaticCCUSKeywordProvider(), search_full_text=False)
        bill = {"url": "/bills/44-1/C-99/"}
        self.assertFalse(finder.bill_contains_keywords(bill))

    def test_find_bills_with_full_text_search(self):
        """A bill not matching title should be found when its full text contains a keyword."""
        bills = [
            {"url": "/bills/44-1/C-99/", "name": {"en": "An Act about something", "fr": ""}},
        ]
        client = MagicMock()
        client.get_bills.return_value = iter(bills)
        client.get_bill_detail.return_value = {
            "full_text": {"en": "Establishes CCUS investment requirements.", "fr": ""},
            "short_title": {"en": "", "fr": ""},
        }
        provider = StaticCCUSKeywordProvider()
        finder = CCUSBillFinder(client=client, keyword_provider=provider, search_full_text=True)
        results = finder.find_bills()
        self.assertEqual(len(results), 1)


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
            match_reason="keyword",
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

    def test_summary_includes_match_reason(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONOutputWriter(Path(tmpdir))
            result = self._make_result()
            writer.write(result)

            with open(Path(tmpdir) / "ccus_summary.json") as f:
                summary = json.load(f)
            self.assertEqual(summary["bills"][0]["match_reason"], "keyword")


# ---------------------------------------------------------------------------
# CSVOutputWriter
# ---------------------------------------------------------------------------

class CSVOutputWriterTest(TestCase):
    def _make_result(self) -> CCUSAnalysisResult:
        actor = PoliticalActor(
            name="Alice",
            politician_url="/politicians/1/",
            party="NDP",
            speeches=[{"time": "2024-03-15 14:00:00"}],
        )
        argument = Argument(type="environmental", text="Reduces emissions", quote="This will cut emissions.")
        opinion_support = Opinion(actor=actor, stance="support", arguments=[argument], confidence="high")

        actor2 = PoliticalActor(
            name="Bob",
            politician_url="/politicians/2/",
            party="Conservative",
            speeches=[{"time": "2024-03-16 10:00:00"}],
        )
        opinion_oppose = Opinion(actor=actor2, stance="oppose", arguments=[], confidence="medium")

        jurisdiction = Jurisdiction(entity="Alberta", label="GPE", context="Alberta will lead CCS efforts.")
        bill_analysis = BillAnalysis(
            bill={
                "url": "/bills/44-1/C-59/",
                "number": "C-59",
                "session": "44-1",
                "name": {"en": "Fall Economic Statement Implementation Act 2023", "fr": ""},
                "introduced": "2023-11-21",
                "status_code": "RoyalAssentGiven",
                "home_chamber": "House",
                "sponsor_politician_url": "/politicians/trudeau/",
            },
            speeches=[],
            actors=[actor, actor2],
            opinions=[opinion_support, opinion_oppose],
            jurisdictions=[jurisdiction],
            match_reason="manual",
        )
        return CCUSAnalysisResult(bills=[bill_analysis], generated_at="2026-02-21T00:00:00+00:00")

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "csv"
            writer = CSVOutputWriter(out)
            writer.write(self._make_result())
            self.assertTrue(out.is_dir())

    def test_writes_all_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "csv"
            writer = CSVOutputWriter(out)
            writer.write(self._make_result())
            for fname in (
                "ccus_bills.csv",
                "ccus_actors_support.csv",
                "ccus_actors_opposition.csv",
                "ccus_arguments.csv",
                "ccus_jurisdictions.csv",
            ):
                self.assertTrue((out / fname).exists(), f"Missing: {fname}")

    def test_bills_csv_columns(self):
        import csv as csv_module
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "csv"
            writer = CSVOutputWriter(out)
            writer.write(self._make_result())
            with open(out / "ccus_bills.csv", encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bill_number"], "C-59")
            self.assertEqual(rows[0]["session"], "44-1")
            self.assertEqual(rows[0]["match_reason"], "manual")

    def test_actors_support_csv(self):
        import csv as csv_module
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "csv"
            writer = CSVOutputWriter(out)
            writer.write(self._make_result())
            with open(out / "ccus_actors_support.csv", encoding="utf-8") as f:
                rows = list(csv_module.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["actor_name"], "Alice")
            self.assertEqual(rows[0]["bill_number"], "C-59")

    def test_actors_opposition_csv(self):
        import csv as csv_module
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "csv"
            writer = CSVOutputWriter(out)
            writer.write(self._make_result())
            with open(out / "ccus_actors_opposition.csv", encoding="utf-8") as f:
                rows = list(csv_module.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["actor_name"], "Bob")

    def test_arguments_csv(self):
        import csv as csv_module
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "csv"
            writer = CSVOutputWriter(out)
            writer.write(self._make_result())
            with open(out / "ccus_arguments.csv", encoding="utf-8") as f:
                rows = list(csv_module.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["actor_name"], "Alice")
            self.assertEqual(rows[0]["argument_label"], "environmental")
            self.assertIn("Reduces", rows[0]["argument_text"])

    def test_jurisdictions_csv(self):
        import csv as csv_module
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "csv"
            writer = CSVOutputWriter(out)
            writer.write(self._make_result())
            with open(out / "ccus_jurisdictions.csv", encoding="utf-8") as f:
                rows = list(csv_module.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_name"], "Alberta")
            self.assertEqual(rows[0]["entity_type"], "GPE")
            self.assertEqual(rows[0]["bill_number"], "C-59")


# ---------------------------------------------------------------------------
# Pipeline (integration-style with mocks)
# ---------------------------------------------------------------------------

class CCUSAnalysisPipelineTest(TestCase):
    def test_run_produces_result(self):
        bill = {"url": "/bills/44-1/C-1/", "number": "C-1", "session": "44-1",
                "name": {"en": "CCUS Act", "fr": ""}}
        speech = {
            "politician_url": "/politicians/1/",
            "attribution": {"en": "Alice"},
            "content": {"en": "<p>I support CCUS.</p>"},
            "content_text": {"en": "I support CCUS."},
        }
        actor = PoliticalActor(name="Alice", politician_url="/politicians/1/", party=None, speeches=[speech])
        opinion = Opinion(actor=actor, stance="support", arguments=[], confidence="high")

        mock_bill_finder = MagicMock()
        mock_bill_finder.find_by_number.return_value = [bill]
        mock_bill_finder.bill_contains_keywords.return_value = True
        mock_bill_finder.speeches_contain_keywords.return_value = False

        mock_hansard = MagicMock()
        mock_hansard.get_speeches.return_value = [speech]

        mock_actor_extractor = MagicMock()
        mock_actor_extractor.extract.return_value = [actor]

        mock_opinion_classifier = MagicMock()
        mock_opinion_classifier.classify.return_value = opinion


        pipeline = CCUSAnalysisPipeline(
            client=MagicMock(),
            keyword_provider=MagicMock(),
            bill_finder=mock_bill_finder,
            hansard_fetcher=mock_hansard,
            actor_extractor=mock_actor_extractor,
            opinion_classifier=mock_opinion_classifier,
            manual_bill_numbers=["C-1"],
        )

        result = pipeline.run()
        self.assertIsInstance(result, CCUSAnalysisResult)
        self.assertEqual(len(result.bills), 1)
        self.assertEqual(result.bills[0].opinions[0].stance, "support")
        self.assertEqual(result.bills[0].match_reason, "manual")

    def test_manual_bills_included(self):
        """Both manual bill numbers produce a BillAnalysis each (one with keywords, one without)."""
        bill_c1 = {"url": "/bills/44-1/C-1/", "number": "C-1", "session": "44-1",
                   "name": {"en": "CCUS Act", "fr": ""}}
        bill_c50 = {"url": "/bills/44-1/C-50/", "number": "C-50", "session": "44-1",
                    "name": {"en": "Sustainable Jobs Act", "fr": ""}}

        mock_bill_finder = MagicMock()
        mock_bill_finder.find_by_number.side_effect = [[bill_c1], [bill_c50]]
        # C-1 has keywords, C-50 does not
        mock_bill_finder.bill_contains_keywords.side_effect = [True, False]
        mock_bill_finder.speeches_contain_keywords.side_effect = [False, False]

        mock_hansard = MagicMock()
        mock_hansard.get_speeches.return_value = []

        mock_actor_extractor = MagicMock()
        mock_actor_extractor.extract.return_value = []

        mock_opinion_classifier = MagicMock()

        pipeline = CCUSAnalysisPipeline(
            client=MagicMock(),
            keyword_provider=MagicMock(),
            bill_finder=mock_bill_finder,
            hansard_fetcher=mock_hansard,
            actor_extractor=mock_actor_extractor,
            opinion_classifier=mock_opinion_classifier,
            manual_bill_numbers=["C-1", "C-50"],
        )

        result = pipeline.run()
        self.assertEqual(len(result.bills), 2)
        # All results are from the manual list
        self.assertTrue(all(ba.match_reason == "manual" for ba in result.bills))

    def test_single_manual_bill_produces_one_result(self):
        """A single manual bill number produces exactly one BillAnalysis."""
        bill = {"url": "/bills/44-1/C-50/", "number": "C-50", "session": "44-1",
                "name": {"en": "Sustainable Jobs Act", "fr": ""}}

        mock_bill_finder = MagicMock()
        mock_bill_finder.find_by_number.return_value = [bill]
        mock_bill_finder.bill_contains_keywords.return_value = False
        mock_bill_finder.speeches_contain_keywords.return_value = False

        mock_hansard = MagicMock()
        mock_hansard.get_speeches.return_value = []

        mock_actor_extractor = MagicMock()
        mock_actor_extractor.extract.return_value = []

        mock_opinion_classifier = MagicMock()

        pipeline = CCUSAnalysisPipeline(
            client=MagicMock(),
            keyword_provider=MagicMock(),
            bill_finder=mock_bill_finder,
            hansard_fetcher=mock_hansard,
            actor_extractor=mock_actor_extractor,
            opinion_classifier=mock_opinion_classifier,
            manual_bill_numbers=["C-50"],
        )

        result = pipeline.run()
        self.assertEqual(len(result.bills), 1)
        self.assertEqual(result.bills[0].match_reason, "manual")

    def test_manual_bill_prints_warning_when_no_keywords(self):
        """A manual bill with no CCUS keywords in text or speeches should print a warning."""
        manual_bill = {
            "url": "/bills/44-1/C-69/",
            "number": "C-69",
            "session": "44-1",
            "name": {"en": "Impact Assessment Act", "fr": ""},
        }

        mock_bill_finder = MagicMock()
        mock_bill_finder.find_bills.return_value = []
        mock_bill_finder.find_by_number.return_value = [manual_bill]
        mock_bill_finder.bill_contains_keywords.return_value = False
        mock_bill_finder.speeches_contain_keywords.return_value = False

        mock_hansard = MagicMock()
        mock_hansard.get_speeches.return_value = []

        mock_actor_extractor = MagicMock()
        mock_actor_extractor.extract.return_value = []
        mock_opinion_classifier = MagicMock()

        pipeline = CCUSAnalysisPipeline(
            client=MagicMock(),
            keyword_provider=MagicMock(),
            bill_finder=mock_bill_finder,
            hansard_fetcher=mock_hansard,
            actor_extractor=mock_actor_extractor,
            opinion_classifier=mock_opinion_classifier,
            manual_bill_numbers=["C-69"],
        )

        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = pipeline.run()

        output = buf.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("C-69", output)
        # Bill is still included in results despite no keyword match
        self.assertEqual(len(result.bills), 1)
        self.assertEqual(result.bills[0].match_reason, "manual")
