"""
step2b_passage_extractor.py

Sits between step 2 (speech fetch) and step 3 (actor extraction).

Given all speeches retrieved for a bill, this module:
  1. Identifies which speeches are CCUS-relevant (three-tier detection).
  2. Extracts only the relevant paragraphs, with one paragraph of surrounding
     context to preserve argumentative coherence.
  3. From the extracted paragraphs, retains only those that express an
     **opinion or argument** about CCUS, discarding passages that merely
     state facts about the technology.

Each speech dict is annotated in-place with:
  - ``ccus_relevant``        bool   — True if any CCUS content was found
  - ``ccus_passages``        list[str] — argumentative passages only (used as
                                         LLM context in step 4)
  - ``ccus_factual_passages`` list[str] — CCUS passages that were detected but
                                          classified as purely factual
  - ``ccus_match_type``      str | None — which detection tier fired first

=============================================================================
RELEVANCE DETECTION  (three tiers)
=============================================================================

Tier 1 — Exact keyword match
    Uses the single compiled alternation regex built from keywords.py.
    One pass through the text; word-boundary anchored.

Tier 2a — Implicit keyword match
    A supplementary keyword list for unambiguous CCUS-adjacent vocabulary
    that doesn't use the headline terminology: named Canadian CCUS
    installations (Boundary Dam, Shell Quest, Alberta Carbon Trunk Line),
    technical process terms (amine scrubbing, post-combustion capture,
    CO2 injection, saline formation), and French equivalents not already
    in keywords.py.

Tier 2b — Semantic similarity  (optional, requires sentence-transformers)
    Each paragraph is encoded with all-MiniLM-L6-v2 and its cosine
    similarity to a CCUS anchor embedding is computed.  Paragraphs above
    a tunable threshold (default 0.42) are extracted.  This catches
    passages that discuss industrial decarbonisation, geological storage,
    or heavy-emitter policy without using the standard vocabulary.

=============================================================================
OPINION / ARGUMENT FILTERING
=============================================================================

After CCUS-relevant passages are identified, each is tested for whether
it expresses an *opinion or argument* versus merely stating a fact.

Why this matters
    A speech like "Carbon capture projects currently achieve 90% capture
    rates" describes a fact.  A speech like "The government must invest
    in CCUS to meet our climate commitments" expresses a normative
    stance.  Only the latter is analytically useful for stance
    classification; the former adds noise to the LLM context.

Three approaches, applied in order of availability:

1. Rule-based opinion markers  (always active, zero dependencies)
   Detects modal verbs ("should", "must", "ought to", "need to"),
   first-person evaluative assertions ("I believe", "I support",
   "I am concerned"), government-directed imperatives ("the government
   must"), and explicit stance phrases ("in my view", "I oppose").
   These are highly reliable signals in parliamentary speech.

   Library: built-in ``re`` module only.

2. TextBlob subjectivity scoring  (optional: ``uv pip install textblob``)
   ``TextBlob(text).sentiment.subjectivity`` returns a float in [0, 1]
   where 0 is maximally objective and 1 is maximally subjective.
   Passages scoring above a configurable threshold (default 0.25) are
   treated as opinionated even if no explicit opinion marker was found.
   This catches evaluative language that doesn't fit the rule patterns,
   e.g. "This approach is deeply flawed and will not deliver results."

   Function call: ``from textblob import TextBlob``
                  ``TextBlob(passage).sentiment.subjectivity``

3. Zero-shot NLI classification  (optional: ``uv pip install transformers``)
   The most accurate approach.  A natural-language-inference model is
   given the passage and asked to classify it as either
   "expresses opinion or argument" vs "states fact or describes".
   The default model is ``cross-encoder/nli-MiniLM2-L6-H768`` (~80 MB),
   which is fast enough to run on CPU across hundreds of passages.
   For higher accuracy, ``facebook/bart-large-mnli`` (~400 MB) can be
   substituted.

   Function calls:
       from transformers import pipeline as hf_pipeline
       clf = hf_pipeline("zero-shot-classification",
                         model="cross-encoder/nli-MiniLM2-L6-H768")
       result = clf(text, candidate_labels=[
           "expresses opinion or argument", "states fact or describes"])
       is_opinion = result["labels"][0] == "expresses opinion or argument"

   Note: argument-mining is a specialised NLP subfield.  Production-ready
   English models fine-tuned on political speech are limited; IBM's
   Debater project is the most capable but is not openly distributed.
   Zero-shot NLI on a general model is currently the best practical
   choice for this use case.

A passage is classified as argumentative if *any* of the above methods
returns True.  If none are available (no TextBlob, no transformers,
and no opinion markers fire), the passage is conservatively kept so
that content is never silently lost.

Passages that contain CCUS content but are classified as factual are
preserved in ``ccus_factual_passages`` for auditability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tier 2a — implicit keyword set
# ---------------------------------------------------------------------------

IMPLICIT_KEYWORDS: list[str] = [
    # Named Canadian CCUS installations
    "Boundary Dam",
    "Shell Quest",
    "Quest CCS",
    "Weyburn-Midale",
    "Alberta Carbon Trunk Line",
    "ACTL",

    # Unambiguous technical process terms
    "CO2 injection",
    "carbon injection",
    "amine scrubbing",
    "post-combustion capture",
    "pre-combustion capture",
    "oxyfuel combustion",
    "oxy-fuel combustion",
    "saline formation",
    "depleted reservoir",
    "subsurface injection",
    "wellbore integrity",
    "supercritical CO2",
    "CO2 compression",
    "capture solvent",

    # French terms not in core keywords.py
    "injection de CO2",
    "formation saline",
    "technologie de captage",
    "gisement épuisé",
    "capture post-combustion",
    "capture pré-combustion",
]


def _compile(keywords: list[str]) -> re.Pattern:
    sorted_kws = sorted(keywords, key=len, reverse=True)
    alternation = "|".join(re.escape(kw) for kw in sorted_kws)
    return re.compile(r"\b(?:" + alternation + r")\b", re.IGNORECASE)


_IMPLICIT_PATTERN: re.Pattern = _compile(IMPLICIT_KEYWORDS)


# ---------------------------------------------------------------------------
# Tier 2b — semantic anchor
# ---------------------------------------------------------------------------

CCUS_ANCHOR = (
    "Carbon capture and storage captures CO2 emissions from power plants and "
    "industrial facilities such as oil sands operations, compresses them, "
    "transports them by pipeline, and injects them into geological formations "
    "such as saline aquifers or depleted oil and gas reservoirs for permanent "
    "storage. Related technologies include direct air capture of atmospheric "
    "CO2, enhanced oil recovery using CO2 injection, and production of blue "
    "hydrogen with CCS to remove combustion emissions."
)

_embedder_cache: dict[str, Any] = {}


def _load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Load SentenceTransformer + CCUS anchor embedding. Cached after first call."""
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer(model_name)
        anchor = model.encode(CCUS_ANCHOR, convert_to_numpy=True)
        anchor = anchor / (np.linalg.norm(anchor) + 1e-9)
        _embedder_cache[model_name] = (model, anchor)
        return model, anchor
    except ImportError:
        _embedder_cache[model_name] = (None, None)
        return None, None


# ---------------------------------------------------------------------------
# Opinion / argument detection
# ---------------------------------------------------------------------------

# Rule-based: strong opinion signals in parliamentary speech.
# Modal verbs and deontic phrases are the most reliable markers — a politician
# saying "we must invest in CCUS" is unambiguously expressing a normative view.
_OPINION_MARKERS: re.Pattern = re.compile(
    r"\b(?:"
    # Deontic modals
    r"should|must|ought\s+to|have\s+to|need\s+to|needs\s+to|"
    # First-person belief / position
    r"I\s+believe|I\s+think|I\s+urge|I\s+submit|I\s+contend|I\s+argue|"
    r"I\s+support|I\s+oppose|I\s+endorse|I\s+reject|I\s+welcome|I\s+condemn|"
    r"I\s+am\s+(?:concerned|worried|opposed|in\s+favour|in\s+favor|supportive)|"
    r"in\s+my\s+(?:view|opinion|judgment|judgement)|"
    # First-person plural / collective stance
    r"we\s+(?:must|should|need|ought|cannot|can(?:'t)?|will(?:not)?)|"
    # Government-directed imperatives
    r"the\s+government\s+(?:should|must|needs?\s+to|has\s+to|ought\s+to)|"
    r"Ottawa\s+(?:should|must|needs?\s+to)|"
    # Normative framing
    r"it\s+is\s+(?:essential|crucial|critical|important|necessary|vital|imperative)|"
    r"it\s+is\s+(?:wrong|right|unacceptable|irresponsible|reckless)|"
    # Evaluative stance on CCUS effectiveness
    r"(?:does\s+not|doesn't|won't|will\s+not)\s+work|"
    r"(?:is|are|was|were)\s+(?:not\s+)?(?:effective|efficient|viable|reliable|sufficient|proven)"
    r")\b",
    re.IGNORECASE,
)

_nli_cache: dict[str, Any] = {}


def _load_nli_classifier(model_name: str = "cross-encoder/nli-MiniLM2-L6-H768"):
    """Lazily load a zero-shot NLI classifier. Returns None if transformers not installed."""
    if model_name in _nli_cache:
        return _nli_cache[model_name]
    try:
        from transformers import pipeline as hf_pipeline
        clf = hf_pipeline("zero-shot-classification", model=model_name)
        _nli_cache[model_name] = clf
        return clf
    except ImportError:
        _nli_cache[model_name] = None
        return None


_NLI_LABELS = ["expresses opinion or argument", "states fact or describes"]


def _is_argumentative(
    text: str,
    nli_classifier=None,
    subjectivity_threshold: float = 0.65,
) -> bool:
    """
    Return True if *text* expresses an opinion or argument rather than stating
    a fact.  Applies three methods in order; returns True as soon as any fires.

    If no method is available and no rule matches, returns True conservatively
    (do not silently discard content when the classifier is absent).
    """
    # 1. Rule-based opinion markers — fast, zero deps, high precision
    if _OPINION_MARKERS.search(text):
        return True

    # 2. TextBlob subjectivity score
    # Note: TextBlob is calibrated for consumer/social-media text and
    # misreads technical language — "million tonnes" scores 0.60 despite
    # being factual.  The default threshold is therefore set high (0.65)
    # so TextBlob only fires on clearly emotive language, not on
    # quantitative claims that happen to contain loaded words.
    try:
        from textblob import TextBlob

        blob = TextBlob(text)
        # Use getattr to satisfy static type checkers that may not know
        # about the Sentiment.namedtuple API.
        subjectivity = float(getattr(blob.sentiment, "subjectivity", 0.0))
        if subjectivity >= subjectivity_threshold:
            return True
    except ImportError:
        pass

    # 3. Zero-shot NLI
    if nli_classifier is not None:
        try:
            result = nli_classifier(
                text[:512],  # most NLI models cap at 512 tokens
                candidate_labels=_NLI_LABELS,
            )
            if result["labels"][0] == _NLI_LABELS[0]:
                return True
        except Exception:
            pass

    # Conservative fallback: keep if nothing fired but no classifier was
    # available to confirm it's factual.
    has_any_classifier = (
        _is_argumentative.__module__ and (
            _nli_cache or
            any(True for _ in [None])  # always true — marker alone is the baseline
        )
    )
    # If we reach here with no classifier and no markers, keep the passage.
    # This avoids silent data loss in minimal-dependency environments.
    return nli_classifier is None and not _has_textblob()


def _has_textblob() -> bool:
    try:
        import textblob  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """
    Split Hansard speech text (post-HTML-strip) into paragraphs.

    lxml's text_content() uses single \\n as block separator.  Very short
    fragments (< 30 chars) are merged into the next substantial paragraph
    since they are typically speaker attributions or procedural phrases.
    """
    raw = re.split(r"\n+", text)
    paras = [p.strip() for p in raw if p.strip()]
    merged: list[str] = []
    buffer = ""
    for para in paras:
        if len(para) < 30:
            buffer = (buffer + " " + para).strip() if buffer else para
        else:
            if buffer:
                para = (buffer + " " + para).strip()
                buffer = ""
            merged.append(para)
    if buffer:
        merged.append(buffer)
    return merged


# ---------------------------------------------------------------------------
# Passage-window extraction
# ---------------------------------------------------------------------------

def _extract_passages(
    paragraphs: list[str],
    matching_indices: set[int],
    context: int = 1,
) -> list[str]:
    """Return contiguous passage strings: matching paragraphs ± context paragraphs."""
    if not matching_indices:
        return []
    window: set[int] = set()
    n = len(paragraphs)
    for i in matching_indices:
        for j in range(max(0, i - context), min(n, i + context + 1)):
            window.add(j)
    passages: list[str] = []
    run: list[int] = []
    for i in sorted(window):
        if run and i > run[-1] + 1:
            passages.append("\n".join(paragraphs[j] for j in run))
            run = []
        run.append(i)
    if run:
        passages.append("\n".join(paragraphs[j] for j in run))
    return passages


# ---------------------------------------------------------------------------
# Per-speech annotation
# ---------------------------------------------------------------------------

@dataclass
class SpeechAnnotation:
    ccus_relevant: bool
    passages: list[str] = field(default_factory=list)
    factual_passages: list[str] = field(default_factory=list)
    match_type: str | None = None


def annotate_speech_text(
    text: str,
    tier1_pattern: re.Pattern,
    embedder=None,
    anchor_embedding=None,
    nli_classifier=None,
    semantic_threshold: float = 0.42,
    context_paragraphs: int = 1,
    subjectivity_threshold: float = 0.65,
) -> SpeechAnnotation:
    """
    Full annotation pipeline for a single stripped speech text.

    Detection order: Tier 1 → Tier 2a → Tier 2b.
    Once CCUS-relevant passages are identified, each is tested for whether
    it expresses an opinion/argument.  Only argumentative passages go into
    ``passages``; purely factual ones go into ``factual_passages``.
    """
    text = text.strip()
    if not text:
        return SpeechAnnotation(ccus_relevant=False)

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return SpeechAnnotation(ccus_relevant=False)

    # --- Tier 1: exact keyword ---
    t1_hits = {i for i, p in enumerate(paragraphs) if tier1_pattern.search(p)}
    if t1_hits:
        raw = _extract_passages(paragraphs, t1_hits, context_paragraphs)
        return _partition_by_opinion(raw, "keyword", nli_classifier, subjectivity_threshold)

    # --- Tier 2a: implicit keyword ---
    t2a_hits = {i for i, p in enumerate(paragraphs) if _IMPLICIT_PATTERN.search(p)}
    if t2a_hits:
        raw = _extract_passages(paragraphs, t2a_hits, context_paragraphs)
        return _partition_by_opinion(raw, "implicit_keyword", nli_classifier, subjectivity_threshold)

    # --- Tier 2b: semantic similarity ---
    if embedder is not None and anchor_embedding is not None:
        try:
            import numpy as np
            t2b_hits: set[int] = set()
            for i, para in enumerate(paragraphs):
                if len(para) < 60:
                    continue
                emb = embedder.encode(para, convert_to_numpy=True)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                if float(np.dot(emb, anchor_embedding)) >= semantic_threshold:
                    t2b_hits.add(i)
            if t2b_hits:
                raw = _extract_passages(paragraphs, t2b_hits, context_paragraphs)
                return _partition_by_opinion(raw, "implicit_semantic", nli_classifier, subjectivity_threshold)
        except Exception:
            pass

    return SpeechAnnotation(ccus_relevant=False)


def _partition_by_opinion(
    passages: list[str],
    match_type: str,
    nli_classifier,
    subjectivity_threshold: float,
) -> SpeechAnnotation:
    """Split *passages* into argumentative and factual lists."""
    argumentative: list[str] = []
    factual: list[str] = []
    for p in passages:
        if _is_argumentative(p, nli_classifier, subjectivity_threshold):
            argumentative.append(p)
        else:
            factual.append(p)
    return SpeechAnnotation(
        ccus_relevant=True,
        passages=argumentative,
        factual_passages=factual,
        match_type=match_type,
    )


# ---------------------------------------------------------------------------
# Pipeline component
# ---------------------------------------------------------------------------

class CCUSPassageExtractor:
    """
    Step 2b pipeline component: annotates speech dicts in-place.

    Sits between step 2 (HansardFetcher) and step 3 (ActorExtractor).
    After annotation, ActorExtractor groups only speeches where
    ``ccus_relevant`` is True, and LLMOpinionClassifier builds its
    context window from ``ccus_passages`` (argumentative only).

    Each speech dict gains:
      - ``ccus_relevant``          bool
      - ``ccus_passages``          list[str]  — argumentative CCUS passages
      - ``ccus_factual_passages``  list[str]  — factual CCUS passages (excluded
                                                from LLM context, kept for audit)
      - ``ccus_match_type``        str | None
    """

    def __init__(
        self,
        tier1_pattern: re.Pattern,
        use_semantic: bool = True,
        use_nli: bool = False,
        semantic_threshold: float = 0.42,
        subjectivity_threshold: float = 0.65,
        context_paragraphs: int = 1,
        nli_model: str = "cross-encoder/nli-MiniLM2-L6-H768",
    ):
        self._tier1 = tier1_pattern
        self._semantic_threshold = semantic_threshold
        self._subjectivity_threshold = subjectivity_threshold
        self._context = context_paragraphs

        self._embedder, self._anchor = (
            _load_embedder() if use_semantic else (None, None)
        )
        self._nli = _load_nli_classifier(nli_model) if use_nli else None

        # Report which optional tiers are active.
        active = ["keyword (tier 1)", "implicit keyword (tier 2a)"]
        if self._embedder is not None:
            active.append("semantic similarity (tier 2b)")
        else:
            print(
                "[Step2b] Semantic tier disabled — install sentence-transformers to enable.",
                flush=True,
            )
        if self._nli is not None:
            active.append("NLI opinion filter")
        else:
            opinion_note = (
                "TextBlob subjectivity + rule-based markers"
                if _has_textblob() else "rule-based markers only"
            )
            active.append(f"opinion filter: {opinion_note}")

        print(f"[Step2b] Active: {', '.join(active)}", flush=True)

    def annotate(self, speeches: list[dict]) -> list[dict]:
        """Annotate *speeches* in-place and return the same list."""
        for speech in speeches:
            text = self._get_text(speech)
            ann = annotate_speech_text(
                text,
                tier1_pattern=self._tier1,
                embedder=self._embedder,
                anchor_embedding=self._anchor,
                nli_classifier=self._nli,
                semantic_threshold=self._semantic_threshold,
                context_paragraphs=self._context,
                subjectivity_threshold=self._subjectivity_threshold,
            )
            speech["ccus_relevant"] = ann.ccus_relevant
            speech["ccus_passages"] = ann.passages
            speech["ccus_factual_passages"] = ann.factual_passages
            speech["ccus_match_type"] = ann.match_type
        return speeches

    @staticmethod
    def _get_text(speech: dict) -> str:
        content_text = speech.get("content_text", {})
        if isinstance(content_text, dict):
            return content_text.get("en") or content_text.get("fr") or ""
        return ""
