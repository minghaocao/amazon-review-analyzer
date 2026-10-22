"""单条评论的结构化 Aspect-Based Sentiment Analysis 管道。"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from aspect_taxonomy import PRODUCT_ASPECTS, STANDARD_ASPECTS, normalize_aspect
from llm_client import LLMClientError, ReviewAnalysisClient, create_review_client


ALLOWED_SENTIMENTS = ("positive", "negative", "neutral")
LOGGER = logging.getLogger(__name__)

_PRICE_VALUE_SUPPORT_RE = re.compile(
    r"\b(?:price|priced|value|cost|money|afford(?:able|ability)?|budget|"
    r"bargain|overpriced|underpriced|good deal|bang for (?:the )?buck|"
    r"for what you get|what you get for|worth (?:it|buying|the purchase|"
    r"paying|the cost|the price))\b",
    flags=re.IGNORECASE,
)
_MONETARY_AMOUNT_RE = re.compile(
    r"(?:[$£€¥]\s*\d|\b\d+(?:\.\d{1,2})?\s*(?:dollars?|usd|gbp|eur|cny|rmb)\b)",
    flags=re.IGNORECASE,
)
_COMPARISON_TARGET_RE = re.compile(
    r"\b(?:(?:the|my|our|an?)\s+)?(?:old|previous|prior|former|rival|"
    r"competitor|alternative)\s+(?:product|item|device|unit|model|one|"
    r"television|tv|monitor|screen|display|set|speaker|headphones?|appliance)\b",
    flags=re.IGNORECASE,
)
_REVIEWED_TARGET_RE = re.compile(
    r"\b(?:this|the reviewed|the new)\s+(?:product|item|device|unit|model|"
    r"one|television|tv|monitor|speaker|headphones?|appliance)\b",
    flags=re.IGNORECASE,
)
_CONTRAST_RE = re.compile(
    r"\b(?:whereas|while|but|although|though|compared (?:with|to)|than)\b",
    flags=re.IGNORECASE,
)
_MINIMUM_ACCEPTABILITY_RE = re.compile(
    r"\b(?:still|remain(?:s|ed)?|yet)\s+usable\b|"
    r"\bworks?\s+(?:well\s+)?for\s+(?:basic|simple|limited)\b|"
    r"\b(?:barely usable|not completely unusable)\b",
    flags=re.IGNORECASE,
)
_NEGATIVE_QUALIFIER_CONTEXT_RE = re.compile(
    r"\b(?:bad|poor|weak|slow|sluggish|lag(?:gy|ging)?|freeze|freezing|"
    r"crash(?:es|ing)?|problem|issue|difficult|hard|unresponsive|"
    r"unreliable|distort(?:s|ed|ion)?)\b",
    flags=re.IGNORECASE,
)
_UNUSED_SERVICE_RE = re.compile(
    r"\b(?:without|never|did not|didn't)\s+(?:having to\s+)?"
    r"(?:call(?:ing|ed)?|contact(?:ing|ed)?|us(?:e|ing|ed)|reach(?:ing|ed)?)\s+"
    r"(?:the\s+)?(?:customer\s+)?(?:support|service|seller)\b",
    flags=re.IGNORECASE,
)
_FACTUAL_PRICE_MATCH_RE = re.compile(
    r"(?:\b(?:exactly|same)\b.*\b(?:listed|advertised|stated|shown)\b|"
    r"\b(?:listed|advertised|stated|shown)\b.*\b(?:exactly|same)\b)",
    flags=re.IGNORECASE,
)
_PRICE_EVALUATION_RE = re.compile(
    r"\b(?:good|great|excellent|fair|reasonable|generous|attractive|strong|"
    r"compelling|poor|weak|low|cheap|affordable|expensive|costly|overpriced|"
    r"bargain|deal|value|worth|worthwhile|justif(?:y|ies|ied)|happy|disappointed)\b",
    flags=re.IGNORECASE,
)
_TRANSACTION_DOCUMENT_RE = re.compile(
    r"\b(?:receipt|invoice|coupon|voucher|transaction|order confirmation|"
    r"payment record|billing statement)\b",
    flags=re.IGNORECASE,
)
_TRANSACTION_FACT_PREDICATE_RE = re.compile(
    r"\b(?:lists?|shows?|records?|states?|contains?|appear(?:s|ed)?|was applied|"
    r"were applied|is applied|are applied|charged|paid|payment of)\b",
    flags=re.IGNORECASE,
)
_PRICE_FRAME_RE = re.compile(
    r"\b(?:at this price|for the price|similarly priced|comparably priced)\b",
    flags=re.IGNORECASE,
)
_COMPARISON_LANGUAGE_RE = re.compile(
    r"\b(?:compared with|compared to|against|versus|relative to)\b",
    flags=re.IGNORECASE,
)
_FAILURE_STATE_RE = re.compile(
    r"\b(?:defect(?:ive)?|faulty|broken|malfunction(?:ing)?|failed?)\b",
    flags=re.IGNORECASE,
)
_OPERATION_FAILURE_RE = re.compile(
    r"\b(?:power on|turn on|start|operate|function|work)\b",
    flags=re.IGNORECASE,
)

# These boundary rules deliberately require both an opinion target and a
# predicate.  A noun alone (for example ``box`` or ``menu``) is never enough to
# override the model's taxonomy choice.
_PACKAGING_TARGET_RE = re.compile(
    r"\b(?:box|carton|package|packaging|packing|foam|padding|wrap|wrapper|"
    r"insert|inserts|sleeve|tape|corner guards?|cushioning|filler|"
    r"bubble wrap|protective material|packing material)\b",
    flags=re.IGNORECASE,
)
_PACKAGING_PREDICATE_RE = re.compile(
    r"\b(?:crush(?:ed|ing)?|tear|torn|rip(?:ped)?|dent(?:ed)?|damage(?:d)?|"
    r"intact|secure|sturdy|weak|thin|flimsy|loose|missing|scuff(?:ed)?|"
    r"protect(?:ed|ive)?|cushion(?:ed|ing)?|securely|secured|seal(?:ed)?|unsealed|taped|"
    r"well[ -]?pack(?:ed)?|poorly[ -]?pack(?:ed)?)\b",
    flags=re.IGNORECASE,
)
_DELIVERY_TARGET_RE = re.compile(
    r"\b(?:delivery|shipping|shipment|courier|carrier|parcel|order|arriv(?:al|e[sd]?))\b",
    flags=re.IGNORECASE,
)
_DELIVERY_PREDICATE_RE = re.compile(
    r"\b(?:late|early|on time|delay(?:ed)?|slow|quick|fast|lost|never arrived|"
    r"delivered|transit|took\s+\d+\s+days?)\b",
    flags=re.IGNORECASE,
)
_PHYSICAL_DELIVERY_TARGET_RE = re.compile(
    r"\b(?:deliver(?:y|ed|s|ing)?|shipping|shipment|courier|carrier|driver|parcel|"
    r"arriv(?:al|e[sd]?))\b",
    flags=re.IGNORECASE,
)
_DELIVERY_EVALUATION_RE = re.compile(
    r"\b(?:late|early|on time|same[ -]day|next[ -](?:day|morning)|overnight|"
    r"ahead of schedule|before (?:the )?(?:scheduled|"
    r"promised|expected)|after (?:the )?(?:scheduled|promised|expected)|"
    r"later than|earlier than|miss(?:ed|es|ing)?|delay(?:ed|s)?|slow|quick|fast|"
    r"lost|never arrived|damag(?:ed|ing)|rough(?:ly)?|careful(?:ly)?|"
    r"longer than|shorter than|within\s+(?:one|two|three|\d+)\s+days?|"
    r"(?:during|inside|within)\s+(?:the\s+)?promised(?:\s+[a-z0-9-]+){0,3}\s+window|"
    r"promised (?:time|window)|scheduled (?:time|date|window|appointment)|"
    r"(?:no|not|never|missing)\s+(?:delivery\s+)?alert|"
    r"(?:delivery\s+)?alert\s+(?:was|is)\s+not\s+sent|"
    r"excellent|great|good|poor|bad|terrible)\b",
    flags=re.IGNORECASE,
)
_SETUP_TARGET_RE = re.compile(
    r"\b(?:setup|set[ -]?up|installation|install(?:ing|ed)?|initial|first[ -]?time|"
    r"mount(?:ing|ed)?|wall template|bracket|assembl(?:y|ing|ed)|attach(?:ing|ed)?|"
    r"bolt(?:ing|ed)?|quick[ -]?start|account screen|account linking|wizard|"
    r"calibration|channel scan)\b",
    flags=re.IGNORECASE,
)
_SETUP_PREDICATE_RE = re.compile(
    r"\b(?:easy|hard|difficult|confusing|clear|straightforward|simple|vague|"
    r"omit(?:s|ted)?|explain(?:s|ed)?|required?|took|takes?|time|minutes?|"
    r"failed?|reset|completed?|matched?|lined? up|found|without (?:help|intervention)|"
    r"first try|pleasant|unpleasant)\b",
    flags=re.IGNORECASE,
)
_CONNECTIVITY_TARGET_RE = re.compile(
    r"\b(?:wi[ -]?fi|bluetooth|wireless|ethernet|hdmi|connection|connectivity|"
    r"pairing|paired)\b",
    flags=re.IGNORECASE,
)
_POSITIVE_SETUP_RE = re.compile(
    r"\b(?:not\s+(?:hard|difficult|confusing)|easy|straightforward|simple|clear|"
    r"less time|without (?:help|intervention)|first try|lined? up immediately|"
    r"matched?\b.{0,24}\bimmediately|took only\s+(?:\d+|a few)\s+minutes?)\b",
    flags=re.IGNORECASE,
)
_SOFTWARE_TARGET_RE = re.compile(
    r"\b(?:menu|interface|ui|user interface|system|software|app|apps|"
    r"home screen|settings screen|settings panel|options panel|overlay|launcher|guide|"
    r"cursor|operating system|os)\b",
    flags=re.IGNORECASE,
)
_SOFTWARE_PERFORMANCE_PREDICATE_RE = re.compile(
    r"\b(?:slow|sluggish|lag(?:s|ged|ging|gy)?|delay(?:ed|s)?|unresponsive|"
    r"freeze|freezes|froze|frozen|crash(?:es|ed|ing)?|hang(?:s|ing)?|"
    r"lock(?:s|ed|ing)? up|drop(?:s|ped|ping)? back|spin(?:s|ning)?|"
    r"takes?\s+(?:several|a few|\d+)\s+seconds?|load(?:s|ed|ing)?\s+"
    r"(?:quickly|slowly|fast)|open(?:s|ed|ing)?\s+(?:instantly|quickly|slowly)|"
    r"open(?:s|ed|ing)?\s+(?:the\s+)?(?:instant|moment)\b|"
    r"appear(?:s|ed|ing)?\s+(?:instantly|immediately|quickly|slowly|after)|keeps? pace|"
    r"snappy|responsive|smooth|stutter(?:s|ed|ing)?)\b",
    flags=re.IGNORECASE,
)
_SOFTWARE_USABILITY_PREDICATE_RE = re.compile(
    r"\b(?:confusing|intuitive|understandable|hard to understand|"
    r"easy to understand|complicated|straightforward|awkward|clear|"
    r"hard to find|easy to find|cannot find|can't find|could not find|"
    r"buried|difficult to navigate|easy to navigate)\b",
    flags=re.IGNORECASE,
)
_REMOTE_TARGET_RE = re.compile(
    r"\b(?:remote(?: control)?|handset|clicker|directional pad|d[ -]?pad)\b",
    flags=re.IGNORECASE,
)
_REMOTE_PART_RE = re.compile(
    r"\b(?:button|buttons|key|keys|keypad|navigation pad|volume control|"
    r"directional pad|d[ -]?pad|battery cover|battery door|back cover|layout|"
    r"surface|finish)\b",
    flags=re.IGNORECASE,
)
_REMOTE_PREDICATE_RE = re.compile(
    r"\b(?:stick(?:s|ing)?|jam(?:s|med)?|click(?:s|y)?|respond(?:s|ed|ing)?|"
    r"react(?:s|ed|ing)?|immediate|tactile|distinguish(?:able|ed)?(?: by touch)?|"
    r"unresponsive|loose|stiff|tiny|large|clear|confusing|well laid out|"
    r"poorly laid out|easy to press|hard to press|feel(?:s)?\s+(?:cheap|solid|nicer))\b",
    flags=re.IGNORECASE,
)
_IMAGE_TARGET_RE = re.compile(
    r"\b(?:picture|image|display|screen image|colors?|blacks?|contrast|"
    r"brightness|sharpness|hdr|motion|dark scenes?|screen detail|"
    r"skin tones?|faces?)\b",
    flags=re.IGNORECASE,
)
_IMAGE_PREDICATE_RE = re.compile(
    r"\b(?:washed out|vivid|dim|bright|crisp|blurry|sharp|accurate|"
    r"inaccurate|rich|deep|clear|muddy|dull|natural|waxy|lifelike|"
    r"unrealistic|detailed|oversaturated|smear(?:s|ed|ing)?|streak(?:s|ed|ing)?|"
    r"ghost(?:s|ed|ing)?|motion blur|trailing|flicker(?:s|ing)?|smooth)\b",
    flags=re.IGNORECASE,
)
_EXTERIOR_TARGET_RE = re.compile(
    r"\b(?:exterior|cabinet|casing|body|bezel|stand|frame|chassis|"
    r"television|tv|monitor|set|unit)\b",
    flags=re.IGNORECASE,
)
_AESTHETIC_PREDICATE_RE = re.compile(
    r"\b(?:sleek|stylish|elegant|beautiful|handsome|modern|dated|ugly|"
    r"bulky|slim|attractive|unattractive|refined|tasteful|"
    r"looks?\s+(?:good|great|nice))\b",
    flags=re.IGNORECASE,
)
_BUILD_TARGET_RE = re.compile(
    r"\b(?:casing|body|bezel|stand|frame|chassis|panel|housing|shell|grille|"
    r"joint|seam|base|"
    r"product|device|unit|television|tv|monitor)\b",
    flags=re.IGNORECASE,
)
_BUILD_PREDICATE_RE = re.compile(
    r"\b(?:solid|sturdy|flimsy|fragile|wobbl(?:e|es|y)|loose|creak(?:s|y)?|"
    r"crack(?:ed|s)?|scratch(?:ed|es)?|scuff(?:ed|s)?|dent(?:ed|s)?|"
    r"flex(?:es|ed|ing)?|bend(?:s|ing)?|bent|bow(?:s|ed|ing)?|warp(?:s|ed|ing)?|"
    r"rigid|rigidity|separat(?:e|es|ed|ing)|gap(?:s|ped|ping)?|"
    r"well built|poorly built|cheap plastic|premium material)\b",
    flags=re.IGNORECASE,
)
_POWER_CORD_TARGET_RE = re.compile(
    r"\b(?:power cord|power cable|mains lead|ac cord)\b",
    flags=re.IGNORECASE,
)
_CORD_REACH_PREDICATE_RE = re.compile(
    r"\b(?:too short|short|long enough|too long|length|reach(?:es|ed)?|"
    r"does not reach|doesn't reach|outlet|socket)\b",
    flags=re.IGNORECASE,
)
_ARRIVAL_COSMETIC_RE = re.compile(
    r"\b(?:arriv(?:e|ed|al)|delivered)\b.*\b(?:scratch(?:ed|es)?|scuff(?:ed|s)?|"
    r"dent(?:ed|s)?|cosmetic|crack(?:ed|s)?|chipped)\b|"
    r"\b(?:scratch(?:ed|es)?|scuff(?:ed|s)?|dent(?:ed|s)?|cosmetic|"
    r"crack(?:ed|s)?|chipped)\b.*\b(?:on arrival|when (?:it )?arrived)\b",
    flags=re.IGNORECASE,
)
_VAGUE_EXPECTATION_RE = re.compile(
    r"\b(?:met expectations|as (?:I |we |you )?(?:expected|assumed|hoped)|"
    r"as anticipated|"
    r"as advertised|seems? (?:good|great|fine)|"
    r"overall (?:good|great|fine)|happy with (?:it|this)|satisfied with (?:it|this))\b",
    flags=re.IGNORECASE,
)
_CONCRETE_RELIABILITY_CONTEXT_RE = re.compile(
    r"\b(?:powers? on|keeps? working|continues? to work|shuts? down|"
    r"restarts?|reboots?|overheats?|fails?|breaks?|stops? working|"
    r"without (?:a )?(?:fault|failure|crash|shutdown|hiccup)|"
    r"for\s+(?:\d+|one|two|three|several)\s+(?:days?|weeks?|months?|years?)|"
    r"every\s+(?:day|night|morning|evening|time))\b",
    flags=re.IGNORECASE,
)
_VAGUE_RAW_TARGET_RE = re.compile(
    r"^(?:it|this|that|thing|item|product|purchase|overall|everything|"
    r"nothing(?:\s+about\s+(?:it|this|that))?(?:\s+[a-z-]+){0,2}|"
    r"general quality|overall quality|experience)$",
    flags=re.IGNORECASE,
)
_GENERIC_OPINION_RE = re.compile(
    r"\b(?:good|great|excellent|fine|okay|ok|bad|poor|terrible|amazing|"
    r"love|hate|satisfied|disappointed|remarkable|unremarkable|nothing special)\b",
    flags=re.IGNORECASE,
)
_SPECIFIC_ATTRIBUTE_RE = re.compile(
    "|".join(
        pattern.pattern
        for pattern in (
            _PACKAGING_TARGET_RE,
            _DELIVERY_TARGET_RE,
            _SOFTWARE_TARGET_RE,
            _REMOTE_TARGET_RE,
            _REMOTE_PART_RE,
            _IMAGE_TARGET_RE,
            _EXTERIOR_TARGET_RE,
            _BUILD_TARGET_RE,
            _POWER_CORD_TARGET_RE,
        )
    ),
    flags=re.IGNORECASE,
)
_PRICE_VALUE_EXPLICIT_RE = re.compile(
    r"\b(?:value for money|good value|great value|poor value|worth (?:it|"
    r"buying|the purchase|paying|the cost|the price)|not worth|affordable|"
    r"overpriced|underpriced|bargain|good deal|fair price|reasonable price|"
    r"price (?:is|was|seems?|feels?) (?:good|great|fair|reasonable|low|high)|"
    r"too expensive|too costly|costs? too much|bang for (?:the )?buck|"
    r"good buy|poor buy|worthwhile|unreasonable|"
    r"overall deal (?:is|was|feels?|seems?) (?:good|great|poor|bad|fair|"
    r"reasonable|unreasonable|worthwhile|neither\s+[a-z-]+\s+nor\s+[a-z-]+)|"
    r"(?:price|deal)\b.{0,40}\bneither\s+[a-z-]+\s+nor\s+[a-z-]+|"
    r"neither fair nor unfair|"
    r"for what you get|what you get for|saved? (?:money|[$£€¥]?\d+))\b",
    flags=re.IGNORECASE,
)
_LOWER_PRICE_COMPARISON_RE = re.compile(
    r"\b(?:costs?|priced?)\s+(?:much\s+|far\s+|a little\s+)?less\s+than\b|"
    r"\b(?:[a-z-]+\s+){0,2}price\s+(?:is|was|seems?)\s+"
    r"(?:much\s+|far\s+|a little\s+)?lower\s+than\b|"
    r"\b(?:cheaper|less expensive|lower[ -]priced)\s+than\b",
    flags=re.IGNORECASE,
)
_NEUTRAL_PRICE_EVALUATION_RE = re.compile(
    r"\b(?:neither fair nor unfair|neither cheap nor expensive|"
    r"not especially (?:cheap|expensive)|about average for the price)\b|"
    r"\b(?:price|deal)\b.{0,40}\bneither\s+[a-z-]+\s+nor\s+[a-z-]+\b",
    flags=re.IGNORECASE,
)
_CLEAR_NEGATIVE_PERFORMANCE_RE = re.compile(
    r"\b(?:hangs?|hung|crash(?:es|ed|ing)?|freeze|freezes|froze|frozen|"
    r"unresponsive|sluggish|lag(?:s|ged|ging|gy)?)\b",
    flags=re.IGNORECASE,
)
_NEGATED_NEGATIVE_RE = re.compile(
    r"\b(?:not|never)\s+(?:(?:nearly|remotely)\s+)?(?:as\s+)?"
    r"(?:bad|poor|flimsy|fragile|wobbly|loose|slow|"
    r"sluggish|blurry|dull|noisy|distorted|overpriced|confusing|hanging|"
    r"hangs?|crashing|crashes|freezing|freezes)\b|"
    r"\b(?:isn't|isnt|wasn't|wasnt)\s+(?:bad|poor|flimsy|fragile|wobbly|"
    r"loose|slow|sluggish|blurry|dull|noisy|distorted|overpriced|confusing)\b|"
    r"\b(?:doesn't|doesnt|does not)\s+(?:feel\s+)?(?:cheap|flimsy|fragile|"
    r"loose|flex|wobble)\b|"
    r"\bno\s+(?:wobble|wobbling|flex|rattle|rattling|lag|delay|flicker|"
    r"distortion|cracks?|scratches?|loose parts?)\b",
    flags=re.IGNORECASE,
)

_POSITIVE_DELIVERY_TIMING_RE = re.compile(
    r"\b(?:same[ -]day|next[ -](?:day|morning)|overnight|on time|"
    r"ahead of schedule|earlier than (?:promised|expected)|"
    r"exactly when promised|within\s+(?:one|two|three|\d+)\s+days?)\b",
    flags=re.IGNORECASE,
)


class ReviewAnalysisError(ValueError):
    """模型结果不符合单条评论分析契约。"""


@dataclass(frozen=True)
class _InvalidEvidence:
    """An otherwise valid aspect whose quote failed exact source validation."""

    index: int
    raw_aspect: str
    suggested_aspect: str
    aspect: str
    sentiment: str
    rejected_evidence: str

    def repair_record(self) -> dict[str, Any]:
        return {
            "aspect_index": self.index,
            "raw_aspect": self.raw_aspect,
            "aspect": self.aspect,
            "sentiment": self.sentiment,
            "invalid_evidence": self.rejected_evidence,
        }


def _parse_payload(raw_response: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw_response, str):
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ReviewAnalysisError("LLM response is not valid JSON.") from exc
    elif isinstance(raw_response, Mapping):
        payload = deepcopy(dict(raw_response))
    else:
        raise ReviewAnalysisError("LLM response must be a JSON object or JSON string.")

    if not isinstance(payload, dict):
        raise ReviewAnalysisError("LLM response must contain a top-level JSON object.")
    return payload


def _required_text(data: Mapping[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReviewAnalysisError(f"{context}.{key} must be a non-empty string.")
    return value.strip()


def _source_evidence(review_text: str, evidence: str, aspect_index: int) -> str:
    """Return an exact source slice; casing and contiguity remain strict."""

    candidate = evidence.strip()
    start = review_text.find(candidate)
    if start < 0:
        raise ReviewAnalysisError(
            f"aspects[{aspect_index}].evidence was not found in the review text."
        )
    return review_text[start : start + len(candidate)]


def _analysis_fields(
    raw_response: str | Mapping[str, Any],
) -> tuple[str, list[Any]]:
    payload = _parse_payload(raw_response)
    summary = _required_text(payload, "summary", "analysis")
    raw_aspects = payload.get("aspects")
    if not isinstance(raw_aspects, list):
        raise ReviewAnalysisError("aspects must be a list.")
    return summary, raw_aspects


def _aspect_metadata(item: Any, index: int) -> dict[str, str]:
    if not isinstance(item, Mapping):
        raise ReviewAnalysisError(f"aspects[{index}] must be an object.")

    raw_aspect = _required_text(item, "raw_aspect", f"aspects[{index}]")
    suggested_aspect = _required_text(item, "aspect", f"aspects[{index}]")
    sentiment = _required_text(item, "sentiment", f"aspects[{index}]")
    evidence = _required_text(item, "evidence", f"aspects[{index}]")
    if sentiment not in ALLOWED_SENTIMENTS:
        raise ReviewAnalysisError(
            f"aspects[{index}].sentiment must be one of: {', '.join(ALLOWED_SENTIMENTS)}."
        )

    return {
        "raw_aspect": raw_aspect,
        "suggested_aspect": suggested_aspect,
        "aspect": normalize_aspect(raw_aspect, suggested_aspect),
        "sentiment": sentiment,
        "evidence": evidence,
    }


def _has_pair(text: str, target: re.Pattern[str], predicate: re.Pattern[str]) -> bool:
    """Return whether one span contains a specific target and its predicate."""

    return bool(target.search(text) and predicate.search(text))


def _raw_target_has_predicate(
    raw_aspect: str,
    evidence: str,
    target: re.Pattern[str],
    predicate: re.Pattern[str],
) -> bool:
    """Require the candidate's own target, not an unrelated evidence clause.

    The predicate may live in the supporting evidence, but the target must be
    present in ``raw_aspect``. This prevents a broad evidence span containing a
    second opinion from hijacking the current mention's taxonomy label.
    """

    return bool(target.search(raw_aspect) and predicate.search(evidence))


def _remote_anaphor_is_grounded(evidence: str, review_text: str) -> bool:
    """Recognize a remote part only when nearby text names the remote hardware."""

    if not _REMOTE_PREDICATE_RE.search(evidence):
        return False
    if _REMOTE_TARGET_RE.search(evidence):
        return True
    if not _REMOTE_PART_RE.search(evidence):
        return False

    start = review_text.find(evidence)
    if start < 0:
        return False
    nearby_prefix = review_text[max(0, start - 120) : start]
    return bool(
        _REMOTE_TARGET_RE.search(nearby_prefix)
        and re.search(r"\b(?:its|it|those|these)\b", evidence, flags=re.IGNORECASE)
    )


def _contextualized_metadata(
    metadata: Mapping[str, str],
    evidence: str,
    review_text: str,
) -> dict[str, str]:
    """Correct a small set of high-confidence taxonomy boundary errors.

    The rules are evidence-preserving and require a paired target plus
    predicate.  They only relabel a model-produced mention; they never invent a
    missing mention or collapse separate mentions.
    """

    resolved = dict(metadata)
    raw_aspect = metadata["raw_aspect"]
    combined = f"{raw_aspect} {evidence}"

    if _raw_target_has_predicate(
        raw_aspect, evidence, _POWER_CORD_TARGET_RE, _CORD_REACH_PREDICATE_RE
    ):
        resolved["aspect"] = "Other"
    elif _raw_target_has_predicate(
        raw_aspect, evidence, _PACKAGING_TARGET_RE, _PACKAGING_PREDICATE_RE
    ):
        resolved["aspect"] = "Packaging"
    elif _remote_anaphor_is_grounded(evidence, review_text):
        resolved["aspect"] = "Remote Control"
    elif (
        _raw_target_has_predicate(
            raw_aspect, evidence, _SETUP_TARGET_RE, _SETUP_PREDICATE_RE
        )
        and not _CONNECTIVITY_TARGET_RE.search(raw_aspect)
    ):
        resolved["aspect"] = "Setup / Installation"
    elif _raw_target_has_predicate(
        raw_aspect, evidence, _SOFTWARE_TARGET_RE, _SOFTWARE_PERFORMANCE_PREDICATE_RE
    ):
        resolved["aspect"] = "System Performance"
    elif _raw_target_has_predicate(
        raw_aspect, evidence, _SOFTWARE_TARGET_RE, _SOFTWARE_USABILITY_PREDICATE_RE
    ):
        resolved["aspect"] = "Ease of Use"
    elif _raw_target_has_predicate(
        raw_aspect, evidence, _IMAGE_TARGET_RE, _IMAGE_PREDICATE_RE
    ):
        resolved["aspect"] = "Picture Quality"
    elif _raw_target_has_predicate(
        raw_aspect, evidence, _EXTERIOR_TARGET_RE, _AESTHETIC_PREDICATE_RE
    ):
        resolved["aspect"] = "Design"
    elif _raw_target_has_predicate(
        raw_aspect, evidence, _BUILD_TARGET_RE, _BUILD_PREDICATE_RE
    ):
        resolved["aspect"] = "Build Quality"
    elif _raw_target_has_predicate(
        raw_aspect, evidence, _DELIVERY_TARGET_RE, _DELIVERY_PREDICATE_RE
    ):
        resolved["aspect"] = "Delivery"

    # Arrival describes context, not long-term operation.  A clearly cosmetic
    # product-condition claim belongs to construction/finish, never Reliability.
    if resolved["aspect"] == "Reliability" and _ARRIVAL_COSMETIC_RE.search(combined):
        if _has_pair(combined, _BUILD_TARGET_RE, _BUILD_PREDICATE_RE):
            resolved["aspect"] = "Build Quality"
        else:
            resolved["aspect"] = "Other"

    return resolved


def _resolved_sentiment(metadata: Mapping[str, str], evidence: str) -> str:
    sentiment = metadata["sentiment"]
    if _NEGATED_NEGATIVE_RE.search(evidence):
        return "positive"
    if metadata["aspect"] == "Delivery" and _POSITIVE_DELIVERY_TIMING_RE.search(
        evidence
    ):
        return "positive"
    if metadata["aspect"] == "Setup / Installation" and _POSITIVE_SETUP_RE.search(
        evidence
    ):
        return "positive"
    if metadata["aspect"] == "System Performance" and _CLEAR_NEGATIVE_PERFORMANCE_RE.search(
        evidence
    ):
        return "negative"
    if metadata["aspect"] == "Price / Value" and _NEUTRAL_PRICE_EVALUATION_RE.search(
        evidence
    ):
        return "neutral"
    if metadata["aspect"] == "Price / Value" and _LOWER_PRICE_COMPARISON_RE.search(
        evidence
    ):
        return "positive"
    if (
        metadata["aspect"] == "Price / Value"
        and sentiment != "neutral"
        and _FACTUAL_PRICE_MATCH_RE.search(evidence)
        and not _PRICE_EVALUATION_RE.search(evidence)
    ):
        return "neutral"
    return sentiment


def _result_aspect(metadata: Mapping[str, str], evidence: str) -> dict[str, str]:
    return {
        "raw_aspect": metadata["raw_aspect"],
        "aspect": metadata["aspect"],
        "sentiment": _resolved_sentiment(metadata, evidence),
        "evidence": evidence,
    }


def _has_direct_semantic_support(
    metadata: Mapping[str, str],
    evidence: str,
    review_text: str,
) -> bool:
    """Apply narrow precision guards without weakening evidence validation.

    These guards cover taxonomy boundaries that cannot be recovered from a
    suggested label alone.  They never add an Aspect or alter source evidence.
    """

    combined_target = f"{metadata['raw_aspect']} {evidence}"
    if metadata["aspect"] == "Delivery":
        evidence_offset = review_text.find(evidence)
        local_context = (
            _sentence_containing(review_text, evidence_offset)
            if evidence_offset >= 0
            else evidence
        )
        # A delivery mention needs a physical-delivery target in its local
        # sentence and an evaluative predicate in this candidate's own span.
        # This rejects tracking routes, order records, inventory, and parcel
        # placement facts without weakening real timing or complaint evidence.
        if not _PHYSICAL_DELIVERY_TARGET_RE.search(
            f"{metadata['raw_aspect']} {local_context}"
        ) or not _DELIVERY_EVALUATION_RE.search(combined_target):
            return False

    if metadata["aspect"] == "Price / Value":
        # Price / Value is semantic and open-ended, so do not require a finite
        # allowlist of positive phrases. Reject only a high-confidence pure
        # transaction/amount fact that has no evaluative predicate.
        has_evaluation = bool(
            _PRICE_EVALUATION_RE.search(combined_target)
            or _PRICE_VALUE_EXPLICIT_RE.search(combined_target)
            or _LOWER_PRICE_COMPARISON_RE.search(combined_target)
        )
        is_document_fact = bool(
            _TRANSACTION_DOCUMENT_RE.search(combined_target)
            and _TRANSACTION_FACT_PREDICATE_RE.search(combined_target)
        )
        is_bare_amount_fact = bool(
            _MONETARY_AMOUNT_RE.search(combined_target)
            and re.search(
                r"\b(?:paid|pay|costs?|charged|charge|amount|total(?:ed)?)\b",
                combined_target,
                flags=re.IGNORECASE,
            )
        )
        if (is_document_fact or is_bare_amount_fact) and not has_evaluation:
            return False
        if (
            re.search(r"\bpremium\b", combined_target, flags=re.IGNORECASE)
            and not _PRICE_VALUE_SUPPORT_RE.search(combined_target)
            and not _MONETARY_AMOUNT_RE.search(combined_target)
            and not _LOWER_PRICE_COMPARISON_RE.search(combined_target)
        ):
            return False

    if (
        _COMPARISON_TARGET_RE.search(metadata["raw_aspect"])
        and _CONTRAST_RE.search(review_text)
        and _REVIEWED_TARGET_RE.search(review_text)
        and not _REVIEWED_TARGET_RE.search(evidence)
    ):
        return False

    if (
        metadata["sentiment"] == "positive"
        and _MINIMUM_ACCEPTABILITY_RE.search(evidence)
        and _CONTRAST_RE.search(review_text)
        and _NEGATIVE_QUALIFIER_CONTEXT_RE.search(review_text)
    ):
        return False

    if (
        metadata["aspect"] == "Customer Service"
        and _UNUSED_SERVICE_RE.search(combined_target)
    ):
        return False

    if metadata["aspect"] == "Reliability":
        if _ARRIVAL_COSMETIC_RE.search(combined_target):
            return False
        if (
            _VAGUE_EXPECTATION_RE.search(evidence)
            and not _CONCRETE_RELIABILITY_CONTEXT_RE.search(evidence)
        ):
            return False

    if metadata["aspect"] == "Other":
        raw_key = re.sub(r"[^a-z0-9]+", " ", metadata["raw_aspect"].casefold()).strip()
        if _ARRIVAL_COSMETIC_RE.search(combined_target):
            return False
        if _VAGUE_RAW_TARGET_RE.fullmatch(raw_key):
            return False
        if (
            _VAGUE_EXPECTATION_RE.search(evidence)
            and not _SPECIFIC_ATTRIBUTE_RE.search(combined_target)
        ):
            return False

    return True


def _unique_points(aspects: list[dict[str, str]], sentiment: str) -> list[str]:
    """Derive stable, source-backed points without asking the LLM twice."""

    return list(
        dict.fromkeys(
            aspect["evidence"]
            for aspect in aspects
            if aspect["sentiment"] == sentiment
        )
    )


def _deduplicate_exact_mentions(
    aspects: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Drop only byte-for-byte duplicate mentions while preserving source order.

    Repeated normalized Aspects are intentional when their raw opinion target or
    evidence differs.  This deliberately narrow key prevents recall-oriented
    extraction from collapsing distinct judgments into one review-level mention.
    """

    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in aspects:
        key = (
            item["aspect"],
            item["sentiment"],
            item["evidence"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _drop_neutral_polarity_duplicates(
    aspects: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Prefer an explicit polarity over neutral for the exact same judgment.

    This does not merge positive and negative opinions, different evidence, or
    independent repeated mentions. It only removes a neutral duplicate when the
    same normalized Aspect and exact source span already has a polar judgment.
    """

    polar = {
        (item["aspect"], item["evidence"])
        for item in aspects
        if item["sentiment"] in {"positive", "negative"}
    }
    return [
        item
        for item in aspects
        if item["sentiment"] != "neutral"
        or (item["aspect"], item["evidence"]) not in polar
    ]


def _merge_elaborative_mentions(
    aspects: list[dict[str, str]],
    review_text: str,
) -> list[dict[str, str]]:
    """Merge same-opinion elaborations without collapsing independent claims."""

    merged: list[dict[str, str]] = []
    index = 0
    while index < len(aspects):
        current = aspects[index]
        if index + 1 >= len(aspects):
            merged.append(current)
            break

        following = aspects[index + 1]
        left_start = review_text.find(current["evidence"])
        right_start = review_text.find(following["evidence"])
        left_end = left_start + len(current["evidence"])
        same_label = (
            current["aspect"] == following["aspect"]
            and current["sentiment"] == following["sentiment"]
        )
        connector = (
            review_text[left_end:right_start]
            if left_start >= 0 and right_start >= left_end
            else ""
        )
        is_with_elaboration = bool(
            re.fullmatch(r"\s*,?\s*with\s+", connector, flags=re.IGNORECASE)
        )
        is_single_failure_state = (
            current["aspect"] == "Reliability"
            and current["sentiment"] == "negative"
            and connector.strip(" ,").casefold() == "and"
            and (
                (
                    _FAILURE_STATE_RE.search(current["evidence"])
                    and _OPERATION_FAILURE_RE.search(following["evidence"])
                )
                or (
                    _OPERATION_FAILURE_RE.search(current["evidence"])
                    and _FAILURE_STATE_RE.search(following["evidence"])
                )
            )
        )
        if same_label and (is_with_elaboration or is_single_failure_state):
            right_end = right_start + len(following["evidence"])
            merged.append(
                {
                    **current,
                    "evidence": review_text[left_start:right_end],
                }
            )
            index += 2
            continue

        merged.append(current)
        index += 1
    return merged


def _sentence_containing(review_text: str, offset: int) -> str:
    start = max(
        review_text.rfind(".", 0, offset),
        review_text.rfind("!", 0, offset),
        review_text.rfind("?", 0, offset),
        review_text.rfind("\n", 0, offset),
    ) + 1
    while start < len(review_text) and review_text[start].isspace():
        start += 1

    endings = [
        position
        for mark in ".!?\n"
        if (position := review_text.find(mark, offset)) >= 0
    ]
    end = min(endings) + 1 if endings else len(review_text)
    return review_text[start:end].strip()


def _add_explicit_comparative_value(
    aspects: list[dict[str, str]],
    review_text: str,
) -> list[dict[str, str]]:
    """Derive value only from an explicit price-framed feature comparison."""

    if any(item["aspect"] == "Price / Value" for item in aspects):
        return aspects
    frame = _PRICE_FRAME_RE.search(review_text)
    if frame is None:
        return aspects
    sentence = _sentence_containing(review_text, frame.start())
    if not _COMPARISON_LANGUAGE_RE.search(sentence):
        return aspects

    product_aspects = set(PRODUCT_ASPECTS) - {"Price / Value"}
    polarities = {
        item["sentiment"]
        for item in aspects
        if item["aspect"] in product_aspects
        and item["sentiment"] in {"positive", "negative"}
        and item["evidence"] in sentence
    }
    if len(polarities) != 1:
        return aspects

    derived = {
        "raw_aspect": frame.group(0),
        "aspect": "Price / Value",
        "sentiment": polarities.pop(),
        "evidence": sentence,
    }
    positioned = [*aspects, derived]
    return sorted(
        positioned,
        key=lambda item: (
            review_text.find(item["evidence"])
            if item["evidence"] in review_text
            else len(review_text)
        ),
    )


def _build_result(
    summary: str,
    aspects: list[dict[str, str]],
    review_text: str,
) -> dict[str, Any]:
    aspects = _drop_neutral_polarity_duplicates(aspects)
    aspects = _deduplicate_exact_mentions(aspects)
    aspects = _merge_elaborative_mentions(aspects, review_text)
    aspects = _add_explicit_comparative_value(aspects, review_text)
    aspects = _deduplicate_exact_mentions(aspects)
    return {
        "summary": summary,
        "aspects": aspects,
        "positive_points": _unique_points(aspects, "positive"),
        "negative_points": _unique_points(aspects, "negative"),
    }


def validate_analysis_result(
    raw_response: str | Mapping[str, Any],
    review_text: str,
) -> dict[str, Any]:
    """Strictly validate a complete response, including every evidence quote."""

    summary, raw_aspects = _analysis_fields(raw_response)
    aspects: list[dict[str, str]] = []
    for index, item in enumerate(raw_aspects):
        metadata = _aspect_metadata(item, index)
        evidence = _source_evidence(review_text, metadata["evidence"], index)
        metadata = _contextualized_metadata(metadata, evidence, review_text)
        if not _has_direct_semantic_support(metadata, evidence, review_text):
            continue
        aspects.append(_result_aspect(metadata, evidence))
    return _build_result(summary, aspects, review_text)


def _partition_evidence(
    raw_aspects: list[Any],
    review_text: str,
) -> tuple[list[tuple[int, dict[str, str]]], list[_InvalidEvidence]]:
    valid: list[tuple[int, dict[str, str]]] = []
    invalid: list[_InvalidEvidence] = []

    for index, item in enumerate(raw_aspects):
        metadata = _aspect_metadata(item, index)
        try:
            evidence = _source_evidence(review_text, metadata["evidence"], index)
        except ReviewAnalysisError:
            invalid.append(
                _InvalidEvidence(
                    index=index,
                    raw_aspect=metadata["raw_aspect"],
                    suggested_aspect=metadata["suggested_aspect"],
                    aspect=metadata["aspect"],
                    sentiment=metadata["sentiment"],
                    rejected_evidence=metadata["evidence"],
                )
            )
            continue
        metadata = _contextualized_metadata(metadata, evidence, review_text)
        if not _has_direct_semantic_support(metadata, evidence, review_text):
            continue
        valid.append((index, _result_aspect(metadata, evidence)))
    return valid, invalid


def _repair_map(
    raw_response: str | Mapping[str, Any],
    expected_indexes: set[int],
) -> dict[int, str]:
    payload = _parse_payload(raw_response)
    raw_repairs = payload.get("repairs")
    if not isinstance(raw_repairs, list):
        raise ReviewAnalysisError("repairs must be a list.")

    repairs: dict[int, str] = {}
    for position, item in enumerate(raw_repairs):
        if not isinstance(item, Mapping):
            raise ReviewAnalysisError(f"repairs[{position}] must be an object.")
        aspect_index = item.get("aspect_index")
        if isinstance(aspect_index, bool) or not isinstance(aspect_index, int):
            raise ReviewAnalysisError(
                f"repairs[{position}].aspect_index must be an integer."
            )
        if aspect_index not in expected_indexes or aspect_index in repairs:
            continue
        repairs[aspect_index] = _required_text(
            item, "evidence", f"repairs[{position}]"
        )
    return repairs


def _log_dropped_evidence(item: _InvalidEvidence, reason: str) -> None:
    LOGGER.warning(
        "Dropping aspect %d (%s) after one evidence repair attempt: %s",
        item.index,
        item.raw_aspect,
        reason,
    )


def analyze_review(
    review_text: str,
    llm_client: ReviewAnalysisClient | None = None,
) -> dict[str, Any]:
    """Analyze one review, repairing only invalid evidence at most once."""

    if not isinstance(review_text, str) or not review_text.strip():
        raise ReviewAnalysisError("review_text must be a non-empty string.")

    client = llm_client or create_review_client()
    raw_response = client.analyze_review(review_text, STANDARD_ASPECTS)
    summary, raw_aspects = _analysis_fields(raw_response)
    indexed_valid, invalid = _partition_evidence(raw_aspects, review_text)

    if invalid:
        repair_method = getattr(client, "repair_evidence", None)
        if not callable(repair_method):
            for item in invalid:
                _log_dropped_evidence(item, "provider does not support evidence repair")
        else:
            try:
                repair_response = repair_method(
                    review_text,
                    [item.repair_record() for item in invalid],
                )
                repairs = _repair_map(
                    repair_response,
                    {item.index for item in invalid},
                )
            except (LLMClientError, ReviewAnalysisError) as exc:
                for item in invalid:
                    _log_dropped_evidence(item, str(exc))
            else:
                for item in invalid:
                    candidate = repairs.get(item.index)
                    if candidate is None:
                        _log_dropped_evidence(item, "repair response omitted this aspect")
                        continue
                    try:
                        evidence = _source_evidence(review_text, candidate, item.index)
                    except ReviewAnalysisError as exc:
                        _log_dropped_evidence(item, str(exc))
                        continue
                    repaired_metadata = {
                        "raw_aspect": item.raw_aspect,
                        "aspect": item.aspect,
                        "sentiment": item.sentiment,
                    }
                    repaired_metadata = _contextualized_metadata(
                        repaired_metadata,
                        evidence,
                        review_text,
                    )
                    if not _has_direct_semantic_support(
                        repaired_metadata,
                        evidence,
                        review_text,
                    ):
                        continue
                    indexed_valid.append(
                        (item.index, _result_aspect(repaired_metadata, evidence))
                    )

    aspects = [aspect for _, aspect in sorted(indexed_valid, key=lambda pair: pair[0])]
    return _build_result(summary, aspects, review_text)
