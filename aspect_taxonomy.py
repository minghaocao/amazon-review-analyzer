"""Step 4 使用的小型电子产品 / TV Aspect taxonomy。"""

from __future__ import annotations

import re


STANDARD_ASPECTS = (
    "Picture Quality",
    "Sound Quality",
    "Price / Value",
    "Remote Control",
    "System Performance",
    "Ease of Use",
    "Design",
    "Build Quality",
    "Reliability",
    "Connectivity",
    "Setup / Installation",
    "Customer Service",
    "Delivery",
    "Packaging",
    "Other",
)


PRODUCT_ASPECTS = (
    "Picture Quality",
    "Sound Quality",
    "Price / Value",
    "Remote Control",
    "System Performance",
    "Ease of Use",
    "Design",
    "Build Quality",
    "Reliability",
    "Connectivity",
    "Setup / Installation",
)

EXPERIENCE_ASPECTS = (
    "Delivery",
    "Packaging",
    "Customer Service",
)

BUSINESS_CATEGORY_BY_ASPECT = {
    **{aspect: "PRODUCT" for aspect in PRODUCT_ASPECTS},
    **{aspect: "EXPERIENCE" for aspect in EXPERIENCE_ASPECTS},
    "Other": "OTHER",
}


ASPECT_DEFINITIONS: dict[str, str] = {
    "Picture Quality": (
        "the displayed or rendered image itself: picture, colors, contrast, "
        "brightness, sharpness, motion rendering, dark-scene detail, or "
        "washed-out images; not the product's exterior, a screen cover, or a "
        "software/account screen"
    ),
    "Sound Quality": (
        "speaker or audio output: clarity, distortion, bass, loudness, or sound at "
        "a volume level; a volume button on a remote is Remote Control, physical "
        "speaker construction is Build Quality, and a functional alert or "
        "confirmation tone outside media reproduction is Other"
    ),
    "Price / Value": (
        "an evaluative relationship between cost and benefits received: price, "
        "affordability, worth, bargain, deal quality, premium justification, or a "
        "purchase tradeoff, including comparative price; commercial bundle/package "
        "language can express value, "
        "while an amount, receipt, discount, or transaction fact without judgment "
        "and generic product praise are not Price / Value"
    ),
    "Remote Control": (
        "the remote handset and controls physically belonging to it, including "
        "button response, layout, navigation keys, its directional pad, weight, "
        "charging components, and a volume button on the remote; these remain "
        "Remote Control even when appearance or ergonomics are discussed"
    ),
    "Reliability": (
        "physical/product operation and long-term functioning: works reliably or "
        "perfectly, keeps or stops working, breakage, failure, defects, random "
        "shutdowns, and operational stability; 'out of the box' can locate when "
        "operation started or failed and does not itself mean Packaging; not menu "
        "or app speed, feature presence, or a general quality adjective"
    ),
    "System Performance": (
        "software behavior: menu, app, or launcher speed, lag, UI responsiveness, "
        "operating system performance, freezing, crashes, and response time; not "
        "whether controls or navigation are understandable"
    ),
    "Ease of Use": (
        "how understandable, intuitive, simple, confusing, or difficult daily "
        "operation and navigation are; not software response speed or the "
        "difficulty of first-time installation/configuration"
    ),
    "Design": (
        "an explicit aesthetic judgment about visual appearance, styling, shape, "
        "bezel appearance, exterior form, or how the physical product looks; not "
        "image rendering, functional layout, accessory usefulness, cleaning, "
        "packaging, or physical strength"
    ),
    "Build Quality": (
        "physical solidity, mechanical construction, materials, casing, frame, "
        "grille, seams, loose parts, flex, bowing, warping, or craftsmanship of "
        "the product body; a component name does not change a structural judgment "
        "into Sound or Connectivity, package materials are Packaging, and "
        "appearance alone is Design"
    ),
    "Connectivity": (
        "data, network, audio, or video links such as Wi-Fi, Bluetooth, HDMI, "
        "Ethernet, pairing, ports, or maintaining a connection; Bluetooth pairing "
        "is Connectivity even during setup, but mere port/capability presence is "
        "not an opinion, and a power cord or power cable is not Connectivity; "
        "separate evaluated connection methods can be independent mentions"
    ),
    "Setup / Installation": (
        "installation, mounting, assembly, initial setup, first-time configuration, "
        "instructions, and time, difficulty, or effort required to get started; not "
        "routine daily navigation or a connection judgment merely occurring during setup"
    ),
    "Customer Service": (
        "seller/support-agent communication, help, promised replies, troubleshooting, "
        "refunds, replacements, or service handling; messages and follow-up emails "
        "are not physical Delivery"
    ),
    "Delivery": (
        "physical shipping, parcel arrival timing, carrier handling, or the delivery "
        "process, with quick or on-time arrival normally positive; message/email "
        "arrival is not Delivery, and product operation after arrival remains a "
        "product Aspect such as Reliability"
    ),
    "Packaging": (
        "the physical shipping container, packing materials such as a carton, "
        "sleeve, foam, or padding, and their protective performance/condition; a "
        "commercial bundle or total package, included accessories, and an inventory "
        "list are not Packaging, product-body materials are Build Quality, shipping "
        "timing is Delivery, and the idiom 'out of the box' is not Packaging "
        "unless the physical package itself is evaluated"
    ),
    "Other": (
        "a concrete product feature, accessory, maintenance property, indicator, "
        "alert, or behavior that is explicitly evaluated but has no more direct "
        "standard taxonomy label; never a fallback for uncertainty, background "
        "facts, feature presence, or attribute-free overall sentiment"
    ),
}


ASPECT_ALIASES: dict[str, tuple[str, ...]] = {
    "Picture Quality": (
        "picture quality",
        "screen quality",
        "display quality",
        "image quality",
        "color quality",
        "picture",
        "screen",
        "display",
        "image",
        "color",
        "colors",
        "brightness",
        "contrast",
        "sharpness",
        "washed out",
        "dark scenes",
    ),
    "Sound Quality": (
        "sound quality",
        "audio quality",
        "speaker",
        "speakers",
        "sound",
        "audio",
        "bass",
        "volume level",
        "sound volume",
    ),
    "Price / Value": (
        "price value",
        "value for money",
        "for what you get",
        "what you get for the cost",
        "worth the purchase",
        "worth paying",
        "at this price",
        "for the price",
        "worth the price",
        "overpriced",
        "affordable",
        "price",
        "value",
        "cost",
    ),
    "Remote Control": (
        "remote control",
        "remote controller",
        "remote",
        "controller",
        "remote buttons",
        "remote layout",
        "button layout",
        "volume button",
        "volume buttons",
        "navigation button",
        "navigation buttons",
        "directional pad",
        "handset",
    ),
    "System Performance": (
        "system performance",
        "software speed",
        "operating system",
        "interface",
        "user interface",
        "home screen lags",
        "home screen freezes",
        "home screen is slow",
        "menu",
        "app",
        "apps",
        "app launcher",
        "software overlay",
        "program guide",
        "software",
        "lag",
        "lags",
        "sluggish",
        "freezing",
        "freeze",
        "crashes",
        "responds instantly",
        "responds slowly",
        "response time",
        "speed",
        "os",
    ),
    "Ease of Use": (
        "ease of use",
        "usability",
        "easy to use",
        "easy to operate",
        "difficult to operate",
        "not difficult",
        "intuitive interface",
        "confusing menu",
        "difficult navigation",
        "simple navigation",
        "easy",
        "controls",
    ),
    "Design": (
        "appearance",
        "aesthetics",
        "exterior",
        "stylish",
        "slim",
        "modern design",
        "looks beautiful",
        "looks elegant",
        "looks stylish",
        "looks modern",
        "style",
        "design",
    ),
    "Build Quality": (
        "build quality",
        "construction",
        "materials",
        "casing",
        "frame",
        "plastic frame",
        "cheap plastic",
        "loose parts",
        "feels solid",
        "feels premium",
        "solid",
        "build",
    ),
    "Reliability": (
        "functionality",
        "durability",
        "reliability",
        "defective",
        "defective unit",
        "works reliably",
        "works perfectly",
        "worked perfectly",
        "worked reliably",
        "works flawlessly",
        "worked flawlessly",
        "keeps working",
        "stopped working",
        "stop working",
        "would not power on",
        "does not power on",
        "will not power on",
        "failed to power on",
        "stopped turning on",
        "long term functioning",
        "operational stability",
        "random shutdown",
        "shutdown",
        "failure",
        "defect",
        "broke",
        "broken",
    ),
    "Connectivity": (
        "wi fi",
        "wifi",
        "bluetooth",
        "bluetooth pairing",
        "wireless pairing",
        "connection",
        "connectivity",
        "hdmi",
        "ethernet",
        "disconnects",
        "drops connection",
        "paired",
        "pairing",
    ),
    "Setup / Installation": (
        "setup installation",
        "installation",
        "assembly",
        "mounting",
        "initial setup",
        "first time configuration",
        "setup",
        "install",
    ),
    "Customer Service": (
        "customer service",
        "customer support",
        "seller support",
        "support",
        "service",
    ),
    "Delivery": (
        "delivery",
        "shipping",
        "shipment",
        "arrived late",
        "arrived early",
        "arrived on time",
        "arrived damaged",
        "delivery arrival",
    ),
    "Packaging": (
        "packaging",
        "packing",
        "damaged box",
        "box was damaged",
        "secure box",
        "protective foam",
        "well packed",
        "poorly packed",
        "carton",
        "padding",
        "sleeve",
        "protective tape",
        "corner guard",
        "corner guards",
        "cushioning",
    ),
    "Other": ("other",),
}


def _aspect_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


_STANDARD_BY_KEY = {_aspect_key(aspect): aspect for aspect in STANDARD_ASPECTS}
_ALIAS_BY_KEY = {
    _aspect_key(alias): aspect
    for aspect, aliases in ASPECT_ALIASES.items()
    for alias in aliases
}
_ALIASES_BY_LENGTH = tuple(
    sorted(_ALIAS_BY_KEY.items(), key=lambda item: len(item[0]), reverse=True)
)
_STRONG_SINGLE_ALIAS_KEYS = {
    "defective",
    "failure",
    "defect",
    "broke",
    "broken",
    "shutdown",
    "overpriced",
    "underpriced",
}


def _is_strong_alias(alias_key: str) -> bool:
    """Return whether an alias carries more meaning than one incidental word."""

    return len(alias_key.split()) >= 2 or alias_key in _STRONG_SINGLE_ALIAS_KEYS


def is_standard_aspect(aspect: str) -> bool:
    """返回 Aspect 是否属于当前标准 taxonomy。"""

    return aspect in STANDARD_ASPECTS


def business_category_for_aspect(aspect: str) -> str:
    """Return deterministic business metadata without changing the Aspect label."""

    return BUSINESS_CATEGORY_BY_ASPECT.get(aspect, "OTHER")


def normalize_aspect(raw_aspect: str, suggested_aspect: str | None = None) -> str:
    """把模型候选和原始属性词标准化，无法匹配时安全回退到 Other。"""

    raw_key = _aspect_key(raw_aspect) if raw_aspect else ""
    suggested_key = _aspect_key(suggested_aspect) if suggested_aspect else ""

    # A concise exact source attribute remains the strongest deterministic signal.
    # This preserves mappings such as raw ``screen`` even if the model suggests
    # an unrelated category.
    if raw_key in _STANDARD_BY_KEY:
        return _STANDARD_BY_KEY[raw_key]
    if raw_key in _ALIAS_BY_KEY:
        return _ALIAS_BY_KEY[raw_key]

    suggested_exact = _STANDARD_BY_KEY.get(suggested_key) or _ALIAS_BY_KEY.get(
        suggested_key
    )

    # A longer raw phrase can legitimately contain incidental category words
    # (for example ``screen cover`` or ``speaker housing``).  Collect all
    # matches, but let a valid model label outrank a one-word substring.  An
    # exact raw alias and a multi-word semantic alias remain strong enough to
    # correct a bad suggestion, preserving mappings such as ``screen quality``
    # and ``remote volume button``.
    raw_matches: list[str] = []
    strong_raw_matches: list[str] = []
    padded_raw = f" {raw_key} "
    for alias, aspect in _ALIASES_BY_LENGTH:
        if alias and f" {alias} " in padded_raw and aspect not in raw_matches:
            raw_matches.append(aspect)
        if (
            alias
            and _is_strong_alias(alias)
            and f" {alias} " in padded_raw
            and aspect not in strong_raw_matches
        ):
            strong_raw_matches.append(aspect)
    if raw_matches:
        if suggested_exact in raw_matches:
            return suggested_exact
        if strong_raw_matches:
            return strong_raw_matches[0]
        if suggested_exact:
            return suggested_exact
        return raw_matches[0]

    if suggested_exact:
        return suggested_exact

    padded_suggested = f" {suggested_key} "
    for alias, aspect in _ALIASES_BY_LENGTH:
        if alias and f" {alias} " in padded_suggested:
            return aspect

    return "Other"


def taxonomy_prompt_text() -> str:
    """返回供模型提示使用的固定分类列表。"""

    return ", ".join(STANDARD_ASPECTS)


def taxonomy_definition_text() -> str:
    """Return concise guidance for categories that are commonly confused."""

    return "\n".join(
        f"- {aspect}: {definition}."
        for aspect, definition in ASPECT_DEFINITIONS.items()
    )
