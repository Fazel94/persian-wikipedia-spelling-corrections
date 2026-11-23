"""Persian text normalizer, tokenizer and affix joiner.

`join_affixes` repairs affixes that corpus text writes with a space (`می خواهم`, `کتاب ها`), so
counts land on the real word. The affix rules follow hazm's `Normalizer.affix_spacing` and
`token_spacing` (roshan-research/hazm), reimplemented here so the pipeline does not install
hazm's pinned dependency tree.
"""

from __future__ import annotations

import unicodedata

import regex

ZWNJ = "\u200c"
LETTERS = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأإؤئ"
# Letters that never join to the following letter, so a ZWNJ after them is redundant.
NON_JOINING = set("اآأإدذرزژوؤء")

_CHAR_MAP = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "\u00ac": ZWNJ})
_DELETE_RE = regex.compile(
    "[\u064b-\u065f\u0670\u0640\u200d\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]"
)
_ZWNJ_RUN_RE = regex.compile(ZWNJ + "+")
# A ZWNJ is meaningful only between a joining Persian letter and a Persian letter.
_JOINING = "".join(c for c in LETTERS if c not in NON_JOINING)
_BAD_ZWNJ_RE = regex.compile(f"(?<![{_JOINING}]){ZWNJ}|{ZWNJ}(?![{LETTERS}])")

STRIP_CHARS = ".,،؛؟!?:;\"'()[]{}«»…-–—"
_WORD_RE = regex.compile(f"[{LETTERS}]+(?:{ZWNJ}[{LETTERS}]+)*")
MAX_WORD_LEN = 30

PREFIXES = {"می", "نمی"}
SUFFIXES = {
    "ها", "های", "هایی", "هایم", "هایت", "هایش", "هایمان", "هایتان", "هایشان",
    "هام", "هات", "هاش", "هامون", "هاتون", "هاشون",
}
# After a word ending in ه these are always clitics (hazm affix_spacing: `خانه ام` → `خانه‌ام`).
SUFFIXES_AFTER_HE = {"ی", "ای", "ام", "ات", "اش", "ایم", "اید", "اند"}
# Joined only when the joined form is a known word (hazm token_spacing does the same lookup).
GATED_SUFFIXES = {"تر", "ترین", "تری", "ای", "ام", "ات", "اش", "ایم", "اید", "اند", "گر", "گری"}
GATED_PREFIXES = {"بی"}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_CHAR_MAP)
    s = _DELETE_RE.sub("", s)
    s = _ZWNJ_RUN_RE.sub(ZWNJ, s)
    return _BAD_ZWNJ_RE.sub("", s)


def tokenize(line: str) -> list[str | None]:
    """Words of `line`; `None` marks a bigram boundary (punctuation, digits, Latin text)."""
    out: list[str | None] = []
    for chunk in line.split():
        word = chunk.strip(STRIP_CHARS)
        if not (word and len(word) <= MAX_WORD_LEN and _WORD_RE.fullmatch(word)):
            out.append(None)
            continue
        if chunk[0] in STRIP_CHARS:
            out.append(None)
        out.append(word)
        if chunk[-1] in STRIP_CHARS:
            out.append(None)
    return out


def zjoin(a: str, b: str) -> str:
    return a + b if a[-1] in NON_JOINING else a + ZWNJ + b


def join_affixes(tokens: list[str | None], valid: set[str] | None) -> list[str | None]:
    """Merge spaced prefixes/suffixes into their word; gated affixes need `valid` membership."""
    out: list[str | None] = []
    for tok in tokens:
        prev = out[-1] if out else None
        if tok is not None and prev is not None and (
            prev in PREFIXES
            or tok in SUFFIXES
            or (prev[-1] == "ه" and tok in SUFFIXES_AFTER_HE)
            or (
                valid is not None
                and (tok in GATED_SUFFIXES or prev in GATED_PREFIXES)
                and zjoin(prev, tok) in valid
            )
        ):
            out[-1] = zjoin(prev, tok)
            continue
        out.append(tok)
    return out
