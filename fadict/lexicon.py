"""Word-validity sources, in two tiers.

*Strong* sources are curated; a word they accept is never mined as a homophone misspelling:
- Lilak hunspell (`fa-IR.aff/.dic`), the affix-aware spell checker, via `validate`. Its affix
  rules also accept typos (`ماسین`, `هستد`), so a word only hunspell accepts still goes through
  the neighbour typo filter unless it is a Lilak lemma plus a regular ending.
- Lilak lexicon lemmas (`load_lemmas`), which also carry the offensive flag.
- hazm verb conjugations generated from `verbs.dat` / `iverbs.dat` (formal and colloquial
  paradigms).
- UD Persian PerDT treebank word forms (human-annotated text with correct half-spaces, proper
  names included).

The *weak* tier is hazm `words.dat` and `iwords.dat`: the first is corpus-derived, the second
lists colloquial spellings, and both include common typos (`سپاسگذار`, `قذا`, `اقای`, `حتمن`),
so their words still go through homophone-misspelling mining and the neighbour typo filter.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from .normalize import ZWNJ, _WORD_RE, normalize_text, zjoin

RAW = Path("data/raw")
WORK = Path("data/work")
LILAK_DIC = RAW / "fa-IR" / "fa-IR"
CACHE = WORK / "lilak_cache.tsv"


def _clean(word: str) -> str | None:
    w = normalize_text(word.strip())
    return w if _WORD_RE.fullmatch(w) else None


def load_lemmas(path: Path = RAW / "lilak_lexicon") -> dict[str, bool]:
    """Lilak lexicon word → offensive flag (column 3 non-empty)."""
    out: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split(",")
        w = _clean(cols[0])
        if w:
            out[w] = out.get(w, False) or (len(cols) > 2 and cols[2].strip() != "")
    return out


# ---------------------------------------------------------------- hazm


def hazm_words(root: Path = RAW / "hazm") -> set[str]:
    """hazm `words.dat` and the colloquial side of `iwords.dat` (weak tier)."""
    out: set[str] = set()
    for line in (root / "words.dat").read_text(encoding="utf-8").splitlines():
        w = _clean(line.split("\t", 1)[0])
        if w:
            out.add(w)
    for line in (root / "iwords.dat").read_text(encoding="utf-8").splitlines():
        if line.strip():
            w = _clean(line.split()[0])
            if w:
                out.add(w)
    return out


PAST_ENDS = ("م", "ی", "", "یم", "ید", "ند")
PRESENT_ENDS = ("م", "ی", "د", "یم", "ید", "ند")
# Colloquial endings (hazm InformalLemmatizer.iconjugations): 3sg is ه, or د after ا.
CHAT_PAST_ENDS = ("م", "ی", "", "یم", "ین", "ن")
CHAT_PRESENT_ENDS = ("م", "ی", "ه", "یم", "ین", "ن")
PERFECT_CLITICS = ("ام", "ای", "ایم", "اید", "اند")


def _with_prefix(prefix: str, stem: str) -> str:
    """ب/ن + stem with the آ→یا and ا→ی spelling rules (بیا, بینداز, نیامد)."""
    if stem[0] == "آ":
        return prefix + "یا" + stem[1:]
    if stem[0] == "ا":
        return prefix + "ی" + stem[1:]
    return prefix + stem


def _present_forms(stem: str, ends: tuple[str, ...], chat: bool) -> list[str]:
    if not chat and stem[-1] in "اآ":
        stem += "ی"  # آ → آیم, خدا-like stems
    forms = []
    for e in ends:
        if chat and e == "ه" and stem[-1] == "ا":
            e = "د"  # می‌خواد
        forms.append(stem + e)
    return forms


def conjugate(past: str, present: str, chat_present: str | None = None) -> set[str]:
    """Single-token forms of a simple verb (compound verbs are several tokens)."""
    out: set[str] = set()
    if past:
        for ends in (PAST_ENDS, CHAT_PAST_ENDS):
            base = [past + e for e in ends]
            out.update(base)
            out.update(_with_prefix("ن", f) for f in base)
            out.update(zjoin("می", f) for f in base)
            out.update(zjoin("نمی", f) for f in base)
        participle = past + "ه"
        for p in (participle, _with_prefix("ن", participle)):
            out.add(p)
            out.update(p + ZWNJ + c for c in PERFECT_CLITICS)
        out.update({past + "ن", past + "نی", _with_prefix("ن", past + "ن")})
    stems = [(present, PRESENT_ENDS, False), (present, CHAT_PRESENT_ENDS, True)]
    if chat_present:
        stems.append((chat_present, CHAT_PRESENT_ENDS, True))
    for stem, ends, chat in stems:
        base = _present_forms(stem, ends, chat)
        out.update(base)
        out.update(zjoin("می", f) for f in base)
        out.update(zjoin("نمی", f) for f in base)
        for prefix in ("ب", "ن"):
            out.update(f if f.startswith("ب") and prefix == "ب" else _with_prefix(prefix, f) for f in base)
        out.update({_with_prefix("ب", stem), _with_prefix("ن", stem)})  # imperative
        if not chat:
            out.add(stem + "نده")
    return {w for w in out if _WORD_RE.fullmatch(w)}


def hazm_conjugations(root: Path = RAW / "hazm") -> set[str]:
    out: set[str] = set()
    for line in (root / "verbs.dat").read_text(encoding="utf-8").splitlines():
        if "#" in line:
            past, present = (normalize_text(x.strip()) for x in line.split("#", 1))
            if present:
                out |= conjugate(past, present)
    for line in (root / "iverbs.dat").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and "#" in parts[0]:
            past, present = (normalize_text(x) for x in parts[0].split("#", 1))
            out |= conjugate(past, present, normalize_text(parts[1]))
    return out


# ---------------------------------------------------------------- PerDT


def perdt_forms(root: Path = RAW / "perdt") -> set[str]:
    """Word forms of the UD Persian PerDT treebank (proper names included)."""
    forms: set[str] = set()
    for path in sorted(root.glob("*.conllu")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line or line[0] in "#\n":
                    continue
                cols = line.split("\t")
                if "-" in cols[0] or "." in cols[0]:
                    continue
                w = _clean(cols[1])
                if w:
                    forms.add(w)
    return forms


def strong_known_words() -> set[str]:
    """Strong tier minus hunspell: Lilak lemmas, hazm verb conjugations, PerDT forms."""
    return set(load_lemmas()) | hazm_conjugations() | perdt_forms()


# ---------------------------------------------------------------- Lilak hunspell


def _load_cache() -> dict[str, bool]:
    cache: dict[str, bool] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            w, _, ok = line.partition("\t")
            cache[w] = ok == "1"
    return cache


def _hunspell_cli(words: list[str]) -> dict[str, bool]:
    proc = subprocess.run(
        ["hunspell", "-d", str(LILAK_DIC), "-l", "-i", "utf-8"],
        input="\n".join(words),
        capture_output=True,
        text=True,
        check=True,
    )
    bad = set(proc.stdout.split("\n"))
    return {w: w not in bad for w in words}


def _hunspell_spylls(words: list[str]) -> dict[str, bool]:
    from spylls.hunspell import Dictionary

    d = Dictionary.from_files(str(LILAK_DIC))
    out = {}
    for i, w in enumerate(words, 1):
        out[w] = bool(d.lookup(w))
        if i % 10_000 == 0:
            print(f"  lilak: {i:,}/{len(words):,}", file=sys.stderr, flush=True)
    return out


def validate(words: Iterable[str]) -> set[str]:
    """The Lilak-hunspell-valid subset of `words` (cached in data/work/lilak_cache.tsv)."""
    cache = _load_cache()
    todo = sorted({w for w in words if w not in cache})
    if todo:
        print(f"lilak: checking {len(todo):,} new words", file=sys.stderr, flush=True)
        result = _hunspell_cli(todo) if shutil.which("hunspell") else _hunspell_spylls(todo)
        WORK.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a", encoding="utf-8") as f:
            for w in todo:
                f.write(f"{w}\t{int(result[w])}\n")
        cache.update(result)
    return {w for w in words if cache.get(w)}
