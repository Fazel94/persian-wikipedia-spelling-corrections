"""Corpus counting (`make count`) and the bigram/context pass used by `build`.

Every corpus line with `line_no % 50 == 0` (per kind) is held out for evaluation and never
counted.
"""

from __future__ import annotations

import gzip
import multiprocessing as mp
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from . import lexicon
from .normalize import GATED_PREFIXES, GATED_SUFFIXES, join_affixes, normalize_text, tokenize, zjoin

RAW = Path("data/raw")
WORK = Path("data/work")
CORPORA = {
    "formal": [
        RAW / "fas_newscrawl_2019_1M-sentences.txt",
        RAW / "fas_newscrawl_2017_1M-sentences.txt",
    ],
    "colloquial": [RAW / "opensubtitles_fa.txt.gz"],
}
HELDOUT_EVERY = 50
BATCH = 20_000
MAX_BIGRAM_ENTRIES = 20_000_000


def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def iter_lines(kind: str) -> Iterator[tuple[bool, str]]:
    """(held_out, sentence) for every line of the corpora of `kind`."""
    n = 0
    for path in CORPORA[kind]:
        with _open(path) as f:
            for line in f:
                # Leipzig lines are `id<TAB>sentence`; OpenSubtitles lines are bare sentences.
                _, tab, rest = line.partition("\t")
                yield n % HELDOUT_EVERY == 0, (rest if tab else line).strip()
                n += 1


def batches(kind: str, heldout_out=None) -> Iterator[list[str]]:
    batch: list[str] = []
    for held, line in iter_lines(kind):
        if held:
            if heldout_out is not None:
                heldout_out.write(line + "\n")
            continue
        batch.append(line)
        if len(batch) == BATCH:
            yield batch
            batch = []
    if batch:
        yield batch


def words(line: str, valid: set[str] | None) -> list[str | None]:
    return join_affixes(tokenize(normalize_text(line)), valid)


def _pool() -> mp.pool.Pool:
    return mp.get_context("fork").Pool(max(1, mp.cpu_count() - 1))


# ---------------------------------------------------------------- pass A


def _count_a(lines: list[str]) -> tuple[Counter, Counter, int]:
    uni: Counter = Counter()
    gated: Counter = Counter()
    ntok = 0
    for line in lines:
        toks = words(line, None)
        prev = None
        for t in toks:
            if t is not None:
                uni[t] += 1
                ntok += 1
                if prev is not None and (t in GATED_SUFFIXES or prev in GATED_PREFIXES):
                    gated[prev, t] += 1
            prev = t
    return uni, gated, ntok


def pass_a(kind: str) -> tuple[Counter, Counter, int]:
    uni: Counter = Counter()
    gated: Counter = Counter()
    ntok = 0
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / f"heldout_{kind}.txt").open("w", encoding="utf-8") as held, _pool() as pool:
        for i, (u, g, n) in enumerate(pool.imap(_count_a, batches(kind, held)), 1):
            uni.update(u)
            gated.update(g)
            ntok += n
            if i % 50 == 0:
                print(f"  {kind}: {i * BATCH:,} lines, {len(uni):,} types", file=sys.stderr, flush=True)
    return uni, gated, ntok


def write_counts(path: Path, counts: Counter) -> None:
    with path.open("w", encoding="utf-8") as f:
        for w, c in counts.most_common():
            if c > 0:
                f.write(f"{w}\t{c}\n")


def read_counts(path: Path) -> Counter:
    out: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            w, _, c = line.rstrip("\n").partition("\t")
            out[w] = int(c)
    return out


def valid_sets(types: set[str]) -> tuple[set[str], set[str]]:
    """(strong, weak) valid subsets of `types`; see `lexicon` for the two tiers."""
    known = lexicon.strong_known_words()
    strong = (types & known) | lexicon.validate(types - known)
    return strong, (types & lexicon.hazm_words()) - strong


def main() -> None:
    unis: dict[str, Counter] = {}
    gateds: dict[str, Counter] = {}
    for kind in CORPORA:
        print(f"pass A: {kind}", file=sys.stderr, flush=True)
        unis[kind], gateds[kind], ntok = pass_a(kind)
        (WORK / f"tokens_{kind}.txt").write_text(f"{ntok}\n")
        with (WORK / f"gated_{kind}.tsv").open("w", encoding="utf-8") as f:
            for (a, b), c in gateds[kind].most_common():
                f.write(f"{a}\t{b}\t{c}\n")

    # Resolve gated joins: a spaced pair becomes one word only when the joined form is valid.
    check = {w for u in unis.values() for w, c in u.items() if c >= 2}
    check |= {zjoin(a, b) for g in gateds.values() for (a, b), c in g.items() if c >= 2}
    valid = set().union(*valid_sets(check))
    for kind in CORPORA:
        uni = unis[kind]
        for (a, b), c in gateds[kind].items():
            j = zjoin(a, b)
            if j in valid:
                uni[j] += c
                uni[a] -= c
                uni[b] -= c
        write_counts(WORK / f"unigrams_{kind}.tsv", uni)
        print(f"{kind}: {len(uni):,} types", file=sys.stderr)


# ---------------------------------------------------------------- pass B (called by build)

_VOCAB: set[str] = set()
_VALID: set[str] = set()
_CANDIDATES: set[str] = set()
CONTEXTS_PER_TYPE = 5


def _count_b(lines: list[str]) -> tuple[Counter, dict[str, list[str]]]:
    bi: Counter = Counter()
    ctx: dict[str, list[str]] = {}
    for line in lines:
        toks = words(line, _VALID)
        prev = None
        for t in toks:
            if t is not None:
                if prev is not None and prev in _VOCAB and t in _VOCAB:
                    bi[prev, t] += 1
                if t in _CANDIDATES:
                    lst = ctx.setdefault(t, [])
                    if len(lst) < CONTEXTS_PER_TYPE:
                        lst.append(normalize_text(line))
            prev = t
    return bi, ctx


def pass_b(vocab: set[str], valid: set[str], candidates: set[str]) -> tuple[Counter, dict[str, list[str]]]:
    """Bigram counts over `vocab` and up to CONTEXTS_PER_TYPE context lines per `candidates` type."""
    global _VOCAB, _VALID, _CANDIDATES
    _VOCAB, _VALID, _CANDIDATES = vocab, valid, candidates
    bi: Counter = Counter()
    ctx: dict[str, list[str]] = {}
    for kind in CORPORA:
        print(f"pass B: {kind}", file=sys.stderr, flush=True)
        with _pool() as pool:
            for i, (b, c) in enumerate(pool.imap(_count_b, batches(kind)), 1):
                bi.update(b)
                for w, lines in c.items():
                    lst = ctx.setdefault(w, [])
                    lst.extend(lines[: CONTEXTS_PER_TYPE - len(lst)])
                if len(bi) > MAX_BIGRAM_ENTRIES:
                    bi = Counter({k: v for k, v in bi.items() if v > 1})
                if i % 50 == 0:
                    print(f"  {kind}: {i * BATCH:,} lines, {len(bi):,} pairs", file=sys.stderr, flush=True)
    return bi, ctx


if __name__ == "__main__":
    main()
