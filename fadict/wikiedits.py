"""Mine word-level corrections (typo → fix) from fawiki full edit history.

Streams each MediaWiki Content History 7z part through `7z x -so`, so the ~10 GB of XML per part
never touches disk. For consecutive article revisions, the edit is a candidate when:
- the size changes by at most `MAX_BYTES_DELTA` bytes,
- the editor is not a bot and the summary is not a revert,
- the new text is not a revert to an earlier revision (same sha1), and the edit itself is not
  reverted later in the same part (vandalism and its cleanup would yield reversed pairs).
Each candidate contributes 1:1 word replacements within `MAX_EDIT_DISTANCE` edits, with the
sentence before and after the edit (wiki markup stripped; empty when over `MAX_CONTEXT` chars).

Output per part, in `data/work/wikiedits/`:
- `<part>.tsv`: wrong, right, rev_id, page_id, timestamp, editor (hash of username or IP, only
  for counting distinct editors; never published), editor_type (`user`/`ip`), summary_flag (1 if
  the summary names a spelling fix), before, after.
- `<part>.freq.tsv`: word counts over the last revision of each article in the part, used to
  tell non-words from real words without outside corpora.
Build the published dataset with `python -m fadict.wikiedits release` (see `release`).
"""

from __future__ import annotations

import csv
import difflib
import gzip
import hashlib
import json
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import regex

from .normalize import ZWNJ, normalize_text, tokenize

OUT = Path("data/work/wikiedits")
RELEASE = Path("dist/fawiki_typos")
MAX_BYTES_DELTA = 8
MAX_EDIT_DISTANCE = 2
MAX_CHANGED_LINES = 3
MAX_CONTEXT = 500
SENTENCE_END_RE = regex.compile(r"(?<=[.!?؟])\s+")
MARKUP_RES = [
    (regex.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>|<[^>]+>", regex.S), ""),
    (regex.compile(r"\{\{[^{}]*\}\}"), ""),
    (regex.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]"), r"\1"),
    (regex.compile(r"\[https?://\S+\s*([^\]]*)\]"), r"\1"),
    (regex.compile(r"'{2,}|^[*#:;=]+|=+$", regex.M), ""),
]
BOT_RE = regex.compile(r"bot\b|ربات|بات$", regex.I)
REVERT_RE = regex.compile(r"واگردان|خنثی|برگرداند|undo|undid|revert|\brv\b|rollback", regex.I)
TYPO_RE = regex.compile(r"املا|املایی|غلط|تایپ|اشتباه|نگارش|typo|spell", regex.I)


def damerau(a: str, b: str, cap: int) -> int:
    """Optimal string alignment distance, returning cap + 1 once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = [], list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[-1]


def changed_lines(old: str, new: str) -> tuple[list[str], list[str]] | None:
    """The differing middle block of lines after trimming common leading and trailing lines."""
    a, b = old.split("\n"), new.split("\n")
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    j = 0
    while j < len(a) - i and j < len(b) - i and a[-1 - j] == b[-1 - j]:
        j += 1
    mid_a, mid_b = a[i : len(a) - j], b[i : len(b) - j]
    if len(mid_a) > MAX_CHANGED_LINES or len(mid_b) > MAX_CHANGED_LINES:
        return None
    return mid_a, mid_b


def strip_markup(s: str) -> str:
    for pattern, repl in MARKUP_RES:
        s = pattern.sub(repl, s)
    return " ".join(s.split())


def plain(s: str) -> str:
    s = strip_markup(s)
    return s if len(s) <= MAX_CONTEXT else ""


def word_pairs(old: str, new: str) -> list[tuple[str, str, str, str]]:
    """(wrong, right, sentence before, sentence after) for each 1:1 word fix."""
    block = changed_lines(old, new)
    if block is None:
        return []
    sa = [s for line in block[0] for s in SENTENCE_END_RE.split(line)]
    sb = [s for line in block[1] for s in SENTENCE_END_RE.split(line)]
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=sa, b=sb, autojunk=False).get_opcodes():
        if op != "replace" or i2 - i1 != j2 - j1:
            continue
        for before, after in zip(sa[i1:i2], sb[j1:j2]):
            a = tokenize(normalize_text(before))
            b = tokenize(normalize_text(after))
            sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            for wop, k1, k2, l1, l2 in sm.get_opcodes():
                if wop == "replace" and k2 - k1 == 1 and l2 - l1 == 1:
                    wrong, right = a[k1], b[l1]
                    if wrong and right and damerau(wrong, right, MAX_EDIT_DISTANCE) <= MAX_EDIT_DISTANCE:
                        out.append((wrong, right, plain(before), plain(after)))
    return out


def scan(part: Path) -> Path:
    """Write the candidate pairs of one 7z part; skipped when its output exists."""
    dest = OUT / (part.name.removesuffix(".xml.7z") + ".tsv")
    if dest.exists():
        return dest
    proc = subprocess.Popen(["7z", "x", "-so", str(part)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    tmp = dest.with_suffix(".part")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, delimiter="\t", lineterminator="\n")
        ns = page_id = None
        # Per page: sha1 → revision index, and pending pairs keyed by revision index.
        seen: dict[str, int] = {}
        pending: list[list[tuple]] = []
        prev_text = prev_bytes = None
        freq: Counter = Counter()
        for event, el in ET.iterparse(proc.stdout, events=("end",)):
            tag = el.tag.rpartition("}")[2]
            if tag == "ns":
                ns = el.text
            elif tag == "id" and page_id is None:
                page_id = el.text  # the page's <id> closes before any revision's
            elif tag == "revision":
                text_el = el.find("{*}text")
                text = (text_el.text or "") if text_el is not None else ""
                size = int(text_el.get("bytes", 0)) if text_el is not None else 0
                sha1 = el.findtext("{*}sha1") or ""
                comment = el.findtext("{*}comment") or ""
                user = el.findtext("{*}contributor/{*}username")
                editor = user or el.findtext("{*}contributor/{*}ip") or ""
                idx = len(pending)
                pairs: list[tuple] = []
                if sha1 in seen:  # revert: drop the pairs of every revision it undid
                    for k in range(seen[sha1] + 1, idx):
                        pending[k] = []
                elif (
                    ns == "0"
                    and prev_text is not None
                    and abs(size - prev_bytes) <= MAX_BYTES_DELTA
                    and not REVERT_RE.search(comment)
                    and not (user and BOT_RE.search(user))
                ):
                    meta = (
                        el.findtext("{*}id"),
                        page_id,
                        el.findtext("{*}timestamp"),
                        hashlib.blake2b(editor.encode(), digest_size=6).hexdigest(),
                        "user" if user else "ip",
                        int(bool(TYPO_RE.search(comment))),
                    )
                    pairs = [(w, r, *meta, b, a) for w, r, b, a in word_pairs(prev_text, text)]
                pending.append(pairs)
                seen.setdefault(sha1, idx)
                prev_text, prev_bytes = text, size
                el.clear()
            elif tag == "page":
                for pairs in pending:
                    out.writerows(pairs)
                if ns == "0" and prev_text:
                    freq.update(t for t in tokenize(normalize_text(strip_markup(prev_text))) if t)
                ns = page_id = prev_text = prev_bytes = None
                seen, pending = {}, []
                el.clear()
    if proc.wait() != 0:
        raise RuntimeError(f"7z failed on {part}")
    with dest.with_suffix(".freq.tsv").open("w", encoding="utf-8") as f:
        f.writelines(f"{w}\t{n}\n" for w, n in freq.items())
    tmp.rename(dest)
    return dest


def kind(wrong: str, right: str) -> str:
    if wrong.replace(ZWNJ, "") == right.replace(ZWNJ, ""):
        return "zwnj"
    return "word"


def is_high(p: dict, freq: Counter) -> bool:
    """A non-word spelling fix: confirmed by 2+ editors, rarely undone, and the old form is rare
    in current articles while the new one is common."""
    w, r = p["wrong"], p["right"]
    return (
        p["kind"] == "word"
        and p["editors"] >= 2
        and p["reverse_edits"] * 3 < p["edits"]
        and freq[w] < 100
        and freq[r] >= max(20, 10 * freq[w])
        and (min(len(w), len(r)) > 3 or damerau(w, r, 1) <= 1)
        and w != r + ZWNJ + "ی"  # ezafe style: `درجه‌ی → درجه`
    )


def rows(paths: list[Path]):
    for path in paths:
        with path.open(encoding="utf-8", newline="") as f:
            yield from csv.reader(f, delimiter="\t")


def release(dest: Path) -> None:
    """Write the dataset under `dest`: pairs/{train,test}.tsv (one row per wrong → right),
    edits/{train,test}.jsonl.gz (one row per fix, with sentence context), validation_sample.tsv
    (200 random high-confidence pairs for hand labelling) and stats.json."""
    csv.field_size_limit(sys.maxsize)
    parts = sorted(p for p in OUT.glob("*.tsv") if not p.name.endswith(".freq.tsv"))
    freq: Counter = Counter()
    for path in OUT.glob("*.freq.tsv"):
        for line in path.open(encoding="utf-8"):
            w, n = line.rstrip("\n").split("\t")
            freq[w] += int(n)

    stats: dict = defaultdict(lambda: {"edits": 0, "tagged": 0, "pages": set(), "editors": set(), "years": set()})
    for wrong, right, _rev, page, ts, editor, _etype, flag, _b, _a in rows(parts):
        s = stats[wrong, right]
        s["edits"] += 1
        s["tagged"] += flag == "1"
        s["pages"].add(page)
        s["editors"].add(editor)
        s["years"].add(int(ts[:4]))
    pairs = {}
    for (w, r), s in stats.items():
        p = {
            "wrong": w,
            "right": r,
            "edits": s["edits"],
            "pages": len(s["pages"]),
            "editors": len(s["editors"]),
            "first_year": min(s["years"]),
            "last_year": max(s["years"]),
            "summary_flagged": s["tagged"],
            "reverse_edits": stats[r, w]["edits"] if (r, w) in stats else 0,
            "wiki_freq_wrong": freq[w],
            "wiki_freq_right": freq[r],
            "kind": kind(w, r),
        }
        p["confidence"] = "high" if is_high(p, freq) else "low"
        pairs[w, r] = p
    del stats

    def split(r: str) -> str:
        """~10% test, by the corrected word so no target word appears in both splits."""
        return "test" if hashlib.blake2b(r.encode(), digest_size=1).digest()[0] < 26 else "train"

    (dest / "pairs").mkdir(parents=True, exist_ok=True)
    (dest / "edits").mkdir(parents=True, exist_ok=True)
    order = sorted(pairs.values(), key=lambda p: (-p["edits"], p["wrong"], p["right"]))
    cols = list(order[0])
    for name in ("train", "test"):
        with (dest / "pairs" / f"{name}.tsv").open("w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            f.writelines("\t".join(str(p[c]) for c in cols) + "\n" for p in order if split(p["right"]) == name)

    examples: dict = {}
    files = {name: gzip.open(dest / "edits" / f"{name}.jsonl.gz", "wt", encoding="utf-8") for name in ("train", "test")}
    for wrong, right, rev, page, ts, _editor, etype, flag, before, after in rows(parts):
        p = pairs[wrong, right]
        row = {
            "rev_id": int(rev),
            "page_id": int(page),
            "timestamp": ts,
            "wrong": wrong,
            "right": right,
            "before": before,
            "after": after,
            "editor_type": etype,
            "summary_flagged": flag == "1",
            "kind": p["kind"],
            "confidence": p["confidence"],
        }
        files[split(right)].write(json.dumps(row, ensure_ascii=False) + "\n")
        if p["confidence"] == "high" and before and after:
            examples.setdefault((wrong, right), (before, after, rev))
    for f in files.values():
        f.close()

    rng = random.Random(0)
    sample = rng.sample(sorted(examples), min(200, len(examples)))
    with (dest / "validation_sample.tsv").open("w", encoding="utf-8") as f:
        f.write("wrong\tright\tbefore\tafter\tdiff_url\tlabel\n")
        for w, r in sample:
            before, after, rev = examples[w, r]
            url = f"https://fa.wikipedia.org/w/index.php?diff={rev}"
            f.write(f"{w}\t{r}\t{before}\t{after}\t{url}\t\n")

    summary = {
        "edits": sum(p["edits"] for p in order),
        "pairs": len(order),
        "pairs_by_kind": Counter(p["kind"] for p in order),
        "pairs_high": sum(p["confidence"] == "high" for p in order),
        "pairs_high_test": sum(p["confidence"] == "high" and split(p["right"]) == "test" for p in order),
        "article_tokens": sum(freq.values()),
    }
    (dest / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if argv[:1] == ["release"]:
        release(RELEASE)
        return 0
    parts = sorted((Path(p) for p in argv), key=lambda p: -p.stat().st_size)  # largest first
    with Pool(min(len(parts), os.cpu_count() or 1)) as pool:
        for dest in pool.imap_unordered(scan, parts):
            print(dest, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
