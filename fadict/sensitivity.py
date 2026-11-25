"""How much the stage 1 `error_type` classes depend on their thresholds.

Each threshold of `fadict.wikiedits` (`KNOWN_FREQ`, `DISPLACE_MIN`, `DISPLACE_RATIO`,
`DISPUTED_SHARE`) is halved and doubled on its own, the others kept at their defaults, and the
released pairs are classified again. Prints a Markdown table: pairs per class, `high` pairs, and
how many of the default `non_word` pairs stay `non_word`.

    python -m fadict.sensitivity
"""

from __future__ import annotations

import csv
import sys
from collections import Counter

from . import wikiedits as w

PARAMS = ("KNOWN_FREQ", "DISPLACE_MIN", "DISPLACE_RATIO", "DISPUTED_SHARE")
INTS = ("edits", "editors", "reverse_edits", "wiki_freq_wrong", "wiki_freq_right")


def load() -> list[dict]:
    pairs = []
    for split in ("train", "test"):
        with (w.RELEASE / "pairs" / f"{split}.tsv").open(encoding="utf-8", newline="") as f:
            for p in csv.DictReader(f, delimiter="\t"):
                for k in INTS:
                    p[k] = int(p[k])
                pairs.append(p)
    return pairs


def classify(pairs: list[dict], freq: Counter) -> tuple[list[str], int]:
    curated, weak = w.known_words({x for p in pairs for x in (p["wrong"], p["right"])}, freq)
    return [w.error_type(p, curated, weak) for p in pairs], sum(w.is_high(p) for p in pairs)


def main() -> int:
    pairs = load()
    freq = Counter()
    for p in pairs:
        freq[p["wrong"]] = p["wiki_freq_wrong"]
        freq[p["right"]] = p["wiki_freq_right"]
    defaults = {k: getattr(w, k) for k in PARAMS}
    settings = [("default", {})] + [
        (f"{k} = {v:g}", {k: v}) for k in PARAMS for v in (defaults[k] / 2, defaults[k] * 2)
    ]
    base = None
    print("| setting | " + " | ".join(w.ERROR_TYPES) + " | high | default non_word kept |")
    print("|---|" + "---:|" * (len(w.ERROR_TYPES) + 2))
    for name, change in settings:
        for k, v in {**defaults, **change}.items():
            setattr(w, k, int(v) if k != "DISPUTED_SHARE" else v)
        types, high = classify(pairs, freq)
        if base is None:
            base = types
        nw = [i for i, t in enumerate(base) if t == "non_word"]
        kept = sum(types[i] == "non_word" for i in nw) / len(nw)
        counts = Counter(types)
        print(f"| {name} | " + " | ".join(f"{counts[t]:,}" for t in w.ERROR_TYPES)
              + f" | {high:,} | {kept:.1%} |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
