"""`p_fix`: the chance that a released `wrong → right` pair is a real spelling fix.

`confidence` (`wikiedits.is_high`) is a hand-written rule. This is a fitted counterpart: a
logistic regression trained on the hand-labelled pairs in `annotation/pair_labels.tsv`. No
pair's `p_fix` reads its own `confidence`. Its inputs (`FEATURES`):

- the pair's edit history, word frequencies in Wikipedia and in the news and subtitle corpora,
  lexicons, and the size of the change;
- where the context stage labelled the pair's edits, the share Claude Sonnet called `spelling`;
- `silver_spelling` and `silver_name`: Sonnet labelled the edits of 28,487 pairs, too few hand
  labels exist to learn which letters people confuse, so a second model (`silver`) learns from
  Sonnet's labels what a spelling fix and a name change look like and scores every pair.

The sheets sample the classes at different rates, so the scores are then recalibrated with
each labelled pair weighted by the released pairs it stands for (`sample_weights`): `p_fix` is
a probability over the release. `fix_tier` is `sure` when `p_fix` is at least `SURE`, else
`unsure`.

    python -m fadict.confidence labels    # Prodigy exports → annotation/pair_labels.tsv
    python -m fadict.confidence annotate  # p_fix, fix_tier → dist/fawiki_typos/pairs/
    python -m fadict.confidence heldout   # p_fix on the context test sheet pairs
    python -m fadict.confidence check     # where `fix_tier` contradicts the edit counts
    python -m fadict.confidence sample    # the next sheet to hand-label → data/work/pair_sample.tsv

Labelling the sheet of `sample` (dataset `pairs2`, open http://localhost:8080/?session=fazel):

    prodigy context.label pairs2 data/work/pair_sample.tsv -F annotation/context_recipe.py
    prodigy db-out pairs2 > data/work/pair_sample_labels.jsonl
    python -m fadict.confidence labels

`scripts/bench_confidence.py` measures it by cross-validation on the hand labels. `annotate`
must be run again after `python -m fadict.wikiedits release`, which rewrites the pairs files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

from . import count
from .context import LLM_LABELS, RASM, STAGE1, VARIANT_SAMPLE, stage1_sample
from .normalize import ZWNJ
from .wikiedits import RELEASE, damerau, known_words, orthographic

LABELS = Path("annotation/pair_labels.tsv")
INTS = ("edits", "pages", "editors", "first_year", "last_year", "summary_flagged", "reverse_edits",
        "wiki_freq_wrong", "wiki_freq_right")
CORPORA = ("formal", "colloquial")  # `fadict.count` unigrams: news crawl, film subtitles
# What is observed about a pair and, after them, what Sonnet said about its edits.
SILVER_INPUTS = ["edits", "pages", "editors", "flagged_share", "reverse_share", "years", "freq_wrong",
                 "freq_right", "freq_ratio", "distance", "len_diff", "min_len", "same_rasm",
                 "orthographic", "zwnj", "wrong_curated", "wrong_weak", "right_curated", "right_weak",
                 "formal_wrong", "formal_right", "formal_ratio",
                 "colloquial_wrong", "colloquial_right", "colloquial_ratio"]
OBSERVED = [*SILVER_INPUTS, "sonnet_labelled", "sonnet_spelling"]
FEATURES = [*OBSERVED, "silver_spelling", "silver_name"]
# `p_fix` from which `fix_tier` is `sure`: as precise as `confidence=high` in cross-validation.
# It was chosen as 0.90 before the recalibration, which moved that score to 0.82.
SURE = 0.82
SILVER_FOLDS = 5
# OpenMP threads for the silver model. With more threads than free cores the fit gets many
# times slower, so this stays well under the core count of a shared machine.
SILVER_THREADS = 4
# Letters pronounced alike in Persian: replacing one with another is the classic misspelling.
HOMOPHONES = ("زذضظ", "سصث", "تط", "حه", "قغ", "اآأإعء", "یئ", "وؤ")
SOUND = {c: i for i, group in enumerate(HOMOPHONES) for c in group}
# Hand-labelled sheets: (name, file, Prodigy session). The context test sheet is not one of
# them, and pairs on it are left out, so it stays unseen.
PRODIGY_SHEETS = (
    ("stage1", Path("data/work/stage1_labels.jsonl"), "stage1-fazel"),
    ("stage1_variant", Path("data/work/stage1_variant_labels.jsonl"), "stage1v-fazel"),
    ("pairs2", Path("data/work/pair_sample_labels.jsonl"), "pairs2-fazel"),  # not labelled yet
)
# The next sheet: the two classes whose few labels leave `p_fix` least tested. `non_word` pairs
# are drawn from `confidence=low`, as on the stage 1 sheet.
SAMPLE = Path("data/work/pair_sample.tsv")
SAMPLE_SIZES = {"disputed": 100, "non_word": 100}
VALIDATION = Path("data/work/validation_sample_labelled.tsv")
CONTEXT_DEV = Path("annotation/context_labels_dev.tsv")
CONTEXT_TEST = Path("annotation/context_labels_test.tsv")

Pair = tuple[str, str]


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_pairs() -> dict[Pair, dict]:
    """Every released pair, keyed by (wrong, right), counts as ints."""
    pairs = {}
    for split in ("train", "test"):
        for p in read_tsv(RELEASE / "pairs" / f"{split}.tsv"):
            for k in INTS:
                p[k] = int(p[k])
            pairs[p["wrong"], p["right"]] = p
    return pairs


def sonnet_votes() -> dict[Pair, Counter]:
    """(wrong, right) → how many of the pair's edits Sonnet gave each label."""
    votes: dict[Pair, Counter] = {}
    for r in read_tsv(LLM_LABELS):
        votes.setdefault((r["wrong"], r["right"]), Counter())[r["sonnet_v3"]] += 1
    return votes


def observed(pairs: list[dict], sonnet: dict[Pair, Counter]) -> list[dict]:
    """The `OBSERVED` features of each pair, by name."""
    freq: Counter = Counter()
    for p in pairs:
        freq[p["wrong"]] = p["wiki_freq_wrong"]
        freq[p["right"]] = p["wiki_freq_right"]
    curated, weak = known_words(set(freq), freq)
    corpora = {c: count.read_counts(count.WORK / f"unigrams_{c}.tsv") for c in CORPORA}
    out = []
    for p in pairs:
        w, r = p["wrong"], p["right"]
        fw, fr = p["wiki_freq_wrong"], p["wiki_freq_right"]
        votes = sonnet.get((w, r))
        row = {
            "edits": math.log1p(p["edits"]), "pages": math.log1p(p["pages"]),
            "editors": math.log1p(p["editors"]), "flagged_share": p["summary_flagged"] / p["edits"],
            # Of the edits between the two forms, the share that went the other way. A share,
            # not `reverse_edits / edits`: that ratio has no upper bound (`به‌طور → بطور`: 182),
            # and the linear fit gave such pairs a `p_fix` near 1.
            "reverse_share": p["reverse_edits"] / (p["edits"] + p["reverse_edits"]),
            "years": p["last_year"] - p["first_year"],
            "freq_wrong": math.log1p(fw), "freq_right": math.log1p(fr),
            "freq_ratio": math.log((fr + 1) / (fw + 1)), "distance": damerau(w, r, 3),
            "len_diff": abs(len(w) - len(r)), "min_len": min(len(w), len(r)),
            "same_rasm": w.translate(RASM) == r.translate(RASM), "orthographic": orthographic(w, r),
            "zwnj": p["kind"] == "zwnj", "wrong_curated": w in curated, "wrong_weak": w in weak,
            "right_curated": r in curated, "right_weak": r in weak,
            "sonnet_labelled": votes is not None,
            "sonnet_spelling": votes["spelling"] / votes.total() if votes else 0.0,
        }
        for c, counts in corpora.items():
            cw, cr = math.log1p(counts[w]), math.log1p(counts[r])
            row |= {f"{c}_wrong": cw, f"{c}_right": cr, f"{c}_ratio": cr - cw}
        out.append(row)
    return out


def change(wrong: str, right: str) -> list[float]:
    """How the two words differ once their common start and end are cut off (`گدرگاه → گذرگاه`
    leaves `د → ذ`): the operation (0 one letter replaced, 1 one added, 2 one removed, 3 two
    swapped, 4 anything else), where it is, whether the two letters sound alike, whether it
    touches the start or the end of the word, how much text went and came, whether a half-space
    is involved, and which letters: the first one removed and added, the first and last of each
    word. Letters are given as code points, for a tree model to split on."""
    limit = min(len(wrong), len(right))
    i = 0
    while i < limit and wrong[i] == right[i]:
        i += 1
    j = 0
    while j < limit - i and wrong[-1 - j] == right[-1 - j]:
        j += 1
    old, new = wrong[i:len(wrong) - j], right[i:len(right) - j]
    shape = (len(old), len(new))
    op = {(1, 1): 0, (0, 1): 1, (1, 0): 2}.get(shape, 3 if shape == (2, 2) and old == new[::-1] else 4)
    return [op, i / len(right), float(op == 0 and SOUND.get(old) == SOUND.get(new, -1)), float(i == 0),
            float(j == 0), len(old), len(new), float(ZWNJ in old or ZWNJ in new),
            ord(old[0]) if old else 0, ord(new[0]) if new else 0,
            ord(wrong[0]), ord(right[0]), ord(wrong[-1]), ord(right[-1])]


def silver(rows: dict[Pair, dict], sonnet: dict[Pair, Counter]) -> dict[Pair, list[float]]:
    """(`silver_spelling`, `silver_name`) for every pair in `rows` (its `OBSERVED` features).

    A gradient-boosted model is fitted on the pairs Sonnet labelled, to predict from
    `SILVER_INPUTS` and `change` which label most of a pair's edits got: `spelling`, `name` or
    anything else. The two scores are its probabilities of the first two. Pairs are dealt into
    `SILVER_FOLDS` folds by their corrected word, and a pair is scored by the model fitted
    without its fold: its score never comes from its own Sonnet label, nor from another pair with
    the same corrected word."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    def label(votes: Counter) -> int:
        return 0 if votes["spelling"] * 2 > votes.total() else 1 if votes["name"] * 2 > votes.total() else 2

    keys = list(rows)
    X = np.array([[*(float(rows[k][f]) for f in SILVER_INPUTS), *change(*k)] for k in keys])
    fold = np.array([hashlib.blake2b(right.encode(), digest_size=1).digest()[0] % SILVER_FOLDS
                     for _wrong, right in keys])
    target = np.array([label(sonnet[k]) if k in sonnet else -1 for k in keys])
    scores = np.zeros((len(keys), 2))
    with threadpool_limits(limits=SILVER_THREADS, user_api="openmp"):
        for n in range(SILVER_FOLDS):
            fit, held = (target >= 0) & (fold != n), fold == n
            if held.any():
                model = HistGradientBoostingClassifier(random_state=0).fit(X[fit], target[fit])
                scores[held] = model.predict_proba(X[held])[:, :2]
    return dict(zip(keys, scores.tolist()))


def featurize(pairs: list[dict]) -> list[list[float]]:
    """`FEATURES` of each pair. Sees no hand labels. The silver model is fitted here on every
    released pair Sonnet labelled, so the result does not depend on which pairs are asked for."""
    sonnet = sonnet_votes()
    released = load_pairs()
    wanted = {(p["wrong"], p["right"]): p for p in [*(released[k] for k in sonnet if k in released), *pairs]}
    rows = dict(zip(wanted, observed(list(wanted.values()), sonnet)))
    scores = silver(rows, sonnet)
    return [[*(float(rows[k][f]) for f in OBSERVED), *scores[k]]
            for k in ((p["wrong"], p["right"]) for p in pairs)]


def sample_weights(pairs: dict[Pair, dict], labelled: list[Pair]) -> list[float]:
    """How many released pairs each labelled pair stands for. One sheet was drawn from the
    `high` pairs and the others from one `error_type` each, so a pair's class is `high` or else
    its `error_type`. This is the only use of `confidence`: it describes how the labels were
    sampled."""
    def drawn_from(p: dict) -> str:
        return "high" if p["confidence"] == "high" else p["error_type"]

    released = Counter(drawn_from(p) for p in pairs.values())
    sampled = Counter(drawn_from(pairs[k]) for k in labelled)
    return [released[c] / sampled[c] for c in (drawn_from(pairs[k]) for k in labelled)]


def train(X: list[list[float]], fix: list[bool], weight: list[float]):
    """Fit fix-vs-rest on `featurize` rows; `weight` is their `sample_weights`. The classifier is
    fitted without the weights: the few labelled `unverified` and `disputed` pairs carry most
    of the weight and the fit follows them. Its scores are then mapped to probabilities by a
    two-parameter logistic fit with the weights, on scores each pair got from a fit that did
    not see it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_predict(classifier, X, fix, cv=folds, method="decision_function")
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(scores.reshape(-1, 1), fix, sample_weight=weight)
    return classifier.fit(X, fix), calibrator


def predict(model, X: list[list[float]]) -> list[float]:
    """`p_fix` for each `featurize` row."""
    classifier, calibrator = model
    return calibrator.predict_proba(classifier.decision_function(X).reshape(-1, 1))[:, 1].tolist()


def majority(rows: list[dict]) -> dict[Pair, bool]:
    """(wrong, right) → is a spelling fix. A pair labelled on several edits is a fix when more
    than half of them are `spelling`."""
    votes: dict[Pair, list[bool]] = {}
    for r in rows:
        votes.setdefault((r["wrong"], r["right"]), []).append(r["label"] == "spelling")
    return {k: sum(v) * 2 > len(v) for k, v in votes.items()}


def read_labels() -> dict[Pair, bool]:
    """The hand labels of `LABELS`, by pair."""
    return majority(read_tsv(LABELS))


def write_labels() -> None:
    """Collect the hand labels of every sheet into `LABELS`, one row per labelled edit."""
    rows = [(r["wrong"], r["right"], "", "spelling" if r["label"] == "fix" else "not_fix", "validation")
            for r in read_tsv(VALIDATION)]
    for name, path, session in PRODIGY_SHEETS:
        if not path.exists():
            print(f"{name}: no labels yet ({path} is missing)", file=sys.stderr)
            continue
        gold = {}
        for line in path.open(encoding="utf-8"):
            e = json.loads(line)  # later answers to the same edit replace earlier ones
            if e["_session_id"] != session:
                continue
            key = (e["wrong"], e["right"], str(e["rev_id"]))
            if e["answer"] == "accept" and e.get("accept"):
                gold[key] = e["accept"][0]
            else:
                gold.pop(key, None)
        rows += [(*key, label, name) for key, label in gold.items()]
    rows += [(r["wrong"], r["right"], r["rev_id"], r["label"], "context_dev") for r in read_tsv(CONTEXT_DEV)]
    held_out = {(r["wrong"], r["right"]) for r in read_tsv(CONTEXT_TEST)}
    kept = [r for r in rows if (r[0], r[1]) not in held_out]
    with LABELS.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, delimiter="\t", lineterminator="\n")
        out.writerow(["wrong", "right", "rev_id", "label", "sheet"])
        out.writerows(kept)
    print(f"{len(kept)} labels, {len({r[:2] for r in kept})} pairs → {LABELS} "
          f"({len(rows) - len(kept)} on the context test sheet left out)", file=sys.stderr)


def add_columns(path: Path, p_fix: dict[Pair, float]) -> Counter:
    """Write `p_fix` and `fix_tier` as the last two columns of the pairs file `path`, replacing
    them when they are already there. Both are empty for `spacing` pairs: none was hand-labelled,
    so the model is untested on them. Returns how many pairs got each tier."""
    with path.open(encoding="utf-8", newline="") as f:
        header, *rows = csv.reader(f, delimiter="\t")
    keep = header.index("p_fix") if "p_fix" in header else len(header)
    wrong, right, error_type = (header.index(c) for c in ("wrong", "right", "error_type"))
    tiers: Counter = Counter()
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, delimiter="\t", lineterminator="\n")
        out.writerow([*header[:keep], "p_fix", "fix_tier"])
        for row in rows:
            p, tier = "", ""
            if row[error_type] != "spacing":
                p = f"{p_fix[row[wrong], row[right]]:.4f}"
                tier = "sure" if float(p) >= SURE else "unsure"
            tiers[tier] += 1
            out.writerow([*row[:keep], p, tier])
    tmp.replace(path)
    return tiers


def annotate() -> None:
    """Fit on all of `LABELS` and add `p_fix` and `fix_tier` to the released pairs files."""
    pairs = load_pairs()
    X = dict(zip(pairs, featurize(list(pairs.values()))))
    labels = read_labels()
    model = train([X[k] for k in labels], list(labels.values()), sample_weights(pairs, list(labels)))
    p_fix = dict(zip(pairs, predict(model, list(X.values()))))
    for path in sorted((RELEASE / "pairs").glob("*.tsv")):
        tiers = add_columns(path, p_fix)
        print(f"{path}: {tiers['sure']:,} sure, {tiers['unsure']:,} unsure, {tiers['']:,} without p_fix",
              file=sys.stderr)


def heldout() -> None:
    """Print how `p_fix` does on the pairs of the context test sheet. `write_labels` keeps them
    out of `LABELS`, so the model fitted here has not seen them. Sonnet's own share of
    `spelling` labels for the pair is printed next to it for comparison."""
    from sklearn.metrics import average_precision_score

    gold = majority([r for r in read_tsv(CONTEXT_TEST) if r["label"] != "ignore"])
    labels = read_labels()
    assert not set(gold) & set(labels), "held-out pairs are among the training labels"
    pairs = load_pairs()
    X = featurize([pairs[k] for k in [*labels, *gold]])
    model = train(X[:len(labels)], list(labels.values()), sample_weights(pairs, list(labels)))
    truth = list(gold.values())
    p_fix = predict(model, X[len(labels):])
    sonnet = [row[FEATURES.index("sonnet_spelling")] for row in X[len(labels):]]
    print(f"{len(truth)} held-out pairs, {sum(truth)} are spelling fixes")
    print(f"average precision: p_fix {average_precision_score(truth, p_fix):.3f}, "
          f"Sonnet share {average_precision_score(truth, sonnet):.3f}")
    for name, picked in ((f"p_fix >= {SURE}", [p >= SURE for p in p_fix]),
                         ("p_fix >= 0.5", [p >= 0.5 for p in p_fix]),
                         ("Sonnet share > 0.5", [s > 0.5 for s in sonnet])):
        hits = sum(t and s for t, s in zip(truth, picked))
        print(f"{name}: {sum(picked)} picked, {hits} are fixes; "
              f"precision {hits / max(sum(picked), 1):.3f}, recall {hits / sum(truth):.3f}")


def sample() -> None:
    """Write `SAMPLE`: `SAMPLE_SIZES` pairs drawn at random per class, none of them on a sheet
    that was already shown to the annotator or held out."""
    stage1_sample(SAMPLE_SIZES, SAMPLE, seed=4, exclude=(LABELS, CONTEXT_TEST, STAGE1, VARIANT_SAMPLE))


def check() -> None:
    """Print how often `fix_tier` in the released pairs files contradicts the edit counts. No
    labels are needed, and the hand labels hold too few `disputed` pairs to show it: run it
    after `annotate` whenever the model changes. Both counts should stay near zero."""
    pairs = load_pairs()
    sure = {k for k, p in pairs.items() if p["fix_tier"] == "sure"}
    both = [k for k in sure if (k[1], k[0]) in sure and k < (k[1], k[0])]
    undone = [k for k in sure if pairs[k]["reverse_edits"] >= 10 * pairs[k]["edits"]]
    print(f"{len(sure):,} sure pairs")
    print(f"sure in both directions: {len(both)}")
    print(f"sure although the reverse edit was made 10 times as often: {len(undone)}")
    for wrong, right in sorted(undone):
        p = pairs[wrong, right]
        print(f"  {wrong} → {right}: {p['edits']} edits, {p['reverse_edits']} the other way, p_fix {p['p_fix']}")


def main(argv: list[str]) -> int:
    commands = {"labels": write_labels, "annotate": annotate, "heldout": heldout, "check": check,
                "sample": sample}
    if len(argv) == 1 and argv[0] in commands:
        commands[argv[0]]()
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
