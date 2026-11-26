"""Stage 2 of `error_type`: score edits in their sentence with a masked language model.

`context_dependent` pairs are both words in common use, so the pair alone cannot tell a
real-word spelling error (`بله → بلخ`) from a grammar, style or content edit (`آن → آنها`,
`دو → سه`). Here the changed word is masked in the corrected sentence and the model gives the
log-probability of the old and the new word in that slot (`lm_wrong`, `lm_right`). A word of
several word pieces is scored piece by piece, left to right, with the pieces still to come
masked (PLL-word-l2r, Kauf & Ivanova 2023).

A logistic regression on the scores and the shape of the edit (edit distance, same letters but
for dots, word-piece counts) turns them into `p_spelling`: the chance the edit fixes a real-word
spelling error. It was fitted on the dev labels (`sample`, labelled in Prodigy) and frozen in
`annotation/context_classifier.json` with a threshold of 0.5 before the test sheet
(`test-sample`, drawn from edits outside the dev sheet) was labelled.

    python -m fadict.context score [--limit N] [--types context_dependent,non_word]
    python -m fadict.context sample [N]        # dev sheet, 300
    python -m fadict.context fit               # dev labels → annotation/context_classifier.json
    python -m fadict.context classify          # → data/work/context_pred.tsv
    python -m fadict.context annotate          # p_spelling, sonnet_label → dist/fawiki_typos/edits/
    python -m fadict.context test-sample [N]   # test sheet, 150, excludes the dev sheet
    python -m fadict.context stage1-sample     # stage 1 validation sheet, 250 pairs
    python -m fadict.context variant-sample    # 50 `variant` pairs after the rule fix

Uses CUDA when available. CPU scores about 6 contexts/s on 16 shared cores, a Colab T4 about
220/s, so the 72k contexts of a full release take about 6 minutes there:

    tar czf job.tgz fadict dist/fawiki_typos/edits
    colab new -s ctx --gpu T4; colab upload -s ctx job.tgz /content/job.tgz
    # in the session: cd /content/job && tar xzf ../job.tgz && python -m fadict.context score
    colab download -s ctx /content/job/data/work/context_scores.tsv data/work/context_scores.tsv

`colab` goes through HTTP(S)_PROXY; unset ALL_PROXY (SOCKS), which makes session calls time out.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path

from .normalize import STRIP_CHARS, normalize_text
from .wikiedits import damerau

MODEL = "HooshvareLab/bert-fa-zwnj-base"
EDITS = Path("dist/fawiki_typos/edits")
SCORES = Path("data/work/context_scores.tsv")
SAMPLE = Path("data/work/context_sample.tsv")  # dev: features and guideline were tuned on it
TEST = Path("data/work/context_test.tsv")  # drawn after the classifier was frozen
STAGE1 = Path("data/work/stage1_sample.tsv")
# Pairs per stage 1 class for measuring its precision. `non_word` is drawn from the `low` tier
# only: the `high` tier was checked on 200 pairs. `context_dependent` has its own test sheet;
# `spacing` is mechanical.
STAGE1_SIZES = {"non_word": 100, "variant": 50, "unverified": 50, "disputed": 50}
VARIANT_SAMPLE = Path("data/work/stage1_variant.tsv")  # `variant` re-checked after the rule fix
LABELS = Path("data/work/context_labels.jsonl")  # Prodigy `db-out` of the dev labels
CLASSIFIER = Path("annotation/context_classifier.json")
PREDICTIONS = Path("data/work/context_pred.tsv")
# Claude Sonnet 5.5 labels of every scored edit (guideline v3, batches of 50); see the dataset card.
LLM_LABELS = Path("annotation/context_sonnet_labels.tsv")
THRESHOLD = 0.5  # fixed before the test set was labelled
WINDOW = 32  # word pieces kept on each side of the masked word
BATCH = 64
# Letters that differ only in their dots map to one skeleton (rasm): `زیان`/`زبان` match.
RASM = str.maketrans("بپتثنیئ" "جچحخ" "دذ" "رزژ" "سش" "صض" "طظ" "عغ" "فق" "کگ",
                     "ببببببب" "حححح" "دد" "ررر" "سس" "صص" "طط" "عع" "فف" "کک")
FEATURES = ["lm_diff", "lm_right", "lm_wrong", "pieces_diff", "distance", "same_rasm", "len_diff", "min_len"]


def slot(before: str, after: str, wrong: str, right: str) -> tuple[str, str] | None:
    """The corrected sentence split around the changed word: (text before it, text after it).
    None when the word cannot be located unambiguously."""
    a, b = normalize_text(before).split(), normalize_text(after).split()
    if len(a) != len(b):
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x.strip(STRIP_CHARS) == wrong and y.strip(STRIP_CHARS) == right:
            start = y.find(right)
            left = " ".join([*b[:i], y[:start]])
            rest = " ".join([y[start + len(right):], *b[i + 1:]])
            return left.strip(), rest.strip()
    return None


def edits(types: set[str]):
    for path in sorted(EDITS.glob("*.tsv.gz")):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["error_type"] in types and row["before"] and row["after"]:
                    yield row


class Scorer:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL).eval().to(self.device)

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"] if text else []

    def score(self, items: list[tuple[str, str, str, str]]) -> list[tuple[float, float, int, int]]:
        """(lm_wrong, lm_right, pieces_wrong, pieces_right) for each (left, rest, wrong, right)."""
        flat = self.logprob([(left, rest, word) for left, rest, wrong, right in items for word in (wrong, right)])
        return [(lw, lr, pw, pr) for (lw, pw), (lr, pr) in zip(flat[::2], flat[1::2])]

    def logprob(self, items: list[tuple[str, str, str]]) -> list[tuple[float, int]]:
        """(log-probability, word pieces) of `word` in the slot between `left` and `rest`, for
        each (left, rest, word). Pieces are scored left to right, later ones masked."""
        tok = self.tok
        jobs = []  # (input ids, masked position, target id, item index)
        pieces = []
        for n, (left, rest, word) in enumerate(items):
            lids, rids = self.ids(left)[-WINDOW:], self.ids(rest)[:WINDOW]
            p = self.ids(word)
            pieces.append(len(p))
            for j in range(len(p)):
                seq = [tok.cls_token_id, *lids, *p[:j], *[tok.mask_token_id] * (len(p) - j), *rids, tok.sep_token_id]
                jobs.append((seq, 1 + len(lids) + j, p[j], n))
        jobs.sort(key=lambda j: len(j[0]))
        totals = [0.0] * len(items)
        torch = self.torch
        with torch.inference_mode():
            for k in range(0, len(jobs), BATCH):
                chunk = jobs[k : k + BATCH]
                width = len(chunk[-1][0])
                ids = torch.full((len(chunk), width), tok.pad_token_id)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                for r, (seq, *_rest) in enumerate(chunk):
                    ids[r, : len(seq)] = torch.tensor(seq)
                    mask[r, : len(seq)] = 1
                hidden = self.model.bert(input_ids=ids.to(self.device), attention_mask=mask.to(self.device)).last_hidden_state
                rows = torch.arange(len(chunk))
                pos = torch.tensor([c[1] for c in chunk])
                target = torch.tensor([c[2] for c in chunk])
                # The vocabulary projection only at the masked slot, not at every position.
                logits = self.model.cls(hidden[rows.to(self.device), pos.to(self.device)])
                logp = logits.log_softmax(-1)[rows.to(self.device), target.to(self.device)].tolist()
                for (_s, _p, _t, n), lp in zip(chunk, logp):
                    totals[n] += lp
        return list(zip(totals, pieces))


def score(types: set[str], limit: int | None) -> None:
    """Write `SCORES`: one row per edit of `types` whose word could be located."""
    scorer = Scorer()
    cache: dict[tuple[str, str, str, str], tuple] = {}
    rows, skipped, t0 = [], 0, time.time()
    for row in edits(types):
        s = slot(row["before"], row["after"], row["wrong"], row["right"])
        if s is None:
            skipped += 1
            continue
        rows.append((row, (*s, row["wrong"], row["right"])))
        if limit and len(rows) >= limit:
            break
    todo = sorted({key for _row, key in rows})
    for k in range(0, len(todo), 1024):
        part = todo[k : k + 1024]
        cache.update(zip(part, scorer.score(part)))
        done = k + len(part)
        rate = done / (time.time() - t0)
        print(f"  {done:,}/{len(todo):,} contexts, {rate:.0f}/s", file=sys.stderr, flush=True)
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    with SCORES.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, delimiter="\t", lineterminator="\n")
        out.writerow(["rev_id", "wrong", "right", "error_type", "lm_wrong", "lm_right", "pieces_wrong", "pieces_right"])
        for row, key in rows:
            lw, lr, pw, pr = cache[key]
            out.writerow([row["rev_id"], row["wrong"], row["right"], row["error_type"], f"{lw:.3f}", f"{lr:.3f}", pw, pr])
    print(f"{len(rows):,} edits ({len(todo):,} distinct contexts) scored, {skipped:,} skipped, "
          f"{time.time() - t0:.0f}s", file=sys.stderr)


def features(r: dict) -> list[float]:
    """`FEATURES` of one `SCORES` row."""
    w, rt = r["wrong"], r["right"]
    lw, lr = float(r["lm_wrong"]), float(r["lm_right"])
    return [lr - lw, lr, lw, int(r["pieces_wrong"]) - int(r["pieces_right"]), damerau(w, rt, 3),
            float(w.translate(RASM) == rt.translate(RASM)), abs(len(w) - len(rt)), min(len(w), len(rt))]


def read_scores() -> list[dict]:
    with SCORES.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def fit() -> None:
    """Train the spelling-vs-rest logistic regression on the dev labels and write its
    coefficients to `CLASSIFIER`. Run once; the file is the frozen model."""
    from sklearn.linear_model import LogisticRegression

    gold = {}
    for line in LABELS.open(encoding="utf-8"):
        e = json.loads(line)  # later answers to the same edit replace earlier ones
        if e["answer"] == "accept" and e.get("accept"):
            gold[e["rev_id"], e["wrong"], e["right"]] = e["accept"][0]
    rows = {(r["rev_id"], r["wrong"], r["right"]): r for r in read_scores()}
    keys = [k for k in gold if k in rows]
    model = LogisticRegression(max_iter=1000).fit(
        [features(rows[k]) for k in keys], [gold[k] == "spelling" for k in keys])
    CLASSIFIER.write_text(json.dumps({
        "features": FEATURES, "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
        "threshold": THRESHOLD, "trained_on": str(LABELS), "n": len(keys),
        "n_spelling": sum(gold[k] == "spelling" for k in keys)}, indent=1) + "\n")
    print(f"fitted on {len(keys)} edits → {CLASSIFIER}", file=sys.stderr)


def classify() -> None:
    """Write `PREDICTIONS`: `p_spelling` and `context_spelling` (p ≥ threshold) for every
    scored edit, from the frozen `CLASSIFIER`."""
    m = json.loads(CLASSIFIER.read_text())
    assert m["features"] == FEATURES, "classifier was fitted on other features"
    n = 0
    with PREDICTIONS.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, delimiter="\t", lineterminator="\n")
        out.writerow(["rev_id", "wrong", "right", "p_spelling", "context_spelling"])
        for r in read_scores():
            z = m["intercept"] + sum(c * x for c, x in zip(m["coef"], features(r)))
            p = 1 / (1 + math.exp(-z))
            out.writerow([r["rev_id"], r["wrong"], r["right"], f"{p:.4f}", p >= m["threshold"]])
            n += 1
    print(f"{n:,} edits → {PREDICTIONS}", file=sys.stderr)


def annotate() -> None:
    """Add `p_spelling` (from `PREDICTIONS`) and `sonnet_label` (from `LLM_LABELS`) to the
    released `EDITS` files, in place. Both are empty for edits that were not scored: other error
    types, and the 5% whose word could not be found in its sentence. Safe to rerun."""
    def load(path: Path, column: str) -> dict:
        with path.open(encoding="utf-8", newline="") as f:
            return {(r["rev_id"], r["wrong"], r["right"]): r[column] for r in csv.DictReader(f, delimiter="\t")}

    added = {"p_spelling": load(PREDICTIONS, "p_spelling"), "sonnet_label": load(LLM_LABELS, "sonnet_v3")}
    csv.field_size_limit(sys.maxsize)
    for path in sorted(EDITS.glob("*.tsv.gz")):
        tmp = path.with_suffix(".tmp")
        n, hits = 0, dict.fromkeys(added, 0)
        with gzip.open(path, "rt", encoding="utf-8", newline="") as src, \
                gzip.open(tmp, "wt", encoding="utf-8", newline="") as dst:
            rows = csv.reader(src, delimiter="\t")
            out = csv.writer(dst, delimiter="\t", lineterminator="\n")
            header = next(rows)
            keep = min([header.index(c) for c in added if c in header], default=len(header))
            out.writerow([*header[:keep], *added])
            col = {c: i for i, c in enumerate(header)}
            for row in rows:
                key = (row[col["rev_id"]], row[col["wrong"]], row[col["right"]])
                values = [values_of.get(key, "") for values_of in added.values()]
                out.writerow([*row[:keep], *values])
                n += 1
                for c, v in zip(added, values):
                    hits[c] += v != ""
        tmp.replace(path)
        print(f"{path}: {n:,} edits; " + ", ".join(f"{h:,} with {c}" for c, h in hits.items()), file=sys.stderr)


def sample(n: int, out_path: Path = SAMPLE, seed: int = 0, exclude: Path | None = None) -> None:
    """Write `out_path`: `n` scored `context_dependent` edits for hand labelling, stratified by
    `lm_right - lm_wrong` into 10 equal bins, so the sample is proportional to the population
    and every score range is covered. Edits in `exclude` (an earlier sheet) are skipped. Scores
    are left out so they cannot sway the annotator; join back on rev_id, wrong, right."""
    skip = set()
    if exclude:
        with exclude.open(encoding="utf-8", newline="") as f:
            skip = {(r["rev_id"], r["wrong"], r["right"]) for r in csv.DictReader(f, delimiter="\t")}
    scored = [r for r in read_scores() if r["error_type"] == "context_dependent"]
    scored.sort(key=lambda r: float(r["lm_right"]) - float(r["lm_wrong"]))
    rng = random.Random(seed)
    bins = 10
    picked = []
    for b in range(bins):
        part = [r for r in scored[len(scored) * b // bins : len(scored) * (b + 1) // bins]
                if (r["rev_id"], r["wrong"], r["right"]) not in skip]
        picked += rng.sample(part, min(n // bins, len(part)))
    rng.shuffle(picked)
    keys = {(r["rev_id"], r["wrong"], r["right"]) for r in picked}
    context = {}
    for row in edits({"context_dependent"}):
        key = (row["rev_id"], row["wrong"], row["right"])
        if key in keys:
            context.setdefault(key, (row["before"], row["after"]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, delimiter="\t", lineterminator="\n")
        out.writerow(["rev_id", "wrong", "right", "before", "after", "diff_url", "label"])
        for r in picked:
            before, after = context[r["rev_id"], r["wrong"], r["right"]]
            url = f"https://fa.wikipedia.org/w/index.php?diff={r['rev_id']}"
            out.writerow([r["rev_id"], r["wrong"], r["right"], before, after, url, ""])
    print(f"{len(picked)} edits → {out_path}", file=sys.stderr)


def stage1_sample(sizes: dict[str, int] = STAGE1_SIZES, out_path: Path = STAGE1, seed: int = 2,
                  exclude: tuple[Path, ...] = ()) -> None:
    """Write `out_path`: pairs drawn at random from each class in `sizes`, each with the first of
    its edits whose changed word can be located in the sentence. Pairs in the `exclude` files
    (earlier sheets) are skipped. The `error_type` column is for analysis; `context_recipe.py`
    does not show it to annotators."""
    skip = set()
    for path in exclude:
        with path.open(encoding="utf-8", newline="") as f:
            skip |= {(r["wrong"], r["right"]) for r in csv.DictReader(f, delimiter="\t")}
    first: dict[tuple[str, str], dict] = {}
    for row in edits(set(sizes)):
        key = (row["wrong"], row["right"])
        if key in first or key in skip or (row["error_type"] == "non_word" and row["confidence"] != "low"):
            continue
        if slot(row["before"], row["after"], row["wrong"], row["right"]):
            first[key] = row
    rng = random.Random(seed)
    picked = []
    for kind, n in sizes.items():
        pool = sorted(k for k, r in first.items() if r["error_type"] == kind)
        picked += [first[k] for k in rng.sample(pool, n)]
    rng.shuffle(picked)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, delimiter="\t", lineterminator="\n")
        out.writerow(["rev_id", "wrong", "right", "before", "after", "diff_url", "label", "error_type"])
        for r in picked:
            url = f"https://fa.wikipedia.org/w/index.php?diff={r['rev_id']}"
            out.writerow([r["rev_id"], r["wrong"], r["right"], r["before"], r["after"], url, "", r["error_type"]])
    print(f"{len(picked)} pairs → {out_path}", file=sys.stderr)


def main(argv: list[str]) -> int:
    if argv[:1] == ["score"]:
        limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
        types = set(argv[argv.index("--types") + 1].split(",")) if "--types" in argv else {"context_dependent"}
        score(types, limit)
    elif argv[:1] == ["sample"]:
        sample(int(argv[1]) if len(argv) > 1 else 300)
    elif argv[:1] == ["test-sample"]:
        sample(int(argv[1]) if len(argv) > 1 else 150, TEST, seed=1, exclude=SAMPLE)
    elif argv[:1] == ["stage1-sample"]:
        stage1_sample()
    elif argv[:1] == ["variant-sample"]:
        stage1_sample({"variant": 50}, VARIANT_SAMPLE, seed=3, exclude=(STAGE1,))
    elif argv[:1] == ["fit"]:
        fit()
    elif argv[:1] == ["classify"]:
        classify()
    elif argv[:1] == ["annotate"]:
        annotate()
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
