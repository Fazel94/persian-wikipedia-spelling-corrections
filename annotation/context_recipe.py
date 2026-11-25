"""Prodigy recipes for labelling the stage 2 sheets of `error_type`: the dev sheet
`data/work/context_sample.tsv` and the test sheet `data/work/context_test.tsv`.

    prodigy context.import ctx_labels data/work/context_sample.tsv -F annotation/context_recipe.py
    prodigy context.label ctx_labels data/work/context_sample.tsv -F annotation/context_recipe.py
    prodigy context.label ctx_test data/work/context_test.tsv -F annotation/context_recipe.py
    prodigy context.import-labels ctx_test data/work/context_test.tsv \
        data/work/context_test_sonnet.tsv -c sonnet_v3 -s sonnet -F annotation/context_recipe.py
    # each annotator opens http://localhost:8080/?session=NAME
    prodigy metric.iaa.doc dataset:ctx_test multiclass
    prodigy db-out ctx_labels > data/work/context_labels.jsonl

Run from the project root so `prodigy.json` (RTL, database in `data/work`) is picked up.
Keys 1-6 pick a label; ignore (space) skips an edit that cannot be judged; the flag marks one
to discuss. Ask the questions in order and take the first "yes". Label the error, not the
word: same rules as `annotation/llm_guideline_v3.txt`.
"""

from __future__ import annotations

import csv
import html
import sys

import prodigy
from prodigy.components.db import connect
from prodigy.util import set_hashes

sys.path.insert(0, ".")
from fadict.context import slot  # noqa: E402

LABELS = [
    ("name", "another transliteration, both acceptable (مارچ → مارس). A typo in a name is spelling"),
    ("variant", "old form is an accepted spelling (هیأت → هیئت، اطاق → اتاق، جابجا → جابه‌جا)"),
    ("spelling", "old word makes no sense here; a key, dot or sound away (زیان → زبان، المان → آلمان، انها → آنها)"),
    ("grammar", "same lemma, inflection or agreement fixed (فردی → فرد)"),
    ("style", "wording or register, both correct (میشن → میشوند، سالگی → سال)"),
    ("content", "old sentence made sense but said something else (بودند → شدند)"),
]
# The labels used in the spreadsheet before Prodigy: 0 spelling, 1 grammar, 2 style, 3 content.
SHEET = {"0": "spelling", "1": "grammar", "2": "style", "3": "content"}
WINDOW = 60  # characters of context shown on each side of the edit


def tasks(source: str):
    csv.field_size_limit(sys.maxsize)
    with open(source, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            wrong, right = row["wrong"], row["right"]
            left, rest = slot(row["before"], row["after"], wrong, right) or ("", "")
            ctx_l, ctx_r = html.escape(left[-WINDOW:]), html.escape(rest[:WINDOW])

            def line(tag: str, color: str, word: str) -> str:
                return (
                    f'<tr><td style="color:#888;white-space:nowrap;padding-left:12px">{tag}</td>'
                    f'<td>…{ctx_l} <b style="color:{color};background:#{"fee" if color == "#c00" else "efe"}">'
                    f"{html.escape(word)}</b> {ctx_r}…</td></tr>"
                )

            body = (
                f'<table dir="rtl" style="font-size:20px;line-height:2;text-align:right">'
                f'{line("قبل (غلط؟)", "#c00", wrong)}{line("بعد (اصلاح)", "#080", right)}</table>'
                f'<details dir="rtl" style="font-size:14px;text-align:right"><summary>جملهٔ کامل</summary>'
                f"<p>{html.escape(row['before'])}</p><p>{html.escape(row['after'])}</p></details>"
                f'<a href="{row["diff_url"]}" target="_blank" style="font-size:12px">diff</a>'
            )
            task = {
                "html": body,
                "options": [{"id": i, "text": f"{i}: {t}"} for i, t in LABELS],
                "meta": {"rev_id": row["rev_id"], "edit": f"{wrong} → {right}"},
                "rev_id": row["rev_id"],
                "wrong": wrong,
                "right": right,
            }
            yield set_hashes(task, input_keys=("rev_id", "wrong", "right"), task_keys=("options",))


@prodigy.recipe(
    "context.label",
    dataset=("Dataset to save answers to", "positional", None, str),
    source=("context_sample.tsv", "positional", None, str),
)
def label(dataset: str, source: str):
    return {
        "dataset": dataset,
        "stream": tasks(source),
        "view_id": "choice",
        "config": {
            "choice_style": "single",
            "choice_auto_accept": True,
            "show_flag_button": True,
            "feed_overlap": True,
            "exclude_by": "input",
        },
    }


@prodigy.recipe(
    "context.import",
    dataset=("Dataset to add the sheet labels to", "positional", None, str),
    source=("context_sample.tsv with a filled `label` column", "positional", None, str),
    session=("Annotator name", "option", "s", str),
)
def import_sheet(dataset: str, source: str, session: str = "fazel"):
    """Store the single, unambiguous labels from the sheet as `session`'s answers. Rows with
    several labels, `?` or `x` are left out so they come up again in `context.label`."""
    csv.field_size_limit(sys.maxsize)
    with open(source, encoding="utf-8", newline="") as f:
        raw = list(csv.reader(f, delimiter="\t"))[1:]
    single = {(r[0], r[1], r[2]): SHEET[r[6]] for r in raw if len(r) > 6 and r[6] in SHEET and not any(r[7:])}
    examples = [
        {**task, "accept": [single[key]]}
        for task in tasks(source)
        if (key := (task["rev_id"], task["wrong"], task["right"])) in single
    ]
    store(dataset, session, examples)


@prodigy.recipe(
    "context.import-labels",
    dataset=("Dataset to add the labels to", "positional", None, str),
    source=("The sheet the labels are for", "positional", None, str),
    labels=("TSV with rev_id, wrong, right and a label column", "positional", None, str),
    column=("Label column in `labels`", "option", "c", str),
    session=("Annotator name", "option", "s", str),
)
def import_labels(dataset: str, source: str, labels: str, column: str = "sonnet_v3", session: str = "sonnet"):
    """Store another annotator's labels (e.g. an LLM's) as `session`'s answers, so
    `metric.iaa.doc` and `review` treat it like a human annotator."""
    with open(labels, encoding="utf-8", newline="") as f:
        given = {(r["rev_id"], r["wrong"], r["right"]): r[column] for r in csv.DictReader(f, delimiter="\t")}
    names = {i for i, _ in LABELS}
    examples = [
        {**task, "accept": [given[key]]}
        for task in tasks(source)
        if given.get(key := (task["rev_id"], task["wrong"], task["right"])) in names
    ]
    store(dataset, session, examples)


def store(dataset: str, session: str, examples: list[dict]) -> None:
    annotator = f"{dataset}-{session}"
    examples = [{**e, "answer": "accept", "_session_id": annotator, "_annotator_id": annotator} for e in examples]
    db = connect()
    if dataset not in db.datasets:
        db.add_dataset(dataset)
    # Prodigy skips what a session answered by reading that session's own dataset.
    if annotator not in db.datasets:
        db.add_dataset(annotator, session=True)
    db.add_examples(examples, datasets=[dataset, annotator])
    print(f"{len(examples)} labels stored for {annotator}")
