---
license: cc-by-sa-4.0
language:
- fa
pretty_name: Persian (Farsi) Wikipedia Spelling Corrections
task_categories:
- text-generation
tags:
- spelling-correction
- spell-check
- typos
- misspellings
- persian
- farsi
- wikipedia
- edit-history
size_categories:
- 100K<n<1M
configs:
- config_name: pairs
  default: true
  data_files:
  - split: train
    path: pairs/train.tsv
  - split: test
    path: pairs/test.tsv
  sep: "\t"
- config_name: edits
  data_files:
  - split: train
    path: edits/train.tsv.gz
  - split: test
    path: edits/test.tsv.gz
  sep: "\t"
---

# Persian (Farsi) Wikipedia Spelling Corrections

If you have ever fixed `برگذار` to `برگزار` on Persian Wikipedia, this dataset is partly yours.
It collects the spelling fixes that Persian Wikipedia editors have made since the site began,
taken from the full edit history: 345,318 single-word replacements covering 156,732 distinct
pairs, each with the sentence before and after the edit. Not every replacement is a spelling
fix; `error_type` separates the 66,762 non-word spelling fixes from grammar, style, content and
naming edits.

The errors come from people typing, so they look like real Persian typos: letters with the
same sound swapped (`حاظر → حاضر`), dots dropped or misplaced (`همجنین → همچنین`), letters
missed (`تلوزیون → تلویزیون`). 12,982 pairs are marked `confidence=high`. On a random sample
of 200 of them checked by hand, 197 were real spelling fixes (98.5%).

| typo | fix | edits | editors |
|---|---|---:|---:|
| `برگذار` | `برگزار` | 461 | 135 |
| `حاظر` | `حاضر` | 352 | 137 |
| `تلوزیون` | `تلویزیون` | 272 | 92 |
| `تاتر` | `تئاتر` | 123 | 93 |
| `همجنین` | `همچنین` | 86 | 70 |

## Loading the data

```python
from datasets import load_dataset

pairs = load_dataset("Phazel/persian-wikipedia-spelling-corrections", "pairs")
edits = load_dataset("Phazel/persian-wikipedia-spelling-corrections", "edits")
test = pairs["test"].filter(lambda r: r["confidence"] == "high")  # 1,313 pairs
```

The same files are on [GitHub](https://github.com/Fazel94/persian-wikipedia-spelling-corrections).

| config | files | one row per |
|---|---|---|
| `pairs` (default) | `pairs/{train,test}.tsv` | misspelling and its fix, with counts |
| `edits` | `edits/{train,test}.tsv.gz` | one fix in one revision, with its sentence |

`validation_sample.tsv` holds the 200 hand-labelled pairs and `stats.json` the totals.

About 10% of pairs are in `test`. The split is decided by a hash of the corrected word, so a
word that appears as a fix in `test` never appears as a fix in `train`. Each edit goes to the
same split as its pair.

### A record from `edits`

Shown as JSON; the file is gzipped TSV with these columns.

```json
{
  "rev_id": 5820417,
  "page_id": 138284,
  "timestamp": "2011-11-08T10:07:42Z",
  "wrong": "برگذار",
  "right": "برگزار",
  "before": "در ۱۲ می ۲۰۰۰، دادگاه عالی پاکستان به مشرف دستور داد تا در ۱۲ اکتبر ۲۰۰۲ انتخابات سراسری را برگذار نماید.",
  "after": "در ۱۲ می ۲۰۰۰، دادگاه عالی پاکستان به مشرف دستور داد تا در ۱۲ اکتبر ۲۰۰۲ انتخابات سراسری را برگزار نماید.",
  "editor_type": "user",
  "summary_flagged": true,
  "kind": "word",
  "confidence": "high",
  "error_type": "non_word",
  "p_spelling": null,
  "sonnet_label": null
}
```

* `rev_id`: the revision that made the fix. Open `https://fa.wikipedia.org/w/index.php?diff=<rev_id>` to see it.
* `page_id`, `timestamp`: the article and the time of the edit.
* `wrong`, `right`: the word before and after, normalized (see below).
* `before`, `after`: the edited sentence with wiki markup removed. Empty when the sentence is longer than 500 characters (2% of edits).
* `editor_type`: `user` for a registered account, `ip` for an anonymous editor.
* `summary_flagged`: the edit summary mentions spelling (`املا`, `غلط`, `تایپ`, `typo` and similar).
* `kind`: `zwnj` when the two words differ only by the half-space (ZWNJ), otherwise `word`.
* `confidence`: `high` or `low`, copied from the pair.
* `error_type`: what kind of edit the pair is (see [Error type](#error-type)), copied from the pair.
* `p_spelling`: for `context_dependent` edits, the estimated probability that this edit fixes a
  real-word spelling error in its sentence (see [Real-word errors in context](#real-word-errors-in-context)).
  Empty for other error types and for the 5% of `context_dependent` edits whose word could not
  be located in its sentence.
* `sonnet_label`: for the same edits, the label Claude Sonnet 5.5 gave from the sentence: one of
  `spelling`, `variant`, `grammar`, `style`, `content`, `name` (see
  [Real-word errors in context](#real-word-errors-in-context)).

### Columns in `pairs`

* `wrong`, `right`, `kind`, `confidence`, `error_type`: as above.
* `edits`: how many revisions made this replacement.
* `pages`, `editors`: how many different articles and editors (accounts or IP addresses) made it.
* `first_year`, `last_year`: when it was first and last made.
* `summary_flagged`: how many of those edits had a summary mentioning spelling.
* `reverse_edits`: how many edits went the other way, replacing `right` with `wrong`.
* `wiki_freq_wrong`, `wiki_freq_right`: how often each word occurs in the current text of all articles.
* `p_fix`: the estimated probability, from 0 to 1, that the pair fixes a spelling error (see
  [Fitted confidence](#fitted-confidence)). Empty for `spacing` pairs.
* `fix_tier`: `sure` when `p_fix` is at least 0.82, otherwise `unsure`. Empty for `spacing` pairs.

## How it was built

The source is the MediaWiki Content History export of Persian Wikipedia, snapshot 2026-11-01
(`dumps.wikimedia.org/other/mediawiki_content_history/fawiki/2026-11-01/`): 54 files, 5.5 GB
compressed. Every revision of every article is read as a stream; the uncompressed XML is never
written to disk.

Each edit is compared with the revision before it. It is kept as a candidate when the page
size changed by 8 bytes or less, the editor's name does not mark a bot (`bot`, `ربات`), and
the summary does not describe a revert (`واگردانی`, `خنثی`, `undo`, `revert`, `rv` and
similar). Reverts are also detected by content: an edit whose text has the same SHA-1 as an
earlier revision is a revert and is skipped, and so is every edit it undid. Without this,
vandalism followed by its cleanup would show up as a pair in the wrong direction. Edits that
change more than 3 lines are skipped.

The changed lines are split into sentences and the old and new sentences are aligned. Wiki
markup is stripped from each: references and templates are removed, category, file and
interlanguage links are dropped (editing them renames a category or file, as in
`[[رده:یادگیری ماشینی]] → [[رده:یادگیری ماشین]]`), and other links are reduced to the text
shown. Each aligned pair is then split into words. Every place where one word was replaced by
exactly one other word within a Damerau-Levenshtein distance of 2 becomes a fix.

Both words are normalized the same way: Unicode NFKC; Arabic `ي`, `ى`, `ك`, `ة` and `ۀ` mapped
to Persian `ی`, `ی`, `ک`, `ه` and `ه`; diacritics, tatweel and bidi marks removed; runs of ZWNJ
collapsed to one, and a ZWNJ that cannot change how letters join dropped. Edits that only
change these characters are therefore not in the dataset. The `before` and `after` sentences
keep their original characters.

### Confidence

A pair is `high` when it looks like a misspelling that is not itself a word. All of these must
hold:

* at least 2 editors made it, which removes one person renaming something across many pages;
* `reverse_edits` is less than a third of `edits`, which removes style disputes such as `او ↔ وی`;
* `kind` is `word`;
* the misspelling occurs fewer than 100 times in current articles, and the fix at least 20 times and at least 10 times as often, which removes edits between two real words such as `یک → دو`;
* when either word has 3 letters or fewer, the two differ by a single edit;
* it is not an ezafe style change such as `درجه‌ی → درجه`.

Everything else is `low`. The `low` pairs still contain plenty of real errors, including
real-word errors where a valid word was used in place of another. Those need the sentence to
judge, which is what the `edits` config is for.

### Fitted confidence

`confidence` is a fixed rule. `p_fix` estimates the same thing with a model fitted to hand
labels: the probability that the pair fixes a spelling error. No pair's `p_fix` is computed
from its `confidence`. A logistic regression computes it from:

* the counts above: `edits`, `pages`, `editors`, the share of edits with a flagged summary, the
  share of reverse edits, the years between the first and last edit, and both frequencies;
* how often each word occurs in a news corpus and in film subtitles;
* whether each word is in a curated lexicon or only in a weaker one;
* the size of the change: edit distance, word lengths, whether only dots differ, whether only
  the hamza seat or a joined `به` differs;
* for pairs whose edits Claude Sonnet 5.5 labelled (see
  [Real-word errors in context](#real-word-errors-in-context)), the share labelled `spelling`;
* two scores from a second model. Sonnet's labels cover 28,487 pairs. A gradient-boosted model
  learned from them, using the inputs above plus which letters changed and where, what a
  spelling fix and what a name change look like, and then scored every pair. Each pair is
  scored by a copy of that model that saw neither its own Sonnet label nor any pair with the
  same corrected word.

The logistic regression was fitted on 726 hand-labelled pairs: the samples described under
[Validation](#validation) and the 299 development edits of the context stage. Those samples
took a larger share of some classes than of others, so its scores were then recalibrated
with every labelled pair weighted by the number of pairs in the dataset it stands for. The
weights are the only place where `confidence` enters: one sample was drawn from the `high`
pairs.

`fix_tier = sure` marks `p_fix ≥ 0.82`, the point where the tier is about as precise as
`confidence = high`. 54,251 pairs are `sure` (46,747 `non_word`, 6,362 `context_dependent`,
1,142 `disputed`), against 12,982 `high` pairs; 10,268 pairs are both. `spacing` pairs have
neither column, because none was hand-labelled.

### Error type

Not every one-word edit is a spelling fix. Editors also change grammar (`آن → آنها`), style
(`مسئولین → مسئولان`, `بین → میان`) and facts (`دو → سه`). Whether such an edit fixes a
misspelled word depends on its sentence. `error_type` sorts each pair using the two words,
Persian lexicons, how often each word occurs in current articles, and the edit counts, and
marks the pairs where that is not enough.

Two kinds of evidence decide whether the old form is a word. *Curated* lexicons (Lilak lemmas,
hazm verb conjugations, PerDT word forms) are trusted. *Weak* evidence (hazm `words.dat`, Lilak
hunspell, or 100+ uses in current articles) also accepts common misspellings such as `برگذار`
and `مذبور`. Frequency settles those: a pair is *displaced* when the old form occurs fewer than
100 times in current articles and the new one at least 20 times and 10 times as often, which
means editors have all but replaced one with the other. The first rule that matches wins:

| `error_type` | rule | examples | pairs | edits |
|---|---|---|---:|---:|
| `disputed` | `reverse_edits` is at least half of `edits`: editors disagree | `او ↔ وی`, `از ↔ در`, `وبگاه ↔ وب‌گاه` | 19,347 | 63,072 |
| `spacing` | `kind` is `zwnj` | `میشود → می‌شود` | 8,569 | 29,151 |
| `unverified` | the fix has no evidence either (rare, in no lexicon) | names, transliterations | 31,095 | 36,879 |
| `variant` | the old form is a word and differs from the fix only in hamza seat or a joined `به` (`orthographic`) | `موثر → مؤثر`, `بطور → به‌طور` | 728 | 18,245 |
| `context_dependent` | the old form is a word, so only the sentence can tell | `آن → آنها`, `جهن → جهان`, `اطاق → اتاق`, `دو → سه` | 30,231 | 87,103 |
| `non_word` | the old form is in no lexicon, or only weak evidence accepts it and it is displaced | `برگذار → برگزار`, `گدرگاه → گذرگاه` | 66,762 | 110,868 |

`non_word` pairs are spelling errors in any sentence. `variant` pairs replace a dispreferred
but accepted spelling of the same word, by hamza or joining convention. Whether to count them as
spelling depends on the use. The `orthographic` test folds `أ`, `إ` to `ا`, `ؤ` to `و`, `ئ` to
`ی` or nothing, drops `ء` and ZWNJ, and accepts `ب` joined to the next word for `به‌`. It applies
only when the old form is a word; a non-word such as `هئیت → هیئت` stays `non_word`. Dropping `ء`
also merges a few distinct words, such as `جز → جزء` (37 edits).

`context_dependent` pairs mix real-word spelling errors (`بله → بلخ`, `جهن → جهان`) with
variant, grammar, style and content edits. Only the sentence can tell these apart, so they are
not counted as spelling fixes; `p_spelling` and `sonnet_label` in the `edits` config judge them
one by one. This includes pairs whose old form is a rare listed word that editors have replaced:
an earlier version called those `variant`, but in a hand-checked sample 21 of 34 were real-word
typos where the old form happens to be a word (`انجم → انجمن`, `قیل → فیل`). Of the `high`
pairs, 12,156 are `non_word`, 578 `context_dependent`, 53 `variant` and 195 `unverified`.

### Real-word errors in context

For each `context_dependent` edit, the changed word is masked in the corrected sentence and
the Persian BERT model [`HooshvareLab/bert-fa-zwnj-base`](https://huggingface.co/HooshvareLab/bert-fa-zwnj-base)
gives the log-probability of the old and the new word in that slot (words of several pieces
are scored piece by piece, left to right). A logistic regression combines the two scores with
the shape of the edit: edit distance, whether the words differ only in dots (`زیان`/`زبان`),
and word-piece counts. It was fitted on 299 hand-labelled edits and then frozen
(`annotation/context_classifier.json` in the code repository); `p_spelling` is its output.
Treat `p_spelling ≥ 0.5` as a real-word spelling fix.

Each labelled edit got one of six labels, judged from the sentence: `spelling` (the old word
makes no sense there and is a key, dot or sound away from the new one: `زیان → زبان`,
`المان → آلمان`), `variant`, `grammar` (`فردی → فرد`), `style` (`میشن → میشوند`), `content`
(`بودند → شدند`) and `name` (another transliteration: `مارچ → مارس`).

The frozen classifier was then tested on 150 new edits, drawn at random across the score
range and labelled by a native Persian speaker (7 skipped as unclear). These were drawn before
the pairs that an earlier version called `variant` moved into `context_dependent` (about 5% of
its edits now):

| | value | Wilson 95% interval |
|---|---:|---|
| `context_dependent` edits that are real-word spelling fixes | 45% (64 of 143) | 37% to 53% |
| precision of `p_spelling ≥ 0.5` | 77% | 65% to 86% |
| recall of `p_spelling ≥ 0.5` | 73% | 62% to 83% |

Most false positives are style edits, name transliterations and spelling variants. Most
misses are slips the language model rates poorly, such as `تلفن → تلفظ` and `منتصب → منتسب`.

`sonnet_label` is a second, independent judgement of the same edits. Claude Sonnet 5.5
(`claude-sonnet-5-5`, adaptive extended thinking at effort `high`) labelled every scored edit
on 2026-11-25 with the six labels above, 50 edits per request, from a window of 80 characters
on each side of the changed word; the prompt is `annotation/llm_guideline_v3.txt` in the code
repository. On the 143 test edits (an earlier run of the same prompt), Sonnet's `spelling` had
a precision of 87% (75% to 93%) and a recall of 72% (60% to 81%), and it agreed with the human
annotator at Cohen's κ = 0.60 over the six labels. Two runs of the same prompt gave the same
label for 89% of edits.

| over all 76,759 scored edits | edits |
|---|---:|
| `p_spelling ≥ 0.5` | 28,409 |
| `sonnet_label = spelling` | 23,994 |
| both | 18,849 |

Of the 4,192 edits that moved in from the old `variant` rule, Sonnet labels 50% `spelling` and
17% `variant`.

Where both agree is the most reliable subset: 90% precision (77% to 96%) on the test edits,
with 58% recall. This combination was chosen after looking at the test results, so treat
that figure as indicative.

## Validation

200 `high` pairs were drawn at random (seed 0) from the previous build of this dataset, which
did not yet strip category and file links, and labelled by a native Persian speaker, who
looked at the sentence before and after each edit. 193 of the 200 are still `high` in this
build. 197 were spelling fixes, a precision of
98.5% (Wilson 95% interval 95.7% to 99.5%). The other three were a transliteration of a name
(`ماینکرافت → ماینکرفت`), a switch between two accepted spellings
(`سپیدپوستان → سفیدپوستان`), and a name split by mistake (`کیی → کی`, from `بان کی مون`).
The labels are in `validation_sample.tsv`, with a link to each diff.

The other classes were checked on fresh random samples drawn after the rules were fixed, one
edit per pair, labelled by the same annotator with the six labels of
[Real-word errors in context](#real-word-errors-in-context) (unclear pairs skipped):

| class | pairs labelled | what was checked | result | Wilson 95% interval |
|---|---:|---|---:|---|
| `non_word`, `low` tier | 96 | fixes a spelling error | 82% | 73% to 89% |
| `variant` | 50 | not a spelling error (variant or name) | 100% | 93% to 100% |
| `variant` | 50 | the label is `variant` (5 were names) | 90% | 79% to 96% |
| `unverified` | 37 | a name or transliteration | 70% | 54% to 83% |
| `disputed` | 41 | fixes a spelling error | 34% | 22% to 49% |

Most `non_word` misses are name transliterations, plus a few ezafe removals (`بقیه‌ی → بقیه`).
`disputed` pairs are not reliably non-errors: a third fix real typos that someone reverted, so
treat the class as uncertain. The `variant` sample replaced an earlier check of a broader rule
(24% precision), which led to the narrower rule described in [Error type](#error-type).

There was one annotator, so no human agreement score is available. Claude Sonnet 5.5 labelled
the same samples with the same guideline; its agreement with the human annotator is Cohen's
κ = 0.60 over the six labels on the 150 `p_spelling` test edits (0.64 for spelling against the
rest), and κ = 0.59 on the stage 1 samples above.

### `p_fix` and `fix_tier`

`p_fix` was measured by 5-fold cross-validation on its 726 hand-labelled pairs, repeated 10
times. Pairs with the same corrected word stayed in one fold, and each class counted in
proportion to its size in the dataset. Recall is the share of all spelling fixes outside
`spacing` that the row finds.

| pairs | precision | bootstrap 95% interval | recall |
|---|---:|---|---:|
| `confidence = high` | 96% | | 15% |
| `fix_tier = sure` (`p_fix ≥ 0.82`) | 95% | 92% to 98% | 67% |
| `p_fix ≥ 0.5` | 88% | 84% to 92% | 84% |

How often `p_fix ≥ 0.5` agrees with the hand label, next to always answering with the more
common label of the class:

| class | pairs labelled | `p_fix ≥ 0.5` | more common label |
|---|---:|---:|---:|
| `context_dependent` | 282 | 82% | 57% |
| `non_word`, `low` tier | 102 | 89% | 83% |
| `disputed` | 41 | 68% | 66% |
| `unverified` | 38 | 90% | 89% |

Only in `context_dependent` is the gain larger than these samples can resolve. In the other
classes use `p_fix` to rank pairs, not to decide single ones.

`p_fix` can be read as a probability. In the same cross-validation, the share of pairs that
are spelling fixes was:

| `p_fix` | spelling fixes |
|---|---:|
| under 0.1 | 9% |
| 0.1 to 0.3 | 16% |
| 0.3 to 0.5 | 34% |
| 0.5 to 0.7 | 64% |
| 0.7 to 0.82 | 71% |
| 0.82 to 0.9 | 84% |
| 0.9 and over | 98% |

Each row but the last rests on 40 to 150 labelled pairs, so the shares are rough.

A second check used pairs the model never saw: the 137 pairs of the 150-edit test sheet of
[Real-word errors in context](#real-word-errors-in-context), 61 of them spelling fixes.

| pairs | picked | spelling fixes | precision | Wilson 95% interval | recall |
|---|---:|---:|---:|---|---:|
| `fix_tier = sure` | 37 | 36 | 97% | 86% to 99.5% | 59% |
| `p_fix ≥ 0.5` | 50 | 43 | 86% | 74% to 93% | 70% |
| most of the pair's `sonnet_label`s are `spelling` | 47 | 43 | 91% | 80% to 97% | 70% |

For a yes-or-no answer on a `context_dependent` pair, Sonnet's label alone is at least as good
as `p_fix ≥ 0.5`. `p_fix` orders these pairs better (average precision 0.86 against 0.79) and
also covers the 120,000 pairs outside `spacing` that Sonnet did not label.

## Limitations

Wikipedia is formal, encyclopedic writing. Mistakes typical of chat or phone keyboards, and
colloquial spellings in general, are rare here.

Some `high` pairs are conventions or transliteration choices that not everyone would call
errors: hamza (`موثر → مؤثر`), joining a prefix (`بطور → به‌طور`), foreign names
(`آینشتاین → اینشتین`). A few renames done by two editors also pass the filter, for example
`کوخری → کوخردی` (155 edits on 151 pages). Whether to keep them depends on what you are
building.

Only errors that someone fixed are here, and common errors are counted many times. Use
`editors` or `pages` if you want to weight them.

Only single-word replacements are extracted. Missing or extra spaces that split or merge
words, and rewrites of several words, are not included.

## Privacy

Everything comes from public Wikipedia history. Usernames and IP addresses are not
published; `editors` is only a count. Every edit can be traced to the public revision through
its `rev_id`.

## Licence

The text comes from Persian Wikipedia and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and so is this dataset.
Attribution: Persian Wikipedia contributors, https://fa.wikipedia.org. The extraction code in
`fadict/`, `scripts/` and `annotation/` on GitHub is MIT-licensed.

## Rebuilding it

The code is in the [GitHub repository](https://github.com/Fazel94/persian-wikipedia-spelling-corrections),
taken from the author's Persian keyboard dictionary project. You need Python 3.10 or newer,
`pip install regex spylls`, and the `7z` and `unzip` commands (`hunspell` is used instead of
`spylls` when installed).

```sh
scripts/fetch_wikihist.sh 2026-11-01                    # 5.5 GB into data/raw/wikihist/
scripts/fetch_lexicons.sh                               # Lilak, hazm, PerDT into data/raw/
python -m fadict.wikiedits data/raw/wikihist/*.7z       # one worker per core
python -m fadict.wikiedits release                      # writes dist/fawiki_typos/
```

On an 8-core machine the download took about 20 minutes and the scan a little over an hour.

`p_spelling` needs `pip install torch transformers` and takes about 6 minutes on one GPU:

```sh
python -m fadict.context score      # BERT scores of context_dependent edits
python -m fadict.context classify   # frozen annotation/context_classifier.json → p_spelling
python -m fadict.context annotate   # adds p_spelling and sonnet_label to dist/fawiki_typos/edits/
```

The hand labels behind the classifier are in `annotation/context_labels_dev.tsv` (299 edits,
used to fit it) and `annotation/context_labels_test.tsv` (150 edits, used only to test it,
with Claude Sonnet 5.5's labels alongside); Sonnet's labels of all scored edits are in
`annotation/context_sonnet_labels.tsv`.

`p_fix` and `fix_tier` need `pip install scikit-learn` and word counts from two other corpora,
the Leipzig Persian news crawls and the OpenSubtitles Persian text:

```sh
scripts/fetch_corpora.sh               # 740 MB into data/raw/
python -m fadict.count                 # counts their words into data/work/
python -m fadict.confidence annotate   # adds p_fix and fix_tier to dist/fawiki_typos/pairs/
python -m fadict.confidence heldout    # the check on the 137 unseen pairs
python -m fadict.confidence check      # pairs where fix_tier contradicts the edit counts
```

`scripts/bench_confidence.py` runs the cross-validation.

The 726 hand labels are in `annotation/pair_labels.tsv`.

## Citation

If you use this dataset, please cite it with its DOI,
[10.57967/hf/11806](https://doi.org/10.57967/hf/11806):

```bibtex
@misc{fazeli2026persianwikispelling,
  author    = {Fazeli, Mohammad},
  title     = {Persian (Farsi) Wikipedia Spelling Corrections},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/11806},
  url       = {https://huggingface.co/datasets/Phazel/persian-wikipedia-spelling-corrections}
}
```

Mohammad Fazeli. 2026. Persian (Farsi) Wikipedia Spelling Corrections. Hugging Face.
https://doi.org/10.57967/hf/11806
