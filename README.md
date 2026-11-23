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
    path: edits/train.jsonl.gz
  - split: test
    path: edits/test.jsonl.gz
---

# Persian (Farsi) Wikipedia Spelling Corrections

If you have ever fixed `برگذار` to `برگزار` on Persian Wikipedia, this dataset is partly yours.
It collects the spelling fixes that Persian Wikipedia editors have made since the site began,
taken from the full edit history: 349,350 single-word fixes covering 148,725 distinct
misspellings, each with the sentence before and after the edit.

The errors come from people typing, so they look like real Persian typos: letters with the
same sound swapped (`حاظر → حاضر`), dots dropped or misplaced (`همجنین → همچنین`), letters
missed (`تلوزیون → تلویزیون`). 12,915 pairs are marked `confidence=high`. On a random sample
of 200 of them checked by hand, 197 were real spelling fixes (98.5%).

| typo | fix | edits | editors |
|---|---|---:|---:|
| `برگذار` | `برگزار` | 463 | 136 |
| `حاظر` | `حاضر` | 354 | 138 |
| `تلوزیون` | `تلویزیون` | 274 | 86 |
| `تاتر` | `تئاتر` | 118 | 87 |
| `همجنین` | `همچنین` | 89 | 72 |

## Loading the data

```python
from datasets import load_dataset

pairs = load_dataset("Phazel/persian-wikipedia-spelling-corrections", "pairs")
edits = load_dataset("Phazel/persian-wikipedia-spelling-corrections", "edits")
test = pairs["test"].filter(lambda r: r["confidence"] == "high")  # 1,289 pairs
```

The same files are on [GitHub](https://github.com/Fazel94/persian-wikipedia-spelling-corrections).

| config | files | one row per |
|---|---|---|
| `pairs` (default) | `pairs/{train,test}.tsv` | misspelling and its fix, with counts |
| `edits` | `edits/{train,test}.jsonl.gz` | one fix in one revision, with its sentence |

`validation_sample.tsv` holds the 200 hand-labelled pairs and `stats.json` the totals.

About 10% of pairs are in `test`. The split is decided by a hash of the corrected word, so a
word that appears as a fix in `test` never appears as a fix in `train`. Each edit goes to the
same split as its pair.

### A record from `edits`

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
  "confidence": "high"
}
```

* `rev_id`: the revision that made the fix. Open `https://fa.wikipedia.org/w/index.php?diff=<rev_id>` to see it.
* `page_id`, `timestamp`: the article and the time of the edit.
* `wrong`, `right`: the word before and after, normalized (see below).
* `before`, `after`: the edited sentence with wiki markup removed. Empty when the sentence is longer than 500 characters (3% of edits).
* `editor_type`: `user` for a registered account, `ip` for an anonymous editor.
* `summary_flagged`: the edit summary mentions spelling (`املا`, `غلط`, `تایپ`, `typo` and similar).
* `kind`: `zwnj` when the two words differ only by the half-space (ZWNJ), otherwise `word`.
* `confidence`: `high` or `low`, copied from the pair.

### Columns in `pairs`

* `wrong`, `right`, `kind`, `confidence`: as above.
* `edits`: how many revisions made this replacement.
* `pages`, `editors`: how many different articles and editors (accounts or IP addresses) made it.
* `first_year`, `last_year`: when it was first and last made.
* `summary_flagged`: how many of those edits had a summary mentioning spelling.
* `reverse_edits`: how many edits went the other way, replacing `right` with `wrong`.
* `wiki_freq_wrong`, `wiki_freq_right`: how often each word occurs in the current text of all articles.

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

The changed lines are split into sentences, the old and new sentences are aligned, and each
aligned pair is split into words. Every place where one word was replaced by exactly one other
word within a Damerau-Levenshtein distance of 2 becomes a fix.

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

## Validation

200 `high` pairs were drawn at random (seed 0) and labelled by a native Persian speaker, who
looked at the sentence before and after each edit. 197 were spelling fixes, a precision of
98.5% (Wilson 95% interval 95.7% to 99.5%). The other three were a transliteration of a name
(`ماینکرافت → ماینکرفت`), a switch between two accepted spellings
(`سپیدپوستان → سفیدپوستان`), and a name split by mistake (`کیی → کی`, from `بان کی مون`).
The labels are in `validation_sample.tsv`, with a link to each diff.

There was one annotator, so no agreement score is available, and the precision of the `low`
pairs was not measured.

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
`fadict/` and `scripts/` on GitHub is MIT-licensed.

## Rebuilding it

The code is in the [GitHub repository](https://github.com/Fazel94/persian-wikipedia-spelling-corrections),
taken from the author's Persian keyboard dictionary project. You need Python 3.10 or newer,
`pip install regex`, and the `7z` command.

```sh
scripts/fetch_wikihist.sh 2026-11-01                    # 5.5 GB into data/raw/wikihist/
python -m fadict.wikiedits data/raw/wikihist/*.7z       # one worker per core
python -m fadict.wikiedits release                      # writes dist/fawiki_typos/
```

On an 8-core machine the download took about 20 minutes and the scan a little over an hour.

## Citation

If you use this dataset, please cite it with its DOI,
[10.57967/hf/11806](https://doi.org/10.57967/hf/11806):

```bibtex
@misc{fazeli2026persianwikispelling,
  author    = {Fazeli, Kiyarash},
  title     = {Persian (Farsi) Wikipedia Spelling Corrections},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/11806},
  url       = {https://huggingface.co/datasets/Phazel/persian-wikipedia-spelling-corrections}
}
```

Kiyarash Fazeli. 2026. Persian (Farsi) Wikipedia Spelling Corrections. Hugging Face.
https://doi.org/10.57967/hf/11806
