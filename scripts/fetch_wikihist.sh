#!/usr/bin/env bash
# Download the fawiki full-history 7z parts (MediaWiki Content History export, ~5.5 GB) for
# `python -m fadict.wikiedits`. Resumable; parts already scanned into data/work/wikiedits are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

DATE=${1:-2026-11-01}
BASE=https://dumps.wikimedia.org/other/mediawiki_content_history/fawiki/$DATE/xml/7z
DEST=data/raw/wikihist
mkdir -p "$DEST"

curl -sS --fail "$BASE/" | grep -oE 'href="fawiki-[^"]+\.xml\.7z"' | cut -d'"' -f2 |
	while read -r f; do
		if [[ -s $DEST/$f || -s data/work/wikiedits/${f%.xml.7z}.tsv ]]; then
			continue
		fi
		echo "fetch $f"
		until curl -L --fail -sS --http1.1 -C - -o "$DEST/$f.part" "$BASE/$f"; do sleep 3; done
		mv "$DEST/$f.part" "$DEST/$f"
	done
