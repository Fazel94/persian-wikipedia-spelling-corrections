#!/usr/bin/env bash
# Download the lexicons that `fadict.lexicon` reads: Lilak (hunspell and lemma list), hazm data
# files and the UD Persian PerDT treebank, into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/hazm data/raw/perdt
# Download directly: a local proxy in the environment cuts long transfers.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

get() { # get <url> <dest>; resumes interrupted transfers
	local url=$1 dest=$2 tries=0
	if [[ -s $dest ]]; then
		echo "skip $dest"
		return
	fi
	echo "fetch $url"
	until curl -L --fail -sS --http1.1 -C - -o "$dest.part" "$url"; do
		((++tries < 50)) || return 1
		sleep 3
	done
	mv "$dest.part" "$dest"
}

get https://github.com/b00f/lilak/releases/download/v3.3/fa-IR.zip data/raw/fa-IR.zip
get https://raw.githubusercontent.com/b00f/lilak/master/src/data/lexicon data/raw/lilak_lexicon
if [[ ! -s data/raw/fa-IR/fa-IR.aff ]]; then
	(cd data/raw && unzip -o -q fa-IR.zip)
fi

HAZM=https://raw.githubusercontent.com/roshan-research/hazm/v0.10.0/hazm/data
for f in words.dat verbs.dat iwords.dat iverbs.dat; do
	get "$HAZM/$f" "data/raw/hazm/$f"
done

PERDT=https://raw.githubusercontent.com/UniversalDependencies/UD_Persian-PerDT/master
for split in train dev test; do
	get "$PERDT/fa_perdt-ud-$split.conllu" "data/raw/perdt/fa_perdt-ud-$split.conllu"
done
