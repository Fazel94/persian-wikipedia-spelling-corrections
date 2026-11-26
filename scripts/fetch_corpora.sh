#!/usr/bin/env bash
# Download the corpora that `fadict.count` reads: the Leipzig Persian news crawls of 2017 and
# 2019 and the OpenSubtitles 2018 Persian text (740 MB), into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
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

LEIPZIG=https://downloads.wortschatz-leipzig.de/corpora
get "$LEIPZIG/fas_newscrawl_2019_1M.tar.gz" data/raw/fas_newscrawl_2019_1M.tar.gz
get "$LEIPZIG/fas_newscrawl_2017_1M.tar.gz" data/raw/fas_newscrawl_2017_1M.tar.gz
get https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/fa.txt.gz data/raw/opensubtitles_fa.txt.gz

# Extract only the sentence files of the Leipzig tarballs.
for tgz in data/raw/fas_newscrawl_*_1M.tar.gz; do
	name=$(basename "$tgz" .tar.gz)
	if [[ ! -s data/raw/$name-sentences.txt ]]; then
		tar -xzf "$tgz" -C data/raw --wildcards --no-anchored '*-sentences.txt' --transform 's#.*/##'
	fi
done
