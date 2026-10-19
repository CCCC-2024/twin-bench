#!/usr/bin/env bash
# Download a GitHub repository (or some of its files) at a pinned commit.
# Used by twins/*/fetch_upstream.sh; third-party code and data are fetched, never committed.
#
#   tools/fetch_github.sh <owner/repo> <commit> <dest>               whole repository (tarball)
#   tools/fetch_github.sh <owner/repo> <commit> <dest> <path>...     only these files
#
# Skips the download when <dest>/.fetched already records the same commit and every listed file is there
# (a file added to a twin's fetch_upstream.sh is then fetched on the next run).
set -euo pipefail
REPO=$1; COMMIT=$2; DEST=$3; shift 3
if [ -f "$DEST/.fetched" ] && [ "$(cat "$DEST/.fetched")" = "$REPO@$COMMIT" ]; then
  missing=0
  for f in "$@"; do [ -e "$DEST/$f" ] || missing=1; done
  if [ $missing -eq 0 ]; then echo "up to date: $DEST ($REPO@${COMMIT:0:7})"; exit 0; fi
fi
mkdir -p "$DEST"
if [ $# -eq 0 ]; then
  curl -fsSL "https://github.com/$REPO/archive/$COMMIT.tar.gz" | tar xz -C "$DEST" --strip-components 1
else
  for f in "$@"; do
    mkdir -p "$DEST/$(dirname "$f")"
    curl -fsSL "https://raw.githubusercontent.com/$REPO/$COMMIT/$f" -o "$DEST/$f"
  done
fi
echo "$REPO@$COMMIT" > "$DEST/.fetched"
echo "fetched: $DEST ($REPO@${COMMIT:0:7})"
