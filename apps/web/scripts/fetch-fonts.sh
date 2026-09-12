#!/usr/bin/env bash
# Re-download the self-hosted webfonts into public/fonts/.
#
# The app does not load fonts from a CDN: the demo has to run with no internet,
# and Google's hashed filenames rotate, so a pinned URL eventually 404s and the
# UI silently drops to a fallback face. Every number column depends on JetBrains
# Mono's tabular figures, so that fallback is not cosmetic.
#
# Run from apps/web/. Needs network. Both families are SIL Open Font License 1.1.
set -euo pipefail

UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
OUT="public/fonts"
mkdir -p "$OUT"

# Pull the woff2 URL for one unicode subset out of the Google Fonts CSS. The
# response lists a @font-face block per subset, each preceded by a /* name */
# comment, so find the comment and take the next url().
subset_url() {
  curl -fsS -H "User-Agent: $UA" "https://fonts.googleapis.com/css2?family=$1&display=swap" \
    | awk -v want="/* $2 */" 'index($0,want){found=1}
                              found && /url\(https/{match($0,/https:[^)]*\.woff2/);
                                                    print substr($0,RSTART,RLENGTH); exit}'
}

fetch() { # query, output basename
  for subset in latin latin-ext; do
    url="$(subset_url "$1" "$subset")"
    if [ -z "$url" ]; then
      echo "could not resolve $2 $subset" >&2
      exit 1
    fi
    curl -fsS -o "$OUT/$2-$subset.woff2" "$url"
    echo "$2-$subset.woff2  $(wc -c < "$OUT/$2-$subset.woff2") bytes"
  done
}

fetch "Inter:wght@100..900" inter
fetch "JetBrains+Mono:wght@100..800" jetbrains-mono

echo "Done. The unicode-range blocks in src/index.css must match these subsets."
