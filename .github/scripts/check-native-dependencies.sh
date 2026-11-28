#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)/rust"
# Check active edges, not every optional package in Cargo.lock.
# Build dependencies and all target-specific branches also count.
graph=$(cargo tree --color never --locked --all-features --target all --edges normal,build --prefix none --format '{p}' | sed 's/ (\*)$//' | sort -u)
if printf '%s\n' "$graph" | rg '^(miniz_oxide|zune-jpeg|zune-core|rav1e|ravif|exr|gif|tiff|image-webp|qoi|ab_glyph|owned_ttf_parser|ttf-parser|rustdct|rustfft|tokenizers|onig|onig_sys|paste|zip) v'; then
    echo 'Unexpected native OCR codec or optional inference pipeline' >&2
    exit 1
fi
mapfile -t deflate < <(printf '%s\n' "$graph" | rg '^zlib-rs v')
if [[ ${#deflate[@]} != 1 || ${deflate[0]} != *github.com/codeandsolder/zlib-rs-fast* ]]; then
    printf 'Expected one active shared zlib-rs-fast source; got:\n%s\n' "${deflate[@]}" >&2
    exit 1
fi
mapfile -t jpeg < <(printf '%s\n' "$graph" | rg '^libjpeg-turbo-rs v')
if [[ ${#jpeg[@]} != 1 ]]; then
    printf 'Expected one active JPEG implementation; got:\n%s\n' "${jpeg[@]}" >&2
    exit 1
fi
printf 'Validated one DEFLATE and one JPEG source:\n%s\n%s\n' "${deflate[0]}" "${jpeg[0]}"
