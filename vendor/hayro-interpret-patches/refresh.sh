#!/usr/bin/env bash
set -euo pipefail
patches="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
vendor="$(dirname -- "$patches")/hayro-interpret"
upstream="${1:?usage: refresh.sh /path/to/upstream-hayro [--check]}"
mode="${2:-}"
if [[ -n "$mode" && "$mode" != --check ]]; then
    echo "Unknown argument: $mode" >&2
    exit 2
fi
rev="$(cat "$patches/UPSTREAM_REV")"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT
git -C "$upstream" archive "$rev" hayro-interpret LICENSE-APACHE LICENSE-MIT | tar -x -C "$scratch"
cp --remove-destination -- "$scratch/LICENSE-APACHE" "$scratch/hayro-interpret/LICENSE-APACHE"
cp --remove-destination -- "$scratch/LICENSE-MIT" "$scratch/hayro-interpret/LICENSE-MIT"
for patch in "$patches"/*.patch; do
    git -C "$scratch" apply --no-index "$patch"
done
if [[ "$mode" == --check ]]; then
    diff -ru "$scratch/hayro-interpret" "$vendor"
else
    rsync -a --delete -- "$scratch/hayro-interpret/" "$vendor/"
fi
