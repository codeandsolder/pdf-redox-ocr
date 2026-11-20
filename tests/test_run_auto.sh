#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/bin" "$scratch/work" "$scratch/output"
export PDF_REDOX_RELEASE_BIN_DIR="$scratch/bin"
export PDF_REDOX_OCR_BIN="$scratch/bin/recognizer"
export PDF_REDOX_WORK_DIR="$scratch/work"
export CALL_LOG="$scratch/calls"

cat > "$scratch/bin/pdf-redox" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
[[ $# == 5 && $2 == --optimize-for && $3 == processing && $4 == -o ]]
printf 'optimize\n' >> "$CALL_LOG"
printf 'optimized:' > "$5"
cat "$1" >> "$5"
MOCK
cat > "$scratch/bin/recognizer" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf 'recognize\n' >> "$CALL_LOG"
[[ $1 == --pdf && $3 == --output && $5 == --output-pdf ]]
input=$2
draft=$4
output=$6
shift 6
printf '%s\n' "$@" > "$CALL_LOG.args"
cat "$input" > "$output"
printf 'draft:' > "$draft"
cat "$input" >> "$draft"
while (( $# )); do
    case "$1" in
        --report) printf 'audit:' > "$2"; cat "$input" >> "$2"; shift 2 ;;
        --keep-native-text|--no-recognition-cache|--no-residual-ocr) shift ;;
        *) shift 2 ;;
    esac
done
if [[ "${FAIL_RECOGNIZER:-0}" == 1 ]]; then exit 17; fi
MOCK
chmod +x "$scratch/bin/"*
printf 'source' > "$scratch/source.pdf"
source="$scratch/source.pdf"
output="$scratch/output/searchable.pdf"

"$root/run-auto.sh" "$source" "$output" --pages 1,3 --confidence 0.99 --ocr-threads 2 --raster-mode all --max-side 1600 --keep-native-text --draft "$scratch/output/draft.json" --report "$scratch/output/audit.json"
[[ $(cat "$CALL_LOG") == $'optimize\nrecognize' ]]
[[ $(cat "$output") == optimized:source ]]
[[ $(cat "$scratch/output/draft.json") == draft:optimized:source ]]
[[ $(cat "$scratch/output/audit.json") == audit:optimized:source ]]
expected=$'--pages\n1,3\n--confidence\n0.99\n--ocr-threads\n2\n--raster-mode\nall\n--max-side\n1600\n--keep-native-text\n--report\n'
[[ $(cat "$CALL_LOG.args") == "$expected$scratch/output/audit.json" ]]
[[ -z $(ls -A "$PDF_REDOX_WORK_DIR") ]]

: > "$CALL_LOG"
PDF_REDOX_RELEASE_BIN_DIR="$scratch/nonexistent" "$root/run-auto.sh" "$source" "$output" --ocr-only --no-recognition-cache
[[ $(cat "$CALL_LOG") == recognize ]]
[[ $(cat "$output") == source ]]
[[ $(cat "$source") == source ]]
[[ -z $(ls -A "$PDF_REDOX_WORK_DIR") ]]

: > "$CALL_LOG"
reject() {
    if "$root/run-auto.sh" "$@" 2>/dev/null; then
        printf 'Unexpected success: %s\n' "$*" >&2
        exit 1
    fi
}
reject "$source" "$source"
reject "$source" "$output" --report "$source"
reject "$source" "$output" --draft "$output"
reject "$source" "$output" --confidence
reject "$source" "$output" --unknown
ln -s "$source" "$scratch/alias"
reject "$source" "$scratch/alias"
[[ ! -s "$CALL_LOG" ]]
[[ $(cat "$source") == source ]]
FAIL_RECOGNIZER=1 reject "$source" "$output" --ocr-only
[[ -z $(ls -A "$PDF_REDOX_WORK_DIR") ]]
printf 'Native runner regression passed\n'
