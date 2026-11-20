#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/bin" "$scratch/work" "$scratch/output"
export PDF_REDOX_OCR_BIN="$scratch/bin/recognizer"
export PDF_REDOX_WORK_DIR="$scratch/work"
export CALL_LOG="$scratch/calls"
cat > "$PDF_REDOX_OCR_BIN" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$CALL_LOG"
exit "${FAIL_RECOGNIZER:-0}"
MOCK
chmod +x "$PDF_REDOX_OCR_BIN"
source="$scratch/source.pdf"
output="$scratch/output/searchable.pdf"
printf source > "$source"

"$root/run-auto.sh" "$source" "$output" --pages 1,3 --confidence 0.99 --ocr-threads 2 --raster-mode all --max-side 1600 --keep-native-text --output "$scratch/output/draft.json" --optimized-pdf "$scratch/output/optimized.pdf" --report "$scratch/output/audit.json"
printf '%s\n' --pdf "$source" --output-pdf "$output" --optimize --work-dir "$PDF_REDOX_WORK_DIR" --pages 1,3 --confidence 0.99 --ocr-threads 2 --raster-mode all --max-side 1600 --keep-native-text --output "$scratch/output/draft.json" --optimized-pdf "$scratch/output/optimized.pdf" --report "$scratch/output/audit.json" > "$scratch/expected"
cmp "$scratch/expected" "$CALL_LOG"
[[ -z $(ls -A "$PDF_REDOX_WORK_DIR") ]]

"$root/run-auto.sh" "$source" "$output" --ocr-only --no-recognition-cache --no-leader-repair
printf '%s\n' --pdf "$source" --output-pdf "$output" --no-recognition-cache --no-leader-repair > "$scratch/expected"
cmp "$scratch/expected" "$CALL_LOG"
[[ -z $(ls -A "$PDF_REDOX_WORK_DIR") ]]
[[ $(cat "$source") == source ]]
status=0
FAIL_RECOGNIZER=17 "$root/run-auto.sh" "$source" "$output" --ocr-only || status=$?
[[ $status == 17 ]]
printf 'Native runner regression passed\n'
