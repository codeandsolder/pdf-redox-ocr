#!/usr/bin/env bash
# Convenience entrypoint; the native executable owns all preparation and output policy.
set -euo pipefail

usage() {
    printf '%s\n' "Usage: $0 SOURCE.pdf OUTPUT.pdf [--ocr-only] [native OCR options...]"
    printf '%s\n' "Processing optimization is enabled unless --ocr-only immediately follows OUTPUT.pdf."
    printf '%s\n' "Use --output DRAFT.json --optimized-pdf PREPARED.pdf to retain a processing-bound draft."
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi

source_pdf=$1
output_pdf=$2
shift 2
preparation=(--optimize)
if [[ ${1:-} == --ocr-only ]]; then
    preparation=()
    shift
elif [[ -n ${PDF_REDOX_WORK_DIR:-} ]]; then
    preparation+=(--work-dir "$PDF_REDOX_WORK_DIR")
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
recognizer="${PDF_REDOX_OCR_BIN:-$script_dir/rust/target/release/pdf-redox-ocr}"
if [[ ! -x "$recognizer" ]]; then
    printf 'Rust OCR executable not found: %s\nSet PDF_REDOX_OCR_BIN to a compiled executable.\n' "$recognizer" >&2
    exit 2
fi
exec "$recognizer" --pdf "$source_pdf" --output-pdf "$output_pdf" "${preparation[@]}" "$@"
