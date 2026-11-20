#!/usr/bin/env bash
# Optimize once, recognize in Rust, and import invisible text in-process.
set -euo pipefail

usage() {
    printf '%s\n' "Usage: $0 SOURCE.pdf OUTPUT.pdf [--ocr-only] [--pages 1,8,10] [--confidence 0.97] [--max-side 1920] [--ocr-threads N] [--raster-mode auto|all|off] [--keep-native-text] [--report PATH] [--draft PATH] [--capture-residual-dir DIR]"
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi
source_pdf=$(realpath -e -- "$1")
output_pdf=$(realpath -m -- "$2")
shift 2

ocr_only=0
draft=''
report=''
capture=''
args=()
while (( $# )); do
    case "$1" in
        --ocr-only) ocr_only=1; shift ;;
        --keep-native-text|--no-residual-ocr|--no-recognition-cache)
            args+=("$1"); shift ;;
        --pages|--confidence|--max-side|--ocr-threads|--raster-mode|--dpi|--native-chunk|--max-native-source-pixels|--max-native-total-pixels|--max-native-sources|--max-native-extraction-pixels|--max-mosaic-group-pixels|--max-mosaic-page-pixels|--max-mosaic-groups|--max-mosaic-dpi|--report|--draft|--capture-residual-dir)
            if (( $# < 2 )); then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            case "$1" in
                --report) report=$(realpath -m -- "$2") ;;
                --draft) draft=$(realpath -m -- "$2") ;;
                --capture-residual-dir) capture=$(realpath -m -- "$2") ;;
                *) args+=("$1" "$2") ;;
            esac
            shift 2 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Protect the original input even when the recognizer sees an optimized copy.
targets=("$output_pdf")
for path in "$draft" "$report" "$capture"; do
    if [[ -n "$path" ]]; then targets+=("$path"); fi
done
for ((i=0; i<${#targets[@]}; i++)); do
    if [[ "${targets[i]}" == "$source_pdf" ]]; then
        printf 'Output must not overwrite the source PDF: %s\n' "${targets[i]}" >&2
        exit 2
    fi
    for ((j=0; j<i; j++)); do
        if [[ "${targets[i]}" == "${targets[j]}" ]]; then
            printf 'Outputs must be distinct: %s\n' "${targets[i]}" >&2
            exit 2
        fi
    done
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
optimizer="${PDF_REDOX_RELEASE_BIN_DIR:-$(dirname -- "$script_dir")/pdf-redox/target/release}/pdf-redox"
recognizer="${PDF_REDOX_OCR_BIN:-$script_dir/rust/target/release/pdf-redox-ocr}"
if [[ ! -x "$recognizer" ]]; then
    printf 'Rust OCR executable not found: %s\nSet PDF_REDOX_OCR_BIN to a compiled executable.\n' "$recognizer" >&2
    exit 2
fi
if (( ! ocr_only )) && [[ ! -x "$optimizer" ]]; then
    printf 'PDF optimizer executable not found: %s\nSet PDF_REDOX_RELEASE_BIN_DIR.\n' "$optimizer" >&2
    exit 2
fi

scratch=$(mktemp -d -p "${PDF_REDOX_WORK_DIR:-${TMPDIR:-/tmp}}" pdf-redox-ocr-XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
input_pdf="$source_pdf"
if (( ! ocr_only )); then
    input_pdf="$scratch/optimized.pdf"
    "$optimizer" "$source_pdf" --optimize-for processing -o "$input_pdf"
fi
if [[ -z "$draft" ]]; then
    draft="$scratch/draft.json"
else
    mkdir -p -- "$(dirname -- "$draft")"
fi
mkdir -p -- "$(dirname -- "$output_pdf")"
if [[ -n "$report" ]]; then
    mkdir -p -- "$(dirname -- "$report")"
    args+=(--report "$report")
fi
if [[ -n "$capture" ]]; then
    args+=(--capture-residual-dir "$capture")
fi
"$recognizer" --pdf "$input_pdf" --output "$draft" --output-pdf "$output_pdf" "${args[@]}"
