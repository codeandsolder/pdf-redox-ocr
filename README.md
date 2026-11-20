# pdf-redox-ocr

Native Rust structure-aware PDF OCR, integrated with
[pdf-redox](https://github.com/codeandsolder/pdf-redox) and its invisible-text
importer. Existing searchable text and visible page paints are preserved;
native image recognition, reconstructed mosaics and residual page OCR are
selected using source geometry and independently checked visibility.

## Build and run

Install Poppler's `pdfimages`, `pdftoppm` and `pdftotext` executables.
The integrated build uses nightly Rust because the pinned optimizer's
compression backend uses portable SIMD. The model-free core supports stable.

```sh
cd rust
cargo +nightly-2026-11-17 build --release --locked --all-features
cd ..
./run-auto.sh input.pdf searchable.pdf
```

The launcher defaults to Processing optimization followed by OCR. Use
`./run-auto.sh input.pdf searchable.pdf --ocr-only` to retain the original
PDF as the recognition input. Use `--pages 1-5` to limit recognition,
`--report audit.json` for evidence and `--output draft.json` for the importer
draft. For optimized input, also specify `--optimized-pdf prepared.pdf`
to retain the exact PDF to which the draft is bound.

The native Small recognizer is the default; the Medium variant and
selective confirmation are explicit options. Source-only pages do not
load recognition models. The OAR cache can be configured with `OAR_HOME`.
For the complete option set, see `pdf-redox-ocr --help`.

## Development

See [the native implementation and diagnostics guide](rust/README.md)
for geometry conventions, source/paint admission, crop repair, vector
catalog association, independently reviewed glyph matching and evaluation.

```sh
cd rust
cargo +stable test --locked --no-default-features
cargo +nightly-2026-11-17 test --locked --all-features
```

The legacy Python pipeline, its tests, all original documentation and
experimental scripts remain available at the immutable baseline
[legacy-python-reference-20261120](https://github.com/codeandsolder/pdf-redox-ocr/tree/legacy-python-reference-20261120).
They are independent comparison tools, not a second runtime maintained
inside the Rust implementation. The Rust rewrite branch remains a draft
cutover candidate until representative corpus quality and performance
are verified; the archived experimental research scripts are not claimed
to have one-for-one replacements.
