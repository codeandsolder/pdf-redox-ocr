# pdf-redox-ocr: native Rust rewrite (development)

The Rust implementation lives in this crate while the existing production
Python pipeline is used as an independent, tested reference. This is **not**
a fully equivalent replacement yet. The branch must not displace production
until the reference corpus and source/image-level visibility tests match.

## Implemented

- Hayro-backed source parsing through the same PDF source engine used by pdf-redox,
  including inherited page rectangles and rotation.
- PDF paint instruction parsing through Hayro, with nested Form XObjects,
  source/paint identity, effective CTMs, clip safety, searchable text and
  conservative unsupported-instruction diagnostics.
- Verified page geometry, polygon clipping, batching Poppler's existing native
  words, resource-budgeted native image selection, OCR draft v3 SHA-256 binding,
  source/paint-specific visibility proof admissions and native/fallback merging.
- CPU PP-OCRv6 Small model via native oar-ocr, in-memory Poppler RGB raster
  and a Rust CLI which can apply the OCR draft with pdf-redox.
- Shared sccache CI, pinned central rust-skills2 strict gate.

## Current runner

After compilation, the prototype entrypoint is:

    pdf-redox-ocr --pdf input.pdf --output observations.json \
        --output-pdf searchable.pdf --pages 1-5 --confidence 0.97

The CLI still runs whole-page OCR; it does not yet extract/recognize native
Image XObjects or conduct differential source-paint visibility proofs. The
existing Python path remains the functional baseline.

The source core is tested using stable Rust 1.100 with
cargo +stable test --locked --no-default-features. The integrated crate
requires pinned nightly because the upstream pdf-redox dependency currently
uses zlib-rs-fast portable SIMD. Shared CI runs strict Clippy for both.

## Correctness policy

A source-image result may replace full-page OCR only if the page is pure
source-raster paint and every visible source was recognized. In mixed pages,
additional source text requires matching source+expanded paint ID and an actual
isolation/differential render visibility proof; no guessed overlap is enough.

The complete parity gate is tracked in the Notion worklog:
https://app.notion.com/p/41bf74be902081209382e49903e8f0ff
