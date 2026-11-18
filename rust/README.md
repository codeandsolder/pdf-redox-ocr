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
- CPU PP-OCRv6 Small model via native oar-ocr, in-memory Poppler RGB raster,
  exact XObject-native source extraction with Poppler, one inference per distinct
  source object, and conservative source-only page fast path.
- Rust CLI which writes the importer draft and optionally applies the OCR draft
  with pdf-redox, keeping original vector content unchanged.
- Shared sccache CI, pinned central rust-skills2 strict gate.

## Current runner

After compilation, the prototype entrypoint is:

    pdf-redox-ocr --pdf input.pdf --output observations.json \
        --output-pdf searchable.pdf --pages 1-5 --confidence 0.97

Native source decoding is explicitly bounded by per-source, total-resident,
source-count and single-extraction pixel ceilings. Those controls are exposed
as CLI options. If an image cannot fit, its page uses ordinary page OCR.

The CLI recognizes native image XObjects once per source and skips page OCR
on independently paint-safe raster-only pages. On mixed pages, native source
labels are admitted only when a detached copy-on-write PDF aliases a single
image Do invocation through its nested Form chain to independently masked
source pixels, and a differential final-page render proves the change survives
the full painting stack. The result carries source, expanded paint identity
and full invocation path through a provenance-gated merge with page OCR.
Repeated source and Form invocations on the same page are independently
audited. Counterfactual image masks work on the original packed sample values
(1/2/4/8/16 bits per component), including Gray, RGB, CMYK, Indexed and
ICC-based color spaces, without changing the original ColorSpace or Decode
mapping. Unresolved color aliases, alpha, unsupported sample codecs and
incomplete source data keep ordinary page OCR. A differential proof also
requires every pixel outside the selected OCR region to remain unchanged.
For fragmented image-only pages, the Rust path now clusters actual native
image paint invocations using scale-aware, spatially indexed connected
components. Adjacent strips are recognized in bounded native-pitch PDF crops,
not as isolated text fragments. Poppler renders the final paint stack, so soft
masks, ICC color, clipping and nested Form paints use the genuine PDF
compositor. Group-specific clip masks, exact affine source-axes rectification (including
arbitrary rotations, reflection and modest source shear), and one shared
inverse crop-pixel-to-PDF transform preserve provenance.
Pages are promoted only after all image paints have been reconstructed and
recognized. Failed geometric, compositing, resource or inference checks use
ordinary full-page OCR rather than the independent strip shortcut. CLI knobs
bound per-group and per-page rendered pixels, group count and native DPI.
For fallback page OCR, a temporary Rust-generated PDF makes native searchable
font shows invisible using mode 3 while retaining the original text-show
operands, advances and geometry. Type3 glyphs, unresolved fonts, existing
invisible shows and text-clipping modes remain as originally painted. Nested
and reused Form XObjects are cloned copy-on-write for their inherited page
resource context, rather than modifying shared source objects. Existing
native text extraction, source visibility proofs, the input SHA-256 and the
final vector overlay still use the **original PDF**. The diagnostic CLI option
`--keep-native-text` disables only this render-time native font suppression.
The original PDF is never modified.

The real-PDF regression corpus covers repeated image extraction, masked source
rewriting, a source visible on one page but occluded on another, and
repeated direct/once-nested/twice-nested source painting on the same page,
with one occurrence covered by later opaque vector paint. A ten-page
real-PDF corpus checks native packed pixel round trips and absence of unrelated
render changes across 1/2/4/8/16-bit device and calibrated color spaces.
The real-PDF fragment regression corpus covers split raster text, soft masks,
clip paths, rotated pages and nonzero MediaBox origins, plus arbitrary 17° and
37° rotations, mirrored −28° raster and sheared 13° input. The five-page native-font
regression verifies that shared Forms with context-dependent inherited fonts
do not hide Type3 ink on sibling pages, and that extracted text and unselected
pages remain unchanged. This fixes a cross-page Form mutation bug in the
current Python reference, which remains the production baseline until corpus
and performance parity is demonstrated.

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
