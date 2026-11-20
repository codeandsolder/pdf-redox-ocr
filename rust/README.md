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
  conservative unsupported-instruction diagnostics. Form BBox corners are
  normalized independently of page MediaBoxes; degenerate Form rectangles
  disable source-visibility admission without aborting ordinary page OCR.
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
Optional Poppler native PNG extraction and decoding failures likewise trigger
page OCR, after preflighting exported raster dimensions against inventory bounds.
The PP-OCRv6 Small detector retains original detail up to 2048 pixels on its
longest side rather than downscaling every page to the OAR 960-pixel preset;
smaller native images are not artificially enlarged. The page raster defaults
to a 1920-pixel longest side (override with `--max-side`), following targeted
1600/1800/1920/2048/2304-pixel tests of photographed electrical markings;
the larger render has higher CPU and memory costs. The detector profile and
actual page raster limit and DPI are included in OCR draft model provenance
so outputs from different resolutions are not mistaken for the same run.

The CLI recognizes native image XObjects once per source and skips page OCR
on independently paint-safe raster-only pages. On mixed pages, native source
labels are admitted only when a detached copy-on-write PDF aliases a single
image Do invocation through its nested Form chain to independently masked
source pixels, and a differential final-page render proves the change survives
the full painting stack. The result carries source, expanded paint identity
and full invocation path through a provenance-gated merge with page OCR.
Repeated source and Form invocations on the same page are independently
audited. The copy-on-write isolation API accepts grouped exact paint
replacements across pages: one replacement image can be reused, and common
invocation-path prefixes clone their Form once. Original resource bindings
and names used by the original content are reserved; duplicate, conflicting
or invalid invocation paths cannot produce an edited PDF. This allows
verified combined residual renders without mutating sibling calls. Counterfactual image masks work on the original packed sample values
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
and reused Form XObjects are cloned copy-on-write for each exact Do
invocation, preserving the selected font object and rendering mode from the
calling graphics state, including q/Q saves and Form-local resource-name
collisions. Native font classification is captured when Tf executes or
when an extended graphics state selects a font through its Font array;
the font object's identity is inherited through nested Forms rather than
re-resolved by name against their different resource tables. The native
paint scanner applies the same selection rule. Independent Form calls
may therefore be hidden, already invisible, clipping, or graphical Type3,
without modifying sibling invocations. Existing native text extraction,
source visibility proofs, the input SHA-256 and the final vector overlay
still use the **original PDF**. The diagnostic CLI option
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
pages remain unchanged. A further three-page fixture exercises multiple
invocations of the same Form under different Tf/Tr modes, nested Forms,
Type3 calls, explicit Form-local font aliases that disagree with inherited
fonts, and collision-safe resource aliases. A two-page extended-graphics-state
fixture additionally exercises native and graphical fonts selected through
Font entries, selection inside Forms, original text extraction and unchanged
unselected-page pixels. This avoids the cross-page Form mutation found in
the Python reference, which remains the production baseline until corpus and
performance parity is demonstrated.

The source core is tested using stable Rust 1.100 with
cargo +stable test --locked --no-default-features. The integrated crate
requires pinned nightly because the upstream pdf-redox dependency currently
uses zlib-rs-fast portable SIMD. Shared CI runs strict Clippy for both.

## Library orchestration

The CLI uses the same `pipeline::PreparedOcr` interface available to native Rust
callers. `PreparedOcr::open` validates the selection and resource policies,
binds the source bytes, and extracts native word exclusion geometry before
model initialization. `recognize` then runs source reuse, source-resolution
mosaics, paint visibility proofs, and the masked full-page fallback through
one `RecognitionBackend`. The backend receives owned RGB rasters and
returns typed `PixelLine` observations in the original raster's pixel frame,
with no PDF placement data. The job-local `RecognitionSession` maps each
occurrence after inference. Successful results (including empty observations)
are reused across native sources, rectified mosaics, residual rasters and
ordinary fallback using a content key of SHA-256 of all RGB bytes and exact
raster dimensions. The default LRU budget retains at most 256 entries and
4 MiB of observation payload; input image buffers are not retained. Failed
inferences are never cached. `--no-recognition-cache` disables lookup and
retention for diagnostic comparison, and `PipelineResult.recognition`
reports model calls, cache hits, evictions and retained payload.
The optional `--report audit.json` writes a separate schema-1 diagnostic
report bound to the original PDF digest and model provenance. It includes
the validated policy, raw native-image observations in source-pixel
coordinates, original paint paths and native-only blockers, the per-page
routing/residual decisions, differential visibility evidence, merge
proposals, and recognition reuse counters. It is not an importer draft and
cannot authorize an overlay.
The scanner also certifies native-only or empty pages using shared paint
closure and source context; those pages retain their native word regions
without any OCR lines, rendering or font masking. The merge validates an
explicit, exhaustive three-way route partition (native-only, raster-only,
page fallback). Only fallback pages need a font-suppressed raster. Named
pattern color spaces, unsupported paint, clipping, extended graphics state,
annotations and malformed content scopes cannot obtain native-only closure.
Pages with visible native shows but no extracted native words also fall back.
The native model loads lazily, so no ONNX session or model download is
needed for a document consisting entirely of native-only pages; the draft
explicitly declares that OCR was unnecessary. Source extraction queries only
pages known to paint selected image objects, retaining repeated appearances
for a later decode if an earlier occurrence exceeds its resource budget. The
`--keep-native-text` diagnostic disables native-only bypass.

Mixed-page verification now runs page-locally, retaining the original proof
control for fallback and using the independently proved treatment raster
directly when all its suppressed source labels remain eligible for merge.
The suppression plan follows the same merge gates at the individual source
line level. Partially occluded or native-excluded labels stay unaltered;
the exact remaining subset is separately masked and re-proved, with
variants shared by original source identity and observation indices within
the page. Different uses of one image may therefore receive different masks.
Unselected source detections retain their exact original packed component
samples including the mask edge padding, even when an eligible neighbor's
fill region overlaps them. This covers low-confidence, short, occluded and
native-text-excluded observations without changing their source color space.
Several treatments can be combined when their changed RGB pixels are
disjoint and the actual grouped copy-on-write PDF renders exactly to the
same pixels. Native-font masking may be combined with those changes only
where the two controls agree. The full fallback merger is reused before and
after residual recognition: if any suppressed label would be lost through
native-word exclusions, overlapping observations or confidence policy, or
if residual inference fails, the ordinary raster is recognized instead.
No treatment changes the source PDF or its semantic overlay. The
`--no-residual-ocr` diagnostic retains all visibility proofs and disables
only masked recognition, for controlled A/B comparison.

The returned `PipelineResult` includes the v3 importer draft, merge statistics,
admission proposals, original per-paint visibility audits, and the exact set
of accepted residual pages. Progress goes to
the caller's callback, not global standard error. The source hash is checked
during preparation and before/after inference. The CLI alone chooses output
paths and publishes the draft or optional annotated PDF.

With only `pdf-integration` enabled, the *entire* pipeline can be tested against
real PDF fixtures using a deterministic mapped recognizer, without ONNX or
model downloads. Regression coverage includes source deduplication across
nested Form calls, occlusion, native text exclusion, bounded fallback,
source-resolution mosaics and stale source rejection.

## Native Rust recognition-only refinement

The independent second pass is provided by `pdf-redox-ocr-refine`. It consumes
an exact-PDF-bound v3 draft, selects uncertain lines, perspective-crops only
those quads from the original page raster, and runs the PP-OCRv6 medium
recognition model without re-detecting lines. The OAR text-line orientation
classifier corrects upside-down crops by default (disable with `--no-cls`).
Models and dictionary default to OAR's verified cache; explicit ONNX,
orientation-model and dictionary paths are supported.

```sh
pdf-redox-ocr-refine --pdf input.pdf --primary-ocr observations.json \
  --report medium-comparison.json --experimental-draft medium-hypotheses.json \
  --max-primary-score 0.97 --pages 8,10
```

Output is a comparison report and a distinct experimental v3 draft, never an
automatic rewrite of the primary PDF. The experimental draft has every line
unreviewed with confidence **zero**, including unchanged lines. Unicode NFKC
and whitespace differences are recorded as equivalent, but altered case,
digits, punctuation, and units remain distinct. No second model is loaded if
nothing qualifies for refinement. For **high-confidence errors**, pass exact
original-draft indices with repeatable `--line PAGE:INDEX` instead of editing a
reduced copy of the OCR observations; these line numbers are zero-based and are
validated against the unchanged source draft, bypassing only confidence and
minimum-length selection:

```sh
pdf-redox-ocr-refine --pdf input.pdf --primary-ocr observations.json \
  --line 28:2 --line 28:11 --report diameter-hypotheses.json \
  --experimental-draft diameter-experimental.json
```

The second-pass report schema 2 identifies `selection_mode` as `confidence`
or `explicit-lines`, and emits null selection thresholds in explicit mode.
Every experimental draft line remains unreviewed with confidence zero regardless
of selection. When searchable native fonts overlap selected pages, an ephemeral
render-only font mask is used as in primary OCR. Both CLIs share strict
page-selection and atomic output handling.

## Source-bound manual OCR quality evaluation

The native `pdf-redox-ocr-eval` command evaluates an unchanged v3 OCR draft
against manually transcribed PDF-space anchors. A human reference must contain
the original PDF SHA-256, provenance, coordinate system `PDF user space`,
and selected `{page, center: [x, y], text, kind}` samples. The command
verifies the original PDF bytes and refuses a mismatched or duplicate anchor.
The matching procedure uses actual (possibly rotated) text quadrilaterals,
a maximum-cardinality one-to-one assignment, and never matches arbitrary
strings elsewhere on a page. Missing observations count as full-length text
errors. Exact and NFKC/whitespace-insensitive character error rates use
Unicode scalar values, including technical units and CJK.

```sh
pdf-redox-ocr-eval --pdf source.pdf --draft primary.json \
  --reference inspected-labels.json --report quality.json
```

Its JSON report has per-sample predictions (including absent observations),
confidence, candidate ambiguity, error counts, overall totals and per-kind
metrics. Its `confidence_curve` records, at every observed confidence threshold,
what share of manually anchored labels would be admitted and their empirical
normalized-text error rate. Missing OCR observations never enter the admitted
set; tied confidence values are always evaluated together. This reports
actual high-confidence failures instead of treating numerical confidence as a
calibrated probability or authorization to insert text. The version-3 quality
report retains each matched observation's original zero-based `line_index` and
`line_quad` (null for unmatched references), so refinement and line audit can
address the same observation without repeating spatial matching. It also contains
a Unicode character-confusion histogram produced
by a minimum-edit alignment of the reference and OCR strings, covering
substitutions, insertions, and deletions. Each recurring confusion includes its
frequency and highest line confidence, exposing high-confidence technical-glyph
failures without installing any hard-coded text replacement rules.
The evaluator needs neither ONNX models nor a PDF rendering engine,
and never edits the input draft or source PDF. The two small, manually reviewed reference manifests are versioned in
`tests/fixtures/quality/` (86 DEFOND and 10 Walsin anchors). Original supplier
PDFs, full OCR observations and benchmark outputs remain in the separate
benchmark workspace; the manifests carry the original PDF digests and
visual-review provenance. The Notion worklog documents their precise paths,
selected-sample limitations and experimental comparisons.

## Independent glyph topology diagnostics

The native `pdf-redox-ocr-glyph-audit` research tool reports thresholded
foreground pixel counts and 4-connected *enclosed background regions* from an
original raster glyph crop. It records each region's area, centroid, bounds,
and crop SHA-256. Results across 64/128/192/224 luminance cutoffs reveal
whether apparent glyph structure is stable without asking another OCR model
to interpret the text.

```sh
pdf-redox-ocr-glyph-audit --image original-glyph-crop.png \
  --report glyph-topology.json
```

Six tiny source-PDF crops under `tests/fixtures/glyph-topology/` exercise
two diagonal-stroke diameter glyphs (two diagonally displaced holes),
three real digit zeros (one hole), and a digit eight (two vertically stacked
holes). They originate from DEFOND page 28 at 600-DPI Poppler rendering;
the precise reproducible crop coordinates and original document digest
are retained in the Notion OCR worklog. This is a **geometric measurement**,
not evidence that the pixels intrinsically mean a particular Unicode
character: cropped edges, touching neighboring strokes, raster resolution
and differing typefaces all matter. No automatic OCR replacement is made.

## Source-PDF line isolation

Use the original draft's exact page and line indices to inspect a bounded
native-resolution source region without preparing individual glyph images:

```sh
pdf-redox-ocr-line-audit --pdf source.pdf --primary-ocr draft.json \
  --line 28:2 --line 28:11 --report line-topology.json
```

The command checks the original PDF digest, renders only the specified line's
bounding region (600 DPI by default), and rectifies its ordered quadrilateral
into an oriented grayscale raster. An elongated vertical observation uses
the longer axis; this does not determine reading direction. The original
observation's luminance histogram is partitioned using Otsu's method. The
background class is estimated from unique perimeter pixels (overall class
occupancy breaks ties). The measured background is mapped to white after
orienting the foreground, while the physical luminance endpoint anchors
black. This preserves the original gray levels on white backgrounds rather
than widening antialiased strokes by stretching the foreground median. The
report retains both class medians, counts and perimeter support; uniform observations produce no ink. Explicit
`--polarity dark` and `--polarity light` override the automatic orientation.
The raster is labeled into eight-connected foreground components at the
segmentation threshold (default 192). Their horizontal interval union
reconstructs the occupied-column projection exactly. Each component retains
its area, centroid and half-open bounds; each projected group retains the
indices of its constituent components. Detached marks sharing columns can
remain grouped, while touching glyphs are not artificially split.
Background topology uses the complementary four-connected traversal.

Optional `--padding-pixels N` (default 0) expands the sampled source region
on each side within the actual page, retaining the original rectification
scale. Padding is not painted synthetically and does not affect polarity
estimation. Version-4 reports identify the original observation rectangle
inside the padded sampling rectangle, both quads, and per-group foreground
pixel counts inside and outside the original observation, plus original
boundary contact. Padding can also introduce unrelated neighboring ink:
padded-only components are retained in the full report, even when their
column extent merges otherwise separate groups. The additional
`supported_runs` projection omits only whole components with zero foreground
pixels inside the original observation; connected boundary continuations
are retained. The `supported_bands` array independently projects those same
components onto rows, then projects each band onto columns. It preserves
component indices, inside/outside pixel support, and mapped PDF quadrilaterals
at both levels. A detached horizontal rule can occupy a separate band without
joining letter runs; detached accents or dots can likewise occupy a separate
band. Neither representation establishes character identity or reading order.
These projections are geometric evidence, not a cleaned image or character assignment. Each full group reports its half-open ink/crop bounds,
original PDF-space quadrilaterals, line-edge contact at the segmentation
threshold, and topology at the selected cutoffs (64/128/192/224 by default).
Topology at a higher threshold can include additional ink at the crop edge.

This is an inspection of visible ink groups, not a character segmentation
guarantee. It does not compare group counts with recognized strings, assign
Unicode, exclude unrelated diagram strokes, or edit the input draft. The
renderer and rectified image are independently bounded by `--max-pixels`.
The report records both original and oriented line geometry.

## Correctness policy

A source-image result may replace full-page OCR only if the page is pure
source-raster paint and every visible source was recognized. In mixed pages,
additional source text requires a nonempty source identity, exact expanded
paint index and an actual isolation/differential render visibility proof;
missing provenance can never authorize a line. Invalid native-source OCR quads
or mismatched raster dimensions reject that source-only fast path and use
ordinary page OCR without aborting the document.

The complete parity gate is tracked in the Notion worklog:
https://app.notion.com/p/41bf74be902081209382e49903e8f0ff
