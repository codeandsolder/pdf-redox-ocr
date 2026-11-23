# pdf-redox-ocr: native Rust rewrite (development)

The native implementation lives in this crate. The independent legacy
Python pipeline and research scripts are frozen at Git tag
`legacy-python-reference-20261120`. The native pipeline is production; the archive
is retained solely for historical comparison.

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
The proof denominator is significant RGB contrast measured on the original and
replacement native image at the exact paint path in its resolved color context,
not the numerical distance between packed sample codes. A palette index one
step away may have high contrast; distant indices may map to the same color.
Per-line visibility audits retain exact changed packed sample tuples, native
and page RGB contrast counts and the projected expected page pixel count.
For fragmented or compositor-dependent image-only pages, the Rust path clusters actual native
image paint invocations using scale-aware, spatially indexed connected
components. Adjacent strips are recognized in bounded native-pitch PDF crops,
not as isolated text fragments. The same path handles independent singleton
groups when native RGB would omit compositing or resource color overrides.
Root page transparency groups forbid standalone raster admission for either
source decoder; mixed pages retain ordinary page OCR. Poppler renders the final paint stack, so soft
masks, ICC color, clipping and nested Form paints use the genuine PDF
compositor. Group-specific clip masks, exact affine source-axes rectification (including
arbitrary rotations, reflection and modest source shear), and one shared
inverse crop-pixel-to-PDF transform preserve provenance.
Pages are promoted only after all image paints have been reconstructed and
recognized. Failed geometric, compositing, resource or inference checks use
ordinary full-page OCR rather than the independent strip shortcut. CLI knobs
bound per-group and per-page rendered pixels, group count and native DPI.
For mixed fallback pages, compositor-dependent image footprints can also seed
bounded high-resolution regional recognition of the actual original page.
Auto mode requires more than 2.25× the ordinary page pixel density by area;
all mode requires a positive resolution gain; off mode disables this path.
Contained lower-resolution requests are skipped and the existing mosaic
pixel, DPI and group budgets bound regional work. Only qualified lines fully
inside the crop, non-overlapping accepted page/source OCR and native text,
supplement the draft. These are page observations, not independent image
source proofs. The schema-19 regional_ocr audit records crop identity, mapping,
observations and admission decisions.
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
returns typed `RasterRecognition` (pixel-space lines and repair evidence) in the original raster's pixel frame,
with no PDF placement data. The job-local `RecognitionSession` maps each
occurrence after inference. Successful results (including empty observations)
are reused across native sources, rectified mosaics, residual rasters and
ordinary fallback using a content key of SHA-256 of all RGB bytes and exact
raster dimensions. The default LRU budget retains at most 256 entries and
4 MiB of observation payload; input image buffers are not retained. Failed
inferences are never cached. `--no-recognition-cache` disables lookup and
retention for diagnostic comparison, and `PipelineResult.recognition`
reports full-raster calls, cache hits, evictions, retained payload and additional crop batches, with separate leader and boundary candidate outcomes.

All inference sessions explicitly register the CPU provider rather than relying
on ONNX Runtime's implicit CPU arena. With pinned OAR 0.10.2 and ort rc13 this
disables the per-session CPU arena; the requested thread policy, models and
exact RGB cache are unchanged. Paired release measurements of Walsin (two
pages), DEFOND (four pages) and Shenler (both recognizers with line orientation)
reduced peak RSS by 41–52% with identical draft pages and at most 3.2% longer
elapsed time. This is not a hard memory ceiling.
The experimental `--line-orientation pp-lcnet` option applies PP-LCNet half-turn classification
to the actual rectified crops before either recognizer. Only a half-turn score
of at least 0.9 triggers rotation. Revised crops and character projection retain
the selected reading direction; the audit includes classifier evidence.
Final OCR quadrilaterals use reading-frame baseline corner order when the actual
crop geometry is validated, including OAR's vertical rotation and a classified
half-turn. The same vertices are retained; raw detector and crop-repair audits
keep the original detector ordering. Native-only documents still need no model
session.

The optional `--report audit.json` writes a separate schema-19 diagnostic
report bound to the original PDF digest and model provenance. It includes
the validated policy, raw native-image observations in source-pixel
coordinates, original paint paths and native-only blockers, the per-page
routing/residual decisions, differential visibility evidence, merge
proposals, per-page residual attempt outcomes and original rejected recognition
lines/quads, exact-RGB-bound crop repair evidence and recognition reuse counters. It is not an importer draft and
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

The independent research pass is provided by `pdf-redox-ocr-refine`. It
consumes an exact-PDF-bound v3 draft and samples only the requested original
visible source line regions at 600 DPI by default. It uses the original
BL/BR/TR/TL reading-order quad, rectified with bilinear RGB sampling, and
never re-detects lines, invokes an orientation classifier, or masks native
fonts. The complete page paint stack is respected. Both outer render pixels
and rectified pixels are charged to the line and page budgets before rendering
(defaults 2M and 32M). Geometry or budget failures are explicitly retained
as skipped targets rather than substituted with another recognition crop.
The source PDF hash is verified before and after refinement.

```sh
pdf-redox-ocr-refine --pdf input.pdf --primary-ocr observations.json \
  --report medium-comparison.json --experimental-draft medium-hypotheses.json \
  --max-primary-score 0.97 --pages 8,10 --ocr-threads 6
```

Use `--recognizer small` or `--recognizer medium` (default) for greedy
hypotheses from the same official recognition model path as production.
The experimental `--recognizer consensus` records both models at
85%, 100%, 120%, and 150% horizontal source-crop width. Each separate
inference records its actual RGB digest, dimensions, greedy prediction
and CTC sequence evidence. The fixed research rule requires at least five
distinct model-and-RGB views, both models represented, and a unique winning text; each
vote must have a distinct runner-up with retained path weight at most
25% of the top hypothesis. Whitespace differences alone are ignored.
The selected literal text comes from an actual winning view, preferring
the unstretched crop. Sequence weights are bounded-search evidence,
not calibrated probabilities of correctness.

For high-confidence errors, target exact original draft indices:

```sh
pdf-redox-ocr-refine --pdf input.pdf --primary-ocr observations.json \
  --line 28:2 --line 28:11 --recognizer consensus \
  --report diameter-hypotheses.json --experimental-draft diameter-experimental.json
```

The refinement report schema 4 records crop geometry, actual sample hash,
render resolution and budgets, per-view recognition evidence and explicit
abstention or skipped targets. Literal non-whitespace changes remain
distinct (including NFKC-compatible forms such as superscripts and full-width
punctuation). The separate experimental v3 draft sets **every** line
confidence to zero and reviewed=false, including unchanged lines. This
research command never changes production OCR admission or overwrites the
input PDF/draft. No model is loaded when no eligible region is sampled.

### Offline topology comparison of saved candidates

The model-free executable `pdf-redox-ocr-candidate-rank` consumes a schema 4
refinement report and independently transcribed schema 3 source glyph banks
at the same sampling DPI. Build it with
`cargo build --release --no-default-features --features cli --bin pdf-redox-ocr-candidate-rank`.
It requires neither a PDF rendering backend nor ONNX Runtime.

```sh
pdf-redox-ocr-candidate-rank --pdf input.pdf --primary-ocr observations.json \
  --refinement-report refinement.json --donor reviewed-a.json \
  --donor reviewed-b.json --report ranking.json \
  --experimental-draft topology-draft.json
```

The report and draft must describe the same original observations (page,
line index, literal text, binary64 confidence and quad coordinates). Each
candidate binding is recomputed from conditional entry intervals and the
shared, unstretched source ink. Only actual retained alternatives are eligible:
every differing scalar must uniquely bind the same visible, unclipped ink
run in both hypotheses, with support from both recognizers. Whitespace
aliases do not contribute extra votes.

The shape rule counts enclosed background regions of at least three
normalized pixels at six fixed coverage thresholds. A count is stable only
if at least four thresholds agree. The alternative label needs at least
two distinct independently transcribed source-line hashes with the observed
count. The incumbent label also needs two independent donors, none with
the observed count. The query page and an identical normalized query crop
are excluded from all donor evidence. This is empirical compatibility with
a finite donor bank, not a universal assertion about character shapes.

Only one eligible entire alternative may win. Insertions, deletions,
unresolved geometry and conflicting alternatives abstain. The rank report
contains donor identities, measured counts, view indices, selected changes
and hashes of the artifacts used. Exact PDF provenance and consistent saved
metadata are checked; the saved report is not a cryptographic proof of OCR.
The output is another research-only draft, preserving source geometry but
clearing **all** confidence and review flags. No PDF or production admission
decision changes.

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
into the original OCR observation's classified reading-order grayscale frame.
A tall singleton retains its actual reading axis, just like recognition;
the physical page rotation and aspect ratio do not reclassify the text. The original
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

Version-9 line reports enumerate local, fully sampled edge-component
chains, including entire neighboring glyphs. Each `boundary_extensions`
entry records the proposed quad, newly added strip, measured component
indices, and overlapping OCR-line or native-text region identities.
Intersection fractions refer independently to the added strip and the
peer quad; original-only peer intersections are included with zero strip
coverage, and full proposed-quad coverage is reported separately; component intersections are bounding-quad evidence, not a
pixel ownership proof. Invalid peer geometry rejects the diagnostic
rather than silently ignoring competing observations. Each proposal also
counts actual normalized foreground sample centers in the original
observation, added strip and remainder of the same raster. Per-component
and per-peer counts retain exact identity; the peer union counts coinciding
regions only once. Peers extending outside the sample are explicitly
marked incomplete. Membership checks are bounded by the per-line
`--max-ink-checks` budget (32M by default), with explicit budget-exceeded
outcomes. These proposals do not authorize recognition, merging, or draft changes.

This is an inspection of visible ink groups, not a character segmentation
guarantee. It does not compare group counts with recognized strings, assign
Unicode, exclude unrelated diagram strokes, or edit the input draft. The
renderer and rectified image are independently bounded by `--max-pixels`.
The report records the original reading-order quad and measured padded sample geometry.

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


## End-to-end runner

The repository-root `run-auto.sh SOURCE.pdf SEARCHABLE.pdf [--ocr-only]`
executes a single native OCR binary. Its default selects the linked
pdf-redox library's `Config::optimize_only().for_processing()` preset;
`--ocr-only` as the first optional argument skips optimization. All other
arguments are passed to the native executable. No second executable,
shell staging directories, temporary draft JSON or Python is needed.
Set `PDF_REDOX_OCR_BIN` for a custom installation. Output parents must
already exist.

Direct CLI usage:

```sh
pdf-redox-ocr --pdf source.pdf --optimize --output-pdf searchable.pdf \
  --report audit.json
pdf-redox-ocr --pdf source.pdf --optimize --output-pdf searchable.pdf \
  --output draft.json --optimized-pdf prepared.pdf --report audit.json
```

Without `--optimize`, the direct CLI recognizes the original bytes.
At least one draft, searchable PDF or audit must be requested.
Retaining a draft with processing requires `--optimized-pdf`, because
its SHA and user-space geometry bind to the exact prepared PDF, not the
original source. The optimizer report and actual configuration are recorded
under `preparation` in schema-19 CLI audits, alongside both input hashes;
the existing `pdf_sha256` refers to the OCR input. Reverify the original
and prepared source hashes before applying or publishing results.
Processing is intentionally not lossless in every case (microstroke
rasterization, ICC fallback and reconstructed raster antialiasing are part
of the established processing policy). Overlay raster comparisons therefore
use the exact prepared input as their baseline. Poppler requires one scoped
temporary prepared PDF; the Rust owner removes it on exit. Use `--work-dir`
or the launcher's `PDF_REDOX_WORK_DIR` to choose a directory.


`--raster-mode auto` selects image-only and high-detail sources.
`--raster-mode all` considers all independently safe source images subject
to the same resource budgets and visibility/merge gates; fragment
reconstruction remains enabled. `--raster-mode off` disables native-source
and fragment inference, but preserves the native-text-only certificate.
`--ocr-threads N` sets ONNX intra-operation threads and inter-operation
threads to one (omitting it retains the backend's default policy).
The explicit thread count appears in model provenance.

## Comparing recognition drafts

```sh
pdf-redox-ocr-compare --pdf original.pdf --baseline python.json --candidate rust.json --report comparison.json
```

Both drafts must have the same source SHA-256 and page set. The model-free
report connects overlapping polygons into bipartite components instead of
matching line indices. Exact/normalized-text agreements, one-to-one text
differences, split/merge components and unassociated lines are listed
separately, with original line indices, text, confidence and geometry.
Native exclusion counts and ordered coordinate differences are recorded.
Threshold-qualified counts include reviewed lines, but do not claim final
overlay admission. This is a disagreement report, not ground-truth accuracy.


### Dotted layout leader evidence

Line audit schema 6 includes a `leaders` array for each original source
observation. The detector operates on actual native-resolution connected
components in the rectified padded line frame. It requires at least five
near-square, similarly sized, regularly spaced dots with at least two full
dots beyond the observation edge and genuine larger body ink inside it.
The proposed body crop preserves every other supported component; the
original observation, text, confidence and source PDF are never changed.
Use sufficient padding (e.g. 120–160 pixels at 600 DPI) so the actual leader
continuation is visible. This is diagnostic geometry, not automatic text
normalization or importer authorization.


For a targeted residual conflict, pass `--capture-residual-dir DIR` with
a new directory whose parent already exists. The observer writes the exact
ordinary and transferred treated RGB inputs as lossless PNG immediately
before attempted residual inference. This is opt-in, uses borrowed images
without changing model inputs, and performs no additional rendering.
Schema-6 pipeline audits identify the required per-page source observations
and retain rejected residual text/quads for correlation with the captures.

Source candidates are vetoed by overlap only with page observations that
qualify under the importer's automatic text/confidence gate (including
explicitly reviewed text). Low-confidence raw lines are retained for review
but cannot exclude a separately proved source label from the automatic
overlay. The audit records both qualified and raw maximum overlap. Native
text exclusion, visibility proofs and source confidence requirements are
unchanged.


### Retained-raster layout leader repair

Normal inference confirms dotted layout leader corrections using the same
PP-OCRv6 Small recognition session as the initial detection/crop pipeline.
The OAR task adapters keep the original detector configuration, quad sorting,
crop rectification, stable aspect-ratio ordering and four-region batching.
With boundary repair disabled, only observations with terminal dot-like punctuation enter bounded geometry
sampling (two line heights of padding, up to 2M sample pixels per line and
16M per raster). The same sampler and polarity estimation as the standalone
line audit provide periodic ink-chain evidence. Only the proved boundary
moves; the body must retain all other supported ink.

Recognition-only body crops run in batches of at most four, without another
detector call or PDF render. Replacement is permitted only when the genuine
prediction matches the full original body exactly after removal of proved
boundary dots/adjacent whitespace, is nonblank, and reaches both 0.97 and the
requested overlay threshold. Geometry and actual recognition scores are
retained, not invented. Failed or conflicting crops leave original
observations unchanged. Failed repair attempts prevent caching the incomplete
raster result; successful exact RGB reuse includes the confirmed repairs.

Pipeline audit schema 20 records original pixel-space observations, sample
frame, component bounds, typed crop reason, proposed quad and each actual
prediction/decision by exact RGB SHA-256 and dimensions. Full-raster inference
calls remain separate from recognition-only batch counters. Disable the stage
with `--no-leader-repair` for exact policy A/B comparisons. Native-only pages
still initialize no model. The standalone line-audit tool remains purely
diagnostic; schema 6 adds source group appearance and the normalized raster digest; original PDF coordinates are unchanged.


### Experimental boundary-ink recovery

`--boundary-repair` extends the retained-raster crop repair policy to clipped
or omitted edge ink. The geometry phase uses local connected components in
the rectified line's own band. A connected component crossing the original
edge, or a nearby detached compact mark with plausible separation from
the body, can authorize extension with a small background margin. Repeated
leaders are not extension candidates. Reject newly intersected unrelated
ink, strips already owned by other detector observations, and conflicting
extension proposals before recognizing any repaired line. Planning uses
immutable original geometry, not partially repaired lines.

The same recognition adapter and four-crop batches confirm the full expanded
crop. Only exact original-body preservation plus additional Unicode scalars
on a geometry-authorized edge are admissible, bounded separately by that
edge's source component count. Confidence must meet both 0.97 and the
overlay threshold; no score is fabricated, no specific character
substitutions are encoded. Accepted crop geometry and model confidence
replace the original; failures prevent exact-RGB result caching. The
experimental flag is off by default until broader representative validation.
All repair kinds share a 2M-pixel per-window and 16M-pixel per-raster sampling
budget, prioritizing terminal-leader candidates, and schema-19 `crop_repairs` records the original prediction,
actual geometry, type-specific evidence and independent prediction/outcome.
`recognition.crop_batches` counts actual recognition-only batches (leader and boundary batches are separate to preserve existing leader inference ordering);
leader/boundary candidate/accepted counters are independent.

DEFOND's manually transcribed p8 line 22 reference was corrected: the
final 度 starts the next line, which OCR had already recognized correctly.
The real p8 errors were the missing asterisk before "Optional mark :" and
period after "(See Table A)". Both can be recovered by source-supported
crop extension rather than dictionary corrections.


### Character-position and source-ink audit

Add `--character-audit --report audit.json` to collect the recognizer's
actual CTC emission columns on its original rectified crops. This is
diagnostic only and never replaces text, geometry or confidence. The OAR
adapter exposes these positions when word-box reporting is enabled;
no second model, additional recognition pass, or PDF rendering is needed.
The decoder normalizes against the **batch tensor width**, which includes
right padding. Recover per-crop pixel positions with the pinned adapter's
base-width minimum, height-48 ceil-resizing, batch-width truncation and
3200-pixel width cap. A timestep is represented by its column center,
not an exact glyph center; emissions in padding remain unclamped and
receive no candidate glyphs.

The same crop pixels are independently segmented into connected ink and
column projections, with light/dark polarity estimation and enclosed-hole
topology at three thresholds. Neighboring nonspace CTC peaks define
approximate reading-axis cells; each anchor records all intersecting ink
runs. Multiple characters may share a connected run, a character may span
several runs, and a blank or padding emission may have none. This is
evidence for further alignment, not a forced one-to-one assignment.
Competing anchors are partitioned by shared source runs. Within each
component, greedy earliest and latest strictly increasing injective
assignments identify pointwise unique correspondences, ambiguous ranges,
or impossible assignments when glyphs are joined. One unsolvable cluster
does not discard independent matches elsewhere on the line. Unmatched,
whitespace and padding emissions have explicit statuses.

Schema-11 `character_audits` groups data by exact parent RGB SHA-256 and
dimensions. Each primary detector entry includes the original line/quad,
sorted detector index (not final draft index), exact rectified RGB crop
hash/dimensions, padding geometry, raw CTC column indices, projected ink
and alignment candidates. Component indices address the entry's component
array. Positions and ink bounds are in the **rectified recognition-crop
frame**. Schema 20 additionally records a reconstructible projective crop
mapping for actual detector quadrilaterals, verified against returned crop
dimensions and the OAR rotation contract. The `raster_character_mappings`
field binds whole-page, regional, native-image and reconstructed mosaic
occurrences to exact SHA-bound PDF transforms. Native-image scopes identify
the paint index; mosaic scopes identify the group index. Supplemental
observations retain their original detector identities through proofs and
admission; only admitted IDs may supply character evidence. Failed mosaic
attempts retain raw audit evidence but never publish accepted mappings.
This permits reviewed glyph sampling without assuming reading order equals
final draft order.
Audit pixel work is bounded to 16M sampled crop pixels per raster; skipped
or invalid observations are counted. Cache hits reuse the same original
raster audit, and enabling the diagnostic requires an explicit audit output.
Existing geometry-supported crop repairs remain independent.

Use `--ctc-audit --report audit.json` to additionally inspect actual CTC class
probabilities (including blank) at the primary greedy emission columns and
five competing sequence hypotheses. This opts into full model output from
the **same** recognition inference, not a second recognition or a different
crop. The bounded prefix search retains 12 prefixes and considers all
transitions into retained prefixes. For new children, the strongest 14
classes per column suffice to reproduce a full-vocabulary beam step
(blank and the repeated last class are the only exceptional transitions).
Subnormal products fall back to bounded full-vocabulary selection to preserve
the deterministic tie break. The audit displays only the top 8 classes
plus blank at each greedy emission. Its reported log mass is a lower bound
on CTC path mass, **not** a calibrated character or sequence correctness
probability. It never changes the greedy prediction, line confidence,
confirmation policy or OCR admission. It also captures alternative-model
evidence when confirmation is requested. Each retained class sequence is
additionally aligned using the full original
probability tensor and a log-domain forward/backward automaton. It reports
the exact unpruned sequence mass and conditional token entry modes and 95%
equal-tail intervals (not label correctness or glyph boxes). Refinement
records a shared source-ink segmentation and candidate-specific geometric
associations across all horizontal views. Ambiguous joins, clipping and
entry intervals extending into padding are not proof of a character.
Ordinary recognition retains OAR's compact output path; full CTC diagnostics
temporarily allocate the
owned class tensor and require an explicit audit output.


Manual-reference files remain separate by selection: the original Walsin
p5 technical/graph sample has ten entries; the new
`tests/fixtures/quality/walsin-taping-groundtruth.json` has three
visually verified p3 taping-specification entries. The latter exposes six
diameter-symbol errors on the current draft (five Ø/0, one Ø/∅). It is a
purposefully selected diagnostic, not a population error estimate. The
slashed-circle glyph is transcribed as Ø according to the established
reference convention; rendered shape alone cannot prove original Unicode.


### Exact-source vector catalog evidence

Recognize the prepared PDF with its existing, exact optimizer catalog:

```sh
pdf-redox-ocr --pdf prepared.pdf --source-glyph-catalog catalog.json \
  --output draft.json --report audit.json
```

Alternatively collect it in the same optimization transaction:

```sh
./run-auto.sh original.pdf searchable.pdf --optimized-pdf prepared.pdf \
  --output-glyph-catalog catalog.json --report audit.json
```

The input catalog must have schema 2 and match the exact input PDF hash.
Generated catalogs are bound to the retained optimized PDF, not to the
unoptimized source or searchable overlay output. Both modes implicitly
enable the character audit. Full-page primary ink runs are mapped into
PDF space with the actual crop geometry and raster occurrence transform.
A source component must have substantial reciprocal polygon coverage and
be unique among overlapping components. Multiple partial components,
joined ink and ambiguous CTC associations cannot silently become a label.
Only the actual accepted ordinary/residual page route and explicitly
admitted regional detections supply votes. Ambiguous repeated executions of
the same RGB raster supply no evidence.
Exact template / page / occurrence ordinal deduplicates repeated detector
observations; conflicting observations of the same occurrence are excluded.
Summary candidates require three agreeing high-line-score observations on
at least two pages. Scores in these records are actual **line** scores,
not per-character probabilities, and repeated predictions are not
independent confirmation. The report never introduces a Unicode mapping
into the source PDF by itself. Native image and mosaic mappings support reviewed glyph sampling but never
supply vector catalog votes: raster paint is not evidence of a vector outline.

`--recognizer medium` selects PP-OCRv6 Medium recognition while preserving
the same Small detector, cropping, layout policy and overlay admission.
The default remains Small. Actual model provenance is retained in drafts
and reports. Models load lazily; native-only PDFs do not need model files.


### Experimental selective second recognizer

Use `--confirm-with medium --report audit.json` with the default Small
recognizer (or `--recognizer medium --confirm-with small`). The second
recognizer is initialized lazily only when an unchanged, below-overlay-threshold
line has a retained original crop. No additional detection or PDF rendering is
needed. Existing eligible lines and layout-repaired lines are untouched.
Unless a full character audit or catalog is requested, only the candidate
crops receive source-ink analysis.

The two actual CTC predictions are aligned independently against the same
source ink. Admission requires the full second-model score to reach both 0.97
and the requested overlay threshold, unchanged source-run coverage and reading
order, and shared text. Non-equivalent substitutions preserve cluster scalar
counts and the individual changed scalars' resolved ink support, including
joined glyph clusters. Expansion of individual support into adjacent ink is
rejected even if aggregate cluster coverage agrees. No dictionary correction,
fabricated score or geometry change is used. The second classifier remains
fallible: matching ink support is not a proof of a character's identity.

Schema-11 character audits retain candidate originals, actual second-model
text/score/CTC anchors and typed decisions, including failures.
`confirmation_attempts`, `confirmation_accepted`, `confirmation_batches` and
`confirmation_failures` distinguish this stage from crop repair and full-raster
inference. Incomplete attempts prevent exact-RGB result caching.
The mode remains opt-in; native-only pages load neither recognizer.

### Independently reviewed glyph appearance

Export actual primary recognition ink shapes using a source-bound manual
reference file:

```sh
pdf-redox-ocr --pdf prepared.pdf --pages 28 \
  --glyph-reference tests/fixtures/quality/defond-manual-groundtruth.json \
  --glyph-samples reviewed28.json --output draft.json
```

The reference must describe the exact recognition PDF. If processing is
enabled, the prepared PDF must be retained. Unique raw detector lines
are located by the reviewed source-space anchor; ambiguous occurrences
are excluded. All minimum edit-distance alignments must agree on the
reviewed label of an emitted scalar, and the original CTC/ink mapping
must independently identify a unique unclipped ink group. Whitespace is
excluded from scalar alignment without case folding. Each exported
sample retains its source PDF, scoped raster occurrence and crop identity, actual prediction,
line score, page, ink geometry and 24x24 grayscale ink coverage together
with the original dimensions. Schema 3 distinguishes raster-local detector
identities across regional crops on the same page. No reference text is applied to the draft.

Compare held-out pages using the model-free matcher:

```sh
pdf-redox-ocr-glyph-match --query reviewed28.json \
  --training reviewed29.json --page 28 --report match.json \
  --max-distance 0.16 --min-margin 0.025 --min-independent-crops 2
```

The matcher excludes the entire queried PDF page and identical crop
hashes across every donor bank, compares distinct labels and preserves
the nearest and competing occurrences. Its distances are not OCR
probabilities. The thresholds above describe an exploratory evaluation,
not a production policy. The manually reviewed page-29 donor fixture is
independent from the page-28 evaluation. On 299 held-out DEFOND page-28
glyphs, cross-font matching admitted 116 candidates (106 correct, ten
wrong substitutions, no corrected primary errors). Page-29-only donors
admitted nine candidates (three correct, six wrong substitutions); both
diameter queries ranked the reviewed label first but failed the distance
limit. Raw shape matching is diagnostic only; its evidence does not
authorize Unicode substitution or overlay admission.

For a controlled high-resolution appearance comparison, specify
--glyph-source-dpi 600 during each bank extraction and
--appearance source in the matching command.
Each reviewed line is independently re-rendered from the unchanged PDF,
using its actual primary reading frame. Source shapes are measured only
for isolated, unclipped groups with substantial reciprocal geometric
coverage and a unique association in both directions, including all
unlabeled competitors. Each shape retains its source group index, exact
normalized line raster digest/dimensions and mapped ink quad. Sampling
failures and unmatched shapes remain explicit. Source-mode comparison
requires equal sampling DPI, excludes identical original and source line
crops, counts distinct source raster digests, and never falls back to
primary appearances. This remains a held-out diagnostic, not a correction
policy or new Unicode attribution.


### Direct native-image decoding experiment

Run `--native-image-decoder hayro --report audit.json` to use the pinned
Hayro interpreter with the already parsed PDF, rather than extracting
all images on a page through `pdfimages`. The default remains Poppler
for controlled A/B comparisons.

The direct path only decodes requested indirect image sources with
matched nested Do paths, native dimensions and image placement transforms. It independently
resolves every occurrence and reuses decoded pixels only for an exact
previously verified interpreter decoding context. Distinct RGB appearances of
the same object receive separate recognition identities; byte-identical
appearances share a raster. Compositing-dependent and over-budget paints
fall back individually. Unmatched or ambiguous interpreter paths reject the
affected page, not independently verified appearances on other pages.
Default device-color-space overrides are not supported by the pinned
interpreter; these paints require the existing page fallback. A rejected
paint is not retried through object-ID-only extraction. Source visibility
proofs, source-local coordinates, OCR recognition, cache and overlay gates
remain unchanged.

Schema-15 `source_rasters` and `source_observations` contain a `variants` map
keyed by document-local appearance ID (each entry holds `source_id` and
`value`), and a `paint_bindings` map indexed by page and paint index. RGB
SHA256 and native dimensions identify each raster; visibility audits identify
the exact appearance. Each paint records its native packed-sample layout.
Named XObject color-space aliases require actual differential visibility proof,
since renderers may disagree on these nonstandard definitions; object-only
Poppler extraction skips them. Counterfactual masks retain the original
color-space reference and exact Form path, with cache identity including
appearance and resolved sample component count.
`direct_decode` records matched, accepted, decoded and reused occurrences,
accepted sources and appearances, pixel counts, and per-paint/page rejections.
The resident source count and aggregate pixel ceilings apply to distinct
appearances. The
existing extraction ceiling limits attempted target pixels per page in
the direct path, not unrelated images. It is a decoded-pixel policy, not
a hard bound on interpreter or compressed-stream allocation.

### Source fragment reconciliation research

`pdf-redox-ocr-fragment-reconcile --pdf FILE --primary-ocr DRAFT --line PAGE:INDEX --report REPORT --experimental-draft RESEARCH`
evaluates bounded source edge extensions and source-supported stroke contractions
with the Small+Medium four-stretch CTC recognizer.
By default the source sampling window scales with the measured line height:
longitudinal margin covers the maximum extension distance and its background
margin, while each orthogonal margin covers one glyph height. An explicit
`--padding-pixels` replaces this with a uniform diagnostic margin. Actual
within-page sampled bounds are recorded per line.
The explicit line-audit command also reports source-edge contraction geometry
for a long connected stroke with predominantly external foreground. The
entire in-observation tip must be a sustained stripe: contiguous column
cross-sections sharing a row, contained in an equally long immediately
adjacent exterior stripe. The cut preserves all other actual in-box
component pixels, even at a shared half-open boundary. This refuses an
attached genuine glyph whose cross-section deviates from the external
stroke. For research reconciliation, the unchanged and contracted
recognition crops are exact subimages of the same normalized source raster.
The contraction must conserve all remaining foreground and produce a literal
edge deletion supported by both models. Original CTC hypotheses are remapped
to an ordered partition of the measured exterior stroke tips and retained
runs; all removed runs must correspond to deleted tokens and every surviving
token must preserve its run identity in the contracted recognition. This
works even when the two partitions meet at a half-open boundary without
a blank column. A contraction and an expansion compete in the same atomic
transaction resolution; neither can supersede the other. Geometry diagnostics
alone never authorize a deletion. The same connected traversal records exact
top/bottom continuation stripes in the transposed reading frame, with
actual source bounds and per-component pixel counts. These can identify
perpendicular drawing/witness lines inside a text crop without assuming
they are separable at a longitudinal edge. Orthogonal stroke evidence includes every exact in-box cross-section.
Recognition additionally measures a counterfactual that replaces only those
thresholded pixels by the original background-class luminance in RGB and
white in the shared normalized raster. Each component is counted once even
if both entry edges prove it. The report binds the mask and paired original
and masked recognitions by hashes, pixel counts and source spans, preserving
all non-mask samples. The original recognition is shared with contractions.
A masked text-only correction requires unique retained-run assignment in
both models, compatible localization of the very same revised class sequence
before masking, and increased full conditional CTC log odds versus the same
primary class sequence. Original and masked crops of identical dimensions share
a batch for each model and stretch. Missing comparison sequences are
scored exactly from the same complete inference distributions, independently
of beam rank, and serialized as explicit queries with their own token mapping.
Queries never participate in recognition consensus. An ordered token assignment must preserve the exact
transcription of every unaffected source group. Only groups with measured
removed foreground may change their labels; split groups retain a common
original identity. Each accepted vote records the path-mass comparison and
an explicit compatible ordered assignment. These text-only operations
participate in the existing atomic conflict resolver. The output remains
research-only and non-importable.

Nondeleting text revisions inside the original observation are handled
separately from crop expansion and stroke removal. The exact occupied-column
projection preselects observations in which an injective revised transcription
is possible. Both models must support the same revised run assignment.
A bounded ordered minimum-cost embedding consumes every original token at a
compatible source run and every revised token in reading order. Literal
retention and insertion have zero cost; a substitution costs one. The number
of insertions is fixed by the revised length. This permits substitutions,
insertions, and mixed edits without inventing deletions or arbitrary old-token
identities. Each view records all possible and required insertion/substitution
positions across minimum-cost complete embeddings; their union and intersection
are retained across accepted views. Every possibly edited run must have independent ink which is either unclipped
or independently proved complete by full padded-sample connected components.
A whole-component partition fully inside a run and the original observation,
with zero outside pixels, no sampled-image edge contact and an exact matching
foreground pixel sum, proves that a crop-edge-touching run has no continuation.
The report retains the component IDs and exact pixels for each such accepted
run. Exact peer accounting still rejects overlapping OCR/native text. Explicit original CTC queries score even absent beam sequences;
the consensus winner must have higher exact mass in each accepted view.
Every text-only correction enters the same atomic conflict resolver and keeps
unchanged original geometry.

The observation band can widen orthogonally only to fully measured,
locally supported components with majority foreground and vertical extent
inside the original band. Their measured height is not capped by the
detected height. Re-measure the expanded band and reject unrelated newly included body
ink; original, longitudinal additions, orthogonal additions and outside
foreground are counted separately. Added tokens must bind to longitudinal
ink, not to repaired original glyph tops or bottoms. Competing peers in
the orthogonal addition cause abstention.
When a measured local extension reaches a separate OCR peer and the peer
exceeds the normal search horizon, a single bounded retry expands the common
sample enough to measure that peer in full. Candidate chains must still obey
local component/gap/obstruction constraints; additional peers cannot recursively
extend the horizon. The ordinary sample is retained unless the retry can account
for a previously incomplete peer, preserves all already complete peer evidence,
and fits the same page and ink-check budgets. The report records the retry plan
and total charged ink checks. Explicit diagnostic padding disables this retry.
Every expanded recognition window is an exact
subimage of that raster, using the original observation's measured luminance normalization.
Recognition runs are independently counted in the common raster, including how much
original and added foreground each contains and each peer's actual foreground count.
Conservation is checked against the connected-component audit.

All added non-whitespace tokens must have a stable, ordered unique assignment to runs
containing actual added foreground in both recognizer models. Existing tokens are
constrained to runs containing original foreground. A consumed neighboring observation
must have its added foreground accounted for by those bindings. A partially captured
peer additionally requires stable independently localized literal text inside the new
observation, and cannot have a shared component cut by the replacement boundary.
Native-text conflicts and incomplete peer samples abstain.

The schema 16 report retains the original observation identity, all considered extensions and contractions,
conditional CTC evidence, common-raster run counts, peer evidence, transactions and
conflicts. Strictly nested candidates can supersede a shorter candidate only if all added
foreground is bound to tokens, the larger component chain and crop contain the
smaller, every preserved token has identical common-raster run bounds and pixel
support, and peer claims are unchanged. The report records every strict
dominator. Remaining operations sharing observations or overlapping replacement
geometry are blocked symmetrically; dominated options do not revive on conflict. The output is a **non-importable schema 0 draft** with every
confidence zero and no reviewed flags. This is an offline diagnostic, not automatic
overlay admission. The combined page pixel budget covers padded source renders,
rectification, variant crops and separately sampled peers. The ink-check budget is
shared across component and run measurements for each original line.
