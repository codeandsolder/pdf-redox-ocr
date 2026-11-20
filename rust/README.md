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
The optional `--report audit.json` writes a separate schema-7 diagnostic
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
under `preparation` in schema-7 CLI audits, alongside both input hashes;
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

Line audit schema 5 includes a `leaders` array for each original source
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

Pipeline audit schema 7 records original pixel-space observations, sample
frame, component bounds, typed crop reason, proposed quad and each actual
prediction/decision by exact RGB SHA-256 and dimensions. Full-raster inference
calls remain separate from recognition-only batch counters. Disable the stage
with `--no-leader-repair` for exact policy A/B comparisons. Native-only pages
still initialize no model. The standalone line-audit tool remains purely
diagnostic; its schema 5 and original PDF coordinate contract are unchanged.


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
budget, prioritizing terminal-leader candidates, and schema-7 `crop_repairs` records the original prediction,
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

Schema-7 `character_audits` groups data by exact parent RGB SHA-256 and
dimensions. Each primary detector entry includes the original line/quad,
sorted detector index (not final draft index), exact rectified RGB crop
hash/dimensions, padding geometry, raw CTC column indices, projected ink
and alignment candidates. Component indices address the entry's component
array. Positions and ink bounds are in the **rectified recognition-crop
frame**, not PDF user space; perspective/rotation mapping is not inferred.
Audit pixel work is bounded to 16M sampled crop pixels per raster; skipped
or invalid observations are counted. Cache hits reuse the same original
raster audit, and enabling the diagnostic requires an explicit audit output.
Existing geometry-supported crop repairs remain independent.
