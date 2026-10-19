# Cat Projector Label Review

Cat TV Play owns the generic review UI and local backend for Sher jump-frame
annotation. House-specific Home Assistant packages and physical calibration
notes can reference this tool, but the implementation lives in this repository.

## Local Backend

Run:

```bash
python3 scripts/cat_projector_label_review_server.py --host 0.0.0.0 --port 8790
```

The server serves `web/calibration-tools/projector-wall-calibrator.html` and
the local API namespace `cat_projector_label_review`:

- `GET /api/cat-projector-label-review/videos` lists whole reviewable
  recordings/frame groups, with corpus browser metadata and saved video review
  status;
- `GET /api/cat-projector-label-review/videos/<video_id>/frames` lists the
  original input frames for that video plus the previous-model output frame
  beside each one when such an output exists;
- `GET /api/cat-projector-label-review/videos/<video_id>/timeline` returns the
  playback frame sequence, per-frame model/review metadata, suspicion reasons,
  and the next-suspect queue used by the video player;
- `POST /api/cat-projector-label-review/videos/<video_id>/status` marks a
  whole video as `relabeled_ok`, `needs_more_work`, or another explicit review
  status without modifying frame labels;
- `GET /api/cat-projector-label-review/cases` lists loose local review frames
  from `CAT_TV_LEARNING_ROOT`, repo-local `datasets/cat-tv-learning`, and
  `~/.openclaw/state/cat-tv-learning` recordings/reviews/datasets;
- `GET /api/cat-projector-label-review/file/<token>` serves only allowlisted
  local files from those corpus roots;
- `POST /api/cat-projector-label-review/segment` proxies positive/negative
  prompts to a configured local SAM service;
- `POST /api/cat-projector-label-review/labels` saves human labels, notes,
  bbox/mask metadata, source frame paths, and source video/recording paths;
- `POST /api/cat-projector-label-review/frames/<case_id>/review` saves the
  video-player review decision for one frame while preserving the old label
  fields used by training;
- `POST /api/cat-projector-label-review/actions` queues explicit
  `retrain_model` or `rescore_recording` records.
- `POST /api/cat-projector-label-review/jobs` starts one visible local
  active-learning job only when the backend was launched with
  `--allow-live-jobs`; otherwise it records a rescore/retrain/rerender request
  and clearly says nothing will run automatically. Live jobs run
  `scripts/cat_projector_active_learning.py`, which materializes reviewed
  labels, rescans reviewable frames with the configured detector, and writes
  the next uncertainty queue. The legacy detector retrains CatBoost only when
  `--detector-backend legacy` is selected.

Household frames are never uploaded by this tool. The server accepts local file
paths or its own file tokens and rejects paths outside the configured corpus and
state roots.

Set `CAT_TV_RECORDINGS_ROOT` when raw session recordings are stored outside
`~/.openclaw/state/cat-tv-learning/recordings`. The review backend adds that
explicit root to its scan and file allowlists; labels, queues, models, and other
small review state remain under `~/.openclaw/state/cat-tv-learning`.

## Video-First Review Workflow

Review mode is meant to be used like a normal annotated video player. Open the
local page, pick the tallest known jump recording on the left, and review the
big central raw-frame player. The model annotation is drawn over the frame;
previous rendered output is available only as an optional compare overlay, not
as the primary work area.

Three annotation layers must remain visible while editing: green is the manual
human mask, blue is the current model from the latest rescore, and orange is the
original model/capture-time overlay. Fresh Telegram jump alerts that have not
been rescored yet reuse the sent `jump_highlight` record from `sessions.jsonl`,
so the exact bbox/top point that triggered the message is still visible as the
original overlay on the linked frame. The current and original model layers may
coincide, but they stay separate in the API/UI so disagreements are visible.
Model overlays may carry a contour `polygon` as well as `bbox_xywh`; the UI draws
the contour without copying it into the manual mask layer unless the operator
explicitly saves a human annotation.

Use the bottom timeline to scrub. Suspicious and reviewed frames are marked, and
the left queue lists the next suspect frames with human-readable reasons. The
right inspector is only for the current frame: model belief, suspicion reason,
review state, notes, geometry tools, save status, and job status.

Review URLs can deep-link directly into a video/frame. Use
`?review_video_id=<video-id>&review_frame_label=<frame-file>` for stable links
to extracted chunk frames such as `chunk_0204_00052.jpg`; the UI also keeps the
current `review_video_id`, `review_frame`, and `review_frame_label` in the URL
as the operator moves through the review timeline.
For Telegram jump links, the backend materializes every decoded source frame
from the linked chunk without fps resampling and resolves older timestamp-based
links to the nearest decoded source frame. This keeps frame-by-frame review tied
to the original recording frames, even when the chunk has variable frame timing.
The Telegram jump bbox/top overlay is a peak-frame marker only. It must not be
drawn on neighboring source frames as if it were an original per-frame model
output. Current-model rescoring provides the per-frame overlay layer; original
capture overlays are shown only when frame-level capture metadata exists for the
selected frame.

### Native projector-source correspondence

Review and active learning use the canonical
`custom_components/cat_tv_play/source_correspondence.py` producer, not a
recording-wide source clock. For new chunks the review adapter automatically
calls `produce_chunk_correspondence` before any configured detector backend
consumes the chunk. A real local cached source must hash to the retained index's source
SHA. Select the compact index with `CAT_TV_SOURCE_INDEX` or the recording
manifest's `source_correspondence.source_index`; when exactly one retained
index has the manifest's exact source URL, the adapter selects it automatically.
Optional `source_correspondence.geometry` freezes the calibrated polygon,
excluded static regions, boundary margin and interior-mask path. No full-source
index rebuild, network-source replacement, or unrestricted native decode follows
from missing configuration or budget exhaustion.

The canonical producer also accepts an HTTP(S) source for bounded recorder
refinement without a local video copy. Beside the retained index,
`http-source-identity-<source-SHA256>.json` must bind the exact source URL to an
independently observed server-side `sha256sum`, unchanged file stat and matching
strong HTTP ETag/length before and after that hash. Capture the server evidence
through the documented, authorized source-host route; a retained index SHA or
size/mtime alone is not a source-identity certificate. This identity record does
not modify the index or its original generation keys.

The producer checks the current representation with an identity-encoding HEAD
and `If-Match`, then uses the same validator on every ffprobe/ffmpeg request.
HTTP seeking uses ranges; only automatically nominated finite intervals may be
decoded, with both an input end time and the native frame budget. The native
decode manifest and chunk provenance retain the identity-record hash. Changed
validators, redirects, missing range support or substituted identity records
leave correspondence unavailable, never a nominal/scalar fallback. HTTP support
in the canonical producer does not remove the review adapter's local-cache
requirement or establish photometry/privacy permission.

The producer retains integer PTS, rational time base, raw RGB SHA, same-decode
framehash/command identity, source/camera media SHA, geometry, code provenance,
camera-only support masks, ordered native candidate families and search coverage
in `label-review/source-correspondence/<camera-SHA>/correspondence.json`.
`correspondence_for_camera_frame` requires exact native identity equality;
unobserved frames stay unresolved. Review JPEGs need an image-SHA-bound
same-decode native certificate; old/unbound images cannot suppress candidates.
Current native review extraction emits that certificate alongside the JPEGs.
Detection uses the bound native RGB, without copying geometry between frames.

Mandatory behavior and acceptance boundaries:

1. Camera-only structural components run before CatBoost or subtraction. With
   at least three same-decode observations, dark departures from their temporal
   median use six robust noise deviations (one-code-value floor); this median
   is not a no-cat background. Single-image input retains the existing dark-body
   intensity/size rule. Support is component pixels, not bounding rectangles;
   frozen interior, screen-edge and socket exclusions apply per pixel, also in
   native gradient scoring. Components are not prey truth or privacy permission.
   Manual diagnostic boxes and source pixels are never support inputs.
2. Retained three-block spatial descriptors nominate local ordered content;
   native RGB refinement records actual PTS/hashes. A nominal index slot only
   bounds retrieval. Chunk number times segment duration, frame number divided
   by nominal FPS, fitted unit-rate clocks, modulo and endpoint clamping must
   never manufacture source identity. Legacy scalar sync files have no authority.
3. The real retained 104-camera/1,722-source-frame oracle must automatically
   recover the changing 1559-second family without manual boxes. Blank,
   decreasing, exact-reversal and the 288-family prefix/suffix compete through
   the same object-gradient evaluator. Missing positive support is a failed
   producer acceptance check, not successful all-unresolved scaffolding.
4. Repeated identities, competing families, missing content and unsearched
   source intervals remain explicit. Candidate-source brackets are **not**
   measured camera exposure; exposure mixture and global loop count stay
   unknown. Source ambiguity cannot be resolved by minimum-residual shopping.
   Adjacent discovery brackets describe searched content only. Every retained
   native alternative must have positive camera-component shape evidence above
   the independently frozen local-null envelope and belong to a complete
   non-strict ordered path above the frozen rejected-sequence envelope. Exact
   forward/backward sums qualify every state, not one argmax; totals are divided
   by the observation count before comparison with sequence means. Local-null
   failures cannot borrow another frame's score. Ties and repeated IDs survive.
   Rejected sequences stay in sequence/search evidence, not candidate families;
   unqualified but not rejected nominations remain explicit unresolved families.
5. Missing PTS, substituted RGB and stale source/geometry/implementation keys
   are rejected. Retained generation keys and raw diagnostics are preserved.
   The former whole-residual-file dependency can migrate only with the recorded
   exact unchanged geometry-function fingerprints and identical descriptor;
   a changed relevant implementation requires an authorized compatible rebuild.

Structural ranking never supplies photometric validity or privacy permission.
Unsupported source evidence preserves ordinary detection and all real-person
fail-closed gates. This repairs the recurring promotion of whole-wall similarity
or a nominal clock into object/native/privacy truth; the real-input and boundary
checks above, rather than receipt wording, are the acceptance bar.

Support-repair retro (2026-10-18): the first producer treated component boxes as
support, then rejected whole boxes crossing the socket. Stationary left-wall
shading therefore erased native observations 0/21/103 despite visible changing
interior content. The repair first inspected those actual RGB identities with
camera-only temporal masks, before rescoring any source frames. The retained
regression requires nonempty localized masks at all three identities, exclusion
integrity, and the unchanged positive/prefix/suffix/negative/review oracle.
Original failed receipts remain under `native-correspondence-producer-20261018`;
new proof is versioned under `native-correspondence-support-repair-20261018`.

Native-membership retro (2026-10-19): the previous global ordered winner was
valid family nomination, but did not qualify each of its 106 adjacent native
frames. It also retained already-rejected decreasing/288 controls as unresolved
surviving alternatives. That per-frame support claim failed acceptance; the old
artifact is never rekeyed. The executable repair records all-state score arrays
and conditional path totals, actual component pixel masks/hashes/boxes, full
camera/source native identities, and per-component local-null scores. A
component is eligible only if it matches every retained state in every surviving
family; unmatched camera proposals remain unresolved physical/projected structure.
The actual dense May decode also exposed independently crossing held-out
argmaxes: multiple observations between two discovery anchors must be completed
by one ordered dynamic program. Per-state qualification uses exact complete-path
envelopes and a separate local-positive gate on that constrained state. A camera
observation without a proposal stays unresolved; it must not erase support at
every other observation by forbidding all complete paths through the chunk.

The optional `qualification_controls` producer argument / `--qualification-controls`
CLI input accepts a `frozen-native-structural-controls-v1` receipt. Default
callers automatically discover only `<configured-index-stem>.qualification-controls.json`
next to their existing source index. The receipt binds source/index SHA256,
immutable baseline evidence path/hash, and native control manifest path/hash /
decode ID. Capture-relative entries declare `local_shape_null` and/or
`sequence_rejected`, applicable to an exact camera-media SHA and geometry SHA.
Separate `source_facts` entries of kind `source-null-native-sample-v1` bind only
source SHA and exact native manifest/identities. Every source sample's RGB hash
and absence of a native dark-body proposal are rechecked; their actual textured
gradients are re-evaluated under the current camera geometry/components. The
six retained near-white source samples are not globally constant, a no-cat
background or photometric calibration. All six native identities remain visible.
Applicable control bundles load automatically within the same source-frame
budget; no bundle label supplies a role or diagnostic-path search supplies data.

Wrong-order complete-path, fixed-content and exact-reversal null envelopes are
independently recomputed for the current capture before qualifying any state.
A nominated family's exact best complete forward path at or below that envelope
is explicit sequence rejection, not best-family/residual selection. Other
unqualified nominations remain unresolved. Sequence-only 288 negatives are not
local nulls: their middle content genuinely repeats. July capture-relative
claims never transfer to May. Source-only null samples can be re-evaluated there.
Absent sidecar/configuration is an internal evidence-wiring gap, not external
unavailable-data proof. The smallest normal-production wiring is the existing
source/index-bound source-fact receipt plus these current-capture ordering
controls, through the existing configured-index sidecar or producer argument;
no manual-label service or per-capture negative fixture is required.

`SourceCandidateFamily.qualification` carries the full camera identity, support
hash, control-receipt hash, searched/qualified coverage and every retained native
state's identity, local score, conditional total/mean and component scores.
`FrameCorrespondence.component_qualifications` retains each camera component's
pixel-mask path/hash, bbox, pixel count, status and all-state matched families.
The persisted loader round-trips these bindings; admission rejects changed
control evidence or component pixels. These are structural prerequisites only,
not independent photometry, measured exposure or privacy authorization.

The left suspect queue can switch between unreviewed suspects, all suspicious
frames, and reviewed suspects. Saved `not_cat` frames do not reappear in the
unreviewed queue. Use Reviewed suspects to audit dirty input labels: it surfaces
already saved frames that still have high segmentation height, high uncertainty,
or conflicting reviewed-vs-measured state.

The video playlist sorts by `max_jump_height_cm` descending when the value is
known. After each live retrain/rescore, `scripts/cat_projector_active_learning.py`
remeasures the fresh detection through the local wall calibration and writes
`label-review/rescores/jump_heights_latest.json`. The preferred measurement is
`mask_top_p5` from a segmentation mask; legacy bbox top is only a low-trust
debug fallback. A frame saved as `not_cat` is excluded from jump-height
aggregation even when the detector still emits a high mask on it, and stale
height summaries whose max frame was reviewed as `not_cat` are ignored by the
playlist. Older batch-review or manual measurement files are used only as
fallback metadata.

Keyboard flow:

- `Space`: play/pause the frame sequence.
- `Left` / `Right`: previous or next frame.
- `Shift+Left` / `Shift+Right`: jump about one second.
- `N`: next suspicious frame/segment.
- `G`: mask/height OK.
- `F`: no cat / false jump.
- `M`: missed cat.
- `B`: fix mask/box/top point.
- `U`: unsure.
- `Enter`: save and move to the next suspect.
- `R`: rescore/rerender, or record the request when live jobs are disabled.

## Queue Order

Cases with detector probabilities closest to `0.5` are shown first because
they are the most useful human review targets. Already saved/reviewed cases are
deprioritized. Rows from `labels.csv` keep compatibility with:

- `label_cat_present`
- `candidate_bbox_xywh`
- `label_candidate_is_cat`

Mask refs and portable mask JSON are added beside those fields instead of
replacing candidate bbox labels.

The video timeline uses the same idea but scores frame positions, not files:
unreviewed frames first, then probability near `0.5`, high uncertainty,
missing/stale model output, jump-height peaks or spikes, impossible motion,
frames near manual corrections, and videos marked `needs_more_work`. When only
probability/uncertainty exists, those fields drive the first implementation and
the response still carries structured `suspicion_reasons` hooks for richer
sources later. A missing rendered-output image is only a suspect reason when the
frame has no fresh detector/measurement metadata; segmentation rescore rows do
not need old rendered panes to count as reviewed evidence.

## Labels And Masks

Durable state is written under:

```text
~/.openclaw/state/cat-tv-learning/label-review/
  labels/<case_id>.json
  masks/<case_id>/<mask_id>.json
  actions/<action_id>.json
  videos/<video_id>.json
```

Mask JSON stores a polygon, bbox, positive prompts, negative prompts, source,
and updated timestamp. The browser can refine polygons manually by dragging
vertices, so the review flow remains usable even when no segmentation model is
installed.

Frame labels preserve existing `labels.csv`-compatible fields and add video
context rather than replacing the old schema:

- `video_id`
- `frame_index`
- `model_output_path`
- `source_video_path`
- `source_recording_dir`

The video-player review actions also add explicit fields:

- `review_decision`: `good`, `false_positive`, `missed_cat`, `bad_geometry`, or
  `unsure`;
- `cat_present`: whether Sher is visible anywhere in the frame;
- `candidate_is_cat`: whether the current model candidate is Sher;
- `geometry_status`: `ok`, `corrected`, `bad`, `missing`, or `null`;
- `review_notes`, `reviewed_at`, and `reviewer_source`.

These fields are additive. Training compatibility fields remain present:
`label_cat_present`, `candidate_bbox_xywh`, and `label_candidate_is_cat`.
For `bad_geometry` and `missed_cat` corrections, the training package uses the
corrected mask bbox as Sher's positive candidate. When the old model candidate
bbox does not intersect that corrected mask bbox, the package also emits the old
candidate as a hard negative with
`negative_reason=old_candidate_no_overlap_corrected_mask`.

This lets a relabel pass move through a whole clip while still producing
portable per-frame labels for model training.

## SAM / SAM2

Frame-level promptable segmentation uses the official local Segment Anything
service shipped in this repo. Install the optional dependencies in an isolated
environment and provide a local Meta SAM checkpoint:

```bash
python3 -m pip install '.[sam]'

CAT_PROJECTOR_SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth \
  python3 scripts/cat_projector_sam_service.py --host 127.0.0.1 --port 8766 --warmup
```

The review backend uses `http://127.0.0.1:8766/segment` by default, matching
that local service. Override it only when the service runs elsewhere:

```bash
CAT_PROJECTOR_SAM_ENDPOINT=http://127.0.0.1:8766/segment
```

The endpoint must be on localhost or a private LAN host. It should accept JSON
with `image_path`, `positive_points`, `negative_points`, and
`existing_polygon`, and returns `polygon`, `bbox_xywh`, `score`, and model
metadata.

If the endpoint is explicitly unset, or if the local SAM service is unavailable,
the UI asks the backend for the CPU click-to-contour fallback and keeps the
manual box/polygon when even that cannot produce a contour. Video mask
propagation is the next hook: persist the accepted frame mask, pass it to an
offline SAM2 video propagator, and store propagated per-frame mask refs under
the same label-review state root.

## Explicit Actions And Live Jobs

The UI buttons do not retrain or rescore immediately unless live jobs are
enabled. Without live jobs they create recorded JSON requests with a visible log
line; no worker will pick those records up automatically. This prevents the Home
Assistant runtime from silently starting model training or old-video rescoring.

Run the legacy active-learning iteration from this checkout when the saved
review labels should become the next CatBoost fallback/debug detector version:

```bash
python3 scripts/cat_projector_active_learning.py --jobs 16
```

That legacy command materializes the current review labels into
`~/.openclaw/state/cat-tv-learning/datasets/`, trains
`~/.openclaw/state/cat-tv-learning/models/cat_projector_candidate_detector_v1.cbm`,
rescores all reviewable input frames, and writes fresh uncertainty data under
`~/.openclaw/state/cat-tv-learning/label-review/rescores/`. It also remeasures
per-frame jump heights from the fresh best masks/detections and writes
`jump_heights.json` plus `jump_heights_latest.json` for the video playlist.
Generated review UI training-package copies are excluded from the next review
queue, so the browser keeps sending the operator back to original corpus frames.

For the modern path, pass a local segmentation model:

```bash
python3 scripts/cat_projector_active_learning.py \
  --detector-backend segmentation \
  --segmentation-model /path/to/sher-yolo-seg.pt \
  --confidence-threshold 0.2
```

Local live jobs inherit `CAT_PROJECTOR_SEGMENTATION_MODEL` when it is set.
Without that model path, the default `auto` backend falls back to the legacy
contrast/CatBoost detector so existing local jobs stay runnable. Use
`--detector-backend segmentation` when a missing segmentation model should be a
hard failure.

Review rescoring intentionally uses a lower detector threshold than strict
alert/eval acceptance. The UI needs to show low-confidence full-frame cats and
additional YOLO candidates, especially when a projector-plane false positive is
the most confident candidate.

The legacy contrast/CatBoost detector remains available with
`--detector-backend legacy` for fallback/debug and hard-negative mining.

The video review UI queues the same explicit actions from the active video:

- `retrain_model` means “train from the reviewed labels/masks later”;
- `rescore_recording` means “rerun the detector/renderer for this old video
  later”.

With live jobs disabled, both are status/log records, not hidden runtime jobs.

For local development only, start the backend with `--allow-live-jobs` to let
the review player start one visible background rescore/retrain/rerender job at
a time through `POST /api/cat-projector-label-review/jobs`. The UI shows
queued/running/done/failed status, a short log tail, and refreshes the current
timeline after completion. Without that flag, the same button says `Record
rescore` and only writes an explicit recorded action request.

## Meow Projector Feedback

The meow feedback extraction is separate from frame `cat/not_cat` review because
the evidence joins audio meows to projector-camera behavior. The durable label
is not a separate audio pipeline: `scripts/cat_projector_meow_feedback.py` can
write the result as normal Home Audio Mesh `acoustic_event_classifications`
facts with `--write-database`.

The training target is a single binary label: `meow: play_projector`. It is
positive only for active play evidence such as jump/paw/pounce/contact with the
projected target. It is negative when a meow-linked projector start has verified
camera coverage and the projector review found no active play highlight. That
derived outcome is recorded as `projector_started_no_active_play`; stricter
human feedback such as `not_near_projector` is also a not-play example.
Watching-only stays abstained/unlabeled for this target unless human feedback
marks it as not-play.

`training_rows.jsonl` may contain positive projector-play rows before the model
target is trainable. Promotion to a trained/deployed projector-outcome model is
blocked until the exact target has both positive and negative examples; current
positive-only active-play rows are review evidence, not a deployable classifier.

Use `--write-database --database-url postgresql:///openclaw_acoustic` for the
canonical training surface. It writes one projector outcome class per meow under
the current default acoustic class set:
`cat_meow_projector_play`, `cat_meow_projector_no_play`, or the audit bucket
`cat_meow_projector_unknown` for abstained rows, with source
`cat_projector_feedback`. The optional
`--write-sidecars` mode writes a root `cat_projector_feedback` block back to the
source meow sidecar as a compatibility/export copy and backs up each touched
sidecar under the run output directory. It must not overwrite
`sher_meow_intent`, `cat_meow_candidate.intent_hint`,
`sher_meow_intent_teacher`, or `cat_meow_intent`.

## Home Assistant Exposure

For Home Assistant, copy the UI asset:

```bash
python3 scripts/deploy_cat_projector_review_ui.py --www-root /config/www
```

Run the backend on the machine that owns the corpus, then point a
`panel_iframe` to:

```text
http://<host>:8790/calibration-tools/projector-wall-calibrator.html
```

The Darafei house package in `tasks-loop` keeps the local sidebar and systemd
wrapper in sync, but should reference this repository as the implementation
source.

## Smoke Test

Run without household data:

```bash
python3 scripts/cat_projector_label_review_server.py --fake-smoke
```

The smoke builds a tiny local fake corpus with `cat`, `not-cat`, and
`borderline` frames plus previous-model output frames, calls the HTTP API,
verifies the borderline case is first, opens the fake video timeline, saves a
missed-cat review with bbox/mask, saves a false-positive review, marks the whole
fake video `relabeled_ok`, and confirms retrain/rescore/job queue records are
visible.

## Rollback

Stop the local backend service and remove the Home Assistant panel iframe or
point it back to an older static calibrator copy. Human labels and masks are
plain JSON under `~/.openclaw/state/cat-tv-learning/label-review/`; keep or move
that directory deliberately before deleting it.
