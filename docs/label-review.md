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
- `POST /api/cat-projector-label-review/actions` queues explicit
  `retrain_model` or `rescore_recording` records.

Household frames are never uploaded by this tool. The server accepts local file
paths or its own file tokens and rejects paths outside the configured corpus and
state roots.

## Queue Order

Cases with detector probabilities closest to `0.5` are shown first because
they are the most useful human review targets. Already saved/reviewed cases are
deprioritized. Rows from `labels.csv` keep compatibility with:

- `label_cat_present`
- `candidate_bbox_xywh`
- `label_candidate_is_cat`

Mask refs and portable mask JSON are added beside those fields instead of
replacing candidate bbox labels.

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

## Explicit Actions

The UI buttons do not retrain or rescore immediately. They create queued JSON
records with a visible log line so an operator or offline job can run the heavy
work later. This prevents the Home Assistant runtime from silently starting
model training or old-video rescoring.

Run the repo-owned active-learning iteration from this checkout when the saved
review labels should become the next detector version:

```bash
python3 scripts/cat_projector_active_learning.py --jobs 16
```

That command materializes the current review labels into
`~/.openclaw/state/cat-tv-learning/datasets/`, trains
`~/.openclaw/state/cat-tv-learning/models/cat_projector_candidate_detector_v1.cbm`,
rescores all reviewable input frames, and writes fresh uncertainty data under
`~/.openclaw/state/cat-tv-learning/label-review/rescores/`. Generated review UI
training-package copies are excluded from the next review queue, so the browser
keeps sending the operator back to original corpus frames.

The video review UI queues the same explicit actions from the active video:

- `retrain_model` means “train from the reviewed labels/masks later”;
- `rescore_recording` means “rerun the detector/renderer for this old video
  later”.

Both are status/log records, not hidden runtime jobs.

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
verifies the borderline case is first, opens the fake video frame list, saves
one cat mask label and one not-cat label, marks the whole fake video
`relabeled_ok`, and confirms retrain/rescore actions are queued records.

## Rollback

Stop the local backend service and remove the Home Assistant panel iframe or
point it back to an older static calibrator copy. Human labels and masks are
plain JSON under `~/.openclaw/state/cat-tv-learning/label-review/`; keep or move
that directory deliberately before deleting it.
