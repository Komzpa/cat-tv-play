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

- `GET /api/cat-projector-label-review/cases` lists local review frames from
  `CAT_TV_LEARNING_ROOT`, repo-local `datasets/cat-tv-learning`, and
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
```

Mask JSON stores a polygon, bbox, positive prompts, negative prompts, source,
and updated timestamp. The browser can refine polygons manually by dragging
vertices, so the review flow remains usable even when no segmentation model is
installed.

## SAM / SAM2

Frame-level promptable segmentation uses the official local Segment Anything
service shipped in this repo. Install the optional dependencies in an isolated
environment and provide a local Meta SAM checkpoint:

```bash
python3 -m pip install '.[sam]'

CAT_PROJECTOR_SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth \
  python3 scripts/cat_projector_sam_service.py --host 127.0.0.1 --port 8766 --warmup
```

Then point the review backend at it:

```bash
CAT_PROJECTOR_SAM_ENDPOINT=http://127.0.0.1:8766/segment
```

The endpoint must be on localhost or a private LAN host. It should accept JSON
with `image_path`, `positive_points`, `negative_points`, and
`existing_polygon`, and returns `polygon`, `bbox_xywh`, `score`, and model
metadata.

If the endpoint is not configured, or if it fails, automatic segmentation fails
visibly and the UI keeps the manual box/polygon. The backend still has a CPU
click-to-contour helper for fake smoke tests and explicit degraded calls, but it
is not presented as SAM in the review UI. Video mask propagation is the next
hook: persist the accepted frame mask, pass it to an offline SAM2 video
propagator, and store propagated per-frame mask refs under the same
label-review state root.

## Explicit Actions

The UI buttons do not retrain or rescore immediately. They create queued JSON
records with a visible log line so an operator or offline job can run the heavy
work later. This prevents the Home Assistant runtime from silently starting
model training or old-video rescoring.

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
`borderline` frames, calls the HTTP API, verifies the borderline case is first,
saves one cat mask label and one not-cat label, and confirms retrain/rescore
actions are queued records.

## Rollback

Stop the local backend service and remove the Home Assistant panel iframe or
point it back to an older static calibrator copy. Human labels and masks are
plain JSON under `~/.openclaw/state/cat-tv-learning/label-review/`; keep or move
that directory deliberately before deleting it.
