# Segmentation-First Cat Jump Measurement

Cat Projector review is a measurement system, not just a cat/no-cat detector.
The primary runtime path should produce a cat instance mask and a calibrated
measurement point. Legacy contrast components and CatBoost candidate scoring are
kept for fallback, debugging, and hard-negative mining only.

## Runtime Pipeline

```text
camera frame
  -> optional source-video subtraction/debug layers
  -> CatDetector backend
  -> CatDetection(mask, bbox, score, model id)
  -> robust mask-top measurement point
  -> wall-plane homography, centimeters
  -> physics gate / Kalman tracker
  -> jump peak/event extraction
  -> review queue and retraining exports
```

The first modern backend is `UltralyticsSegmentationDetector`, configured with a
local YOLO segmentation model path. It consumes instance masks. Bboxes are kept
as crop/debug helpers and as a low-trust legacy fallback, not as measurement
truth.

## Detector Backends

`custom_components/cat_tv_play/detection.py` defines:

- `CatDetector.detect(frame, context) -> list[CatDetection]`
- `CatDetection`: bbox, score, optional mask, source/model id, frame/time, debug
  metadata.
- `UltralyticsSegmentationDetector`: YOLO segmentation backend. It fails cleanly
  if `ultralytics` or the configured weights are absent.
- `FakeSegmentationDetector`: deterministic mask backend for tests and smoke
  checks. It never downloads weights.
- `LegacyContrastDetector`: wraps the old contrast/components + CatBoost scorer.
  Use it explicitly for fallback/debug and hard-negative mining. When source
  subtraction supplies residual components, the legacy path carries the
  component mask through `CatDetection`; CatBoost still ranks the candidate, but
  measurement uses the residual contour instead of the bbox top.

Active-learning rescoring accepts:

```bash
python3 scripts/cat_projector_active_learning.py \
  --detector-backend segmentation \
  --segmentation-model /path/to/sher-yolo-seg.pt \
  --device cuda:0 \
  --confidence-threshold 0.5
```

If `--detector-backend auto` is used with `--segmentation-model` or
`CAT_PROJECTOR_SEGMENTATION_MODEL`, the segmentation backend is selected.
Without a segmentation model, auto falls back to the old contrast/CatBoost path
so existing local jobs remain runnable. Use `--detector-backend segmentation`
when a missing YOLO model should fail loudly, or `--detector-backend legacy`
when you explicitly want fallback/debug or hard-negative mining.

## Measurement Points

`custom_components/cat_tv_play/measurement.py` defines:

- `MeasurementPoint`: point type, image xy, optional wall xy/cm, confidence,
  uncertainty, source, debug metadata.
- `JumpMeasurement`: event id, provisional peak, source point type,
  confidence/trust flags, debug metadata.
- `mask_top_measurement_point`: robust top-of-mask extraction.
- `legacy_bbox_top_measurement_point`: low-trust debug fallback.

Mask top is not a single minimum-y pixel. It uses a small top band/percentile and
the median x inside that band, so one noisy mask pixel or whisker does not create
a false record. The selected point is then transformed through the existing
wall-plane homography.

Probe rows now include both:

- `best_measurement_point`, `measurement_source`, `best_top_height_cm`
- `best_has_mask` and `best_mask_polygon` when the detector produced a contour
- `legacy_bbox_top_height_cm` for comparison/debug
- `tracker_status`, `tracker_reason`, `tracker_confirmed`, and
  `tracker_height_cm` from the wall-plane physics gate

Review overlays should make the measurement source visible. Model overlays may
carry both `polygon` and `bbox_xywh`; the polygon is the evidence layer when
present, while the bbox is retained for compatibility and labels. A mask-based
point is the trusted path; bbox-top is a warning/debug fallback.

## Dataset Export

Reviewed local masks can be exported to YOLO segmentation format:

```bash
python3 scripts/export_cat_projector_yolo_segmentation.py \
  --output ~/.openclaw/state/cat-tv-learning/exports/sher-yolo-seg-$(date -u +%Y%m%dT%H%M%SZ) \
  --symlink
```

Validate without writing:

```bash
python3 scripts/export_cat_projector_yolo_segmentation.py --validate-only
```

The exporter reads `label-review/labels/*.json`, loads inline masks plus
`mask_refs` sidecars, uses saved mask polygons for positive cat examples, writes
`images/{train,val}`, `labels/{train,val}`, `dataset.yaml`, and `manifest.json`.
The manifest includes session/frame/positive/negative counts, image sizes,
split-key sources, skipped labels, mask refs, image hashes, polygon area, and a
deterministic dataset hash. Cat labels without usable masks are warnings, not
silent omissions.

Splits are by recording/session key, not by random frame, so adjacent
near-duplicate frames do not leak between train and validation.

Hard negatives are exported as empty YOLO label files and should also remain in
the reviewed label store / legacy hard-negative packages:

- projected prey;
- shadows;
- empty wall;
- table/stool/edge artifacts;
- old non-overlapping model boxes from corrected masks.

YOLO segmentation positive masks and legacy hard negatives are complementary:
the former trains the modern detector, the latter keeps the old detector useful
as a disagreement source.

Current real export evidence from May 23, 2026:

- export dir:
  `~/.openclaw/state/cat-tv-learning/exports/sher-yolo-seg-20260523T013502Z`
- dataset hash:
  `b2560fbe2fb256d991644acc0c362c9d2ff2e0aa0e4698e9be1cc03afb0369ea`
- frames: `463`; positives: `252`; hard negatives: `211`;
  train/val: `365/98`; sessions: `79`; warnings: `18`; hard errors: `0`.

## Training And Evaluation

Training is explicit and local. The normal command refuses to download a base
model; point it at a local YOLO segmentation `.pt`:

```bash
make train-sher-yolo-seg \
  YOLO_EXPORT=~/.openclaw/state/cat-tv-learning/exports/sher-yolo-seg-20260523T013502Z \
  YOLO_BASE_MODEL=/path/to/local/yolo11n-seg.pt \
  YOLO_MODEL=~/.openclaw/state/cat-tv-learning/models/sher-yolo-seg.pt \
  YOLO_DEVICE=cuda:0
```

Equivalent direct command:

```bash
python3 scripts/cat_projector_yolo_segmentation.py train \
  --dataset ~/.openclaw/state/cat-tv-learning/exports/sher-yolo-seg-20260523T013502Z/dataset.yaml \
  --base-model /path/to/local/yolo11n-seg.pt \
  --out ~/.openclaw/state/cat-tv-learning/models/sher-yolo-seg.pt \
  --epochs 80 --imgsz 960 --batch 8 --device cuda:0
```

Each run writes command/config, git state, dataset hash/summary, base model hash,
metrics, and model path under `~/.openclaw/state/cat-tv-learning/yolo-runs/`.

Evaluate and generate visual error reports:

```bash
python3 scripts/cat_projector_yolo_segmentation.py eval \
  --dataset ~/.openclaw/state/cat-tv-learning/exports/sher-yolo-seg-20260523T013502Z/dataset.yaml \
  --model ~/.openclaw/state/cat-tv-learning/models/sher-yolo-seg.pt \
  --out ~/.openclaw/state/cat-tv-learning/evals/sher-yolo-seg-$(date -u +%Y%m%dT%H%M%SZ)
```

The eval report records cat-presence precision/recall, false positives on hard
negatives, mask IoU, top-of-mask pixel error, and overlay buckets for false
positives, false negatives, low-confidence true positives, and other nontrivial
prediction cases. Do not trust live jump height until segmentation beats the
legacy baseline on hard negatives and the reviewed val split has acceptable
presence recall plus mask IoU.

## Active Learning Priority

The review queue should prioritize frames that can change jump records:

- high or record-breaking mask measurements;
- low-confidence measurements near apex;
- detector/tracker disagreement;
- physically rejected high measurements;
- source-subtraction conflict;
- segmentation-vs-legacy disagreement;
- corrected masks adjacent to peak frames.

Classifier probability near `0.5` is still useful, but it is no longer the center
of the system.

## Home Assistant Runtime

Do not train large models in Home Assistant runtime. Runtime should load a local
model/backend through the detector interface or consume model outputs generated
by offline jobs. Training/exporting is an offline review operation.
