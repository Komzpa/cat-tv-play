# Cat TV Play

Cat TV Play is a Home Assistant custom integration for projector or TV play
sessions for cats.

It starts a video/game stimulus on a media player, keeps session metadata, turns
optional recording/snapshot switches on for review, stores human or automation
observations, and measures jump height on a wall from camera frames.

The integration is intentionally generic. It does not assume a specific cat, a
specific projector, OpenClaw, Frigate, Android TV, or any private entity names.

## Status

HACS candidate. The repository layout and manifest are HACS-shaped, but this is
still a pilot until it has broader Home Assistant runtime testing.

## Features

- Config flow for a display `media_player` and optional review `camera`.
- `cat_tv_play.start_session` starts playback and records an active session.
- `cat_tv_play.stop_session` stops playback and closes the session.
- Optional recorder/snapshot switch lists are turned on during sessions.
- `cat_tv_play.record_observation` stores behaviors such as `watching`, `paw`,
  `jump`, or `left`.
- `cat_tv_play.save_calibration` stores a wall-plane homography from measured
  markers.
- `cat_tv_play.measure_image_point` converts image coordinates into wall
  centimeters and returns `jump_height_cm`.
- Source-subtraction helpers can remove the known projected clip from review
  camera frames before proposing cat candidates.
- Live eye-safety overlay tooling can black out the projected head band when a
  person is detected in the projector beam.
- Segmentation-first detector abstractions support local YOLO-style cat masks
  for jump measurement; the old contrast/CatBoost detector is retained as an
  explicit legacy fallback and hard-negative source.
- Wall-plane tracking rejects physically impossible candidate jumps before
  smoothing and extracts peak height from accepted raw detections.
- `sensor.cat_tv_play_session` exposes the active session and recent observation.

## Installation

For manual testing, copy `custom_components/cat_tv_play` into Home Assistant's
`custom_components` directory and restart Home Assistant.

For a HACS custom repository test:

1. Open HACS.
2. Add `https://github.com/Komzpa/cat-tv-play` as a custom integration
   repository.
3. Install Cat TV Play.
4. Restart Home Assistant.
5. Add the integration from **Settings -> Devices & services**.

## Basic Setup

Choose:

- `Display media player`: the TV, projector, cast target, DLNA renderer, or
  Android TV media player that can play your Cat TV URL.
- `Review camera`: a camera that sees the cat and the wall or screen.
- `Default media URL`: an absolute URL reachable by the display device.
- `Recorder switch entities`: optional camera/Frigate/NVR switches to turn on
  while the session is active.
- `Snapshot switch entities`: optional camera snapshot switches to turn on while
  the session is active.

Then call:

```yaml
service: cat_tv_play.start_session
data:
  media_url: "http://homeassistant.local:8123/local/cat-tv/bugs.mp4"
```

Stop:

```yaml
service: cat_tv_play.stop_session
data:
  reason: "cat_left"
```

Record a review note:

```yaml
service: cat_tv_play.record_observation
data:
  behavior: jump
  jump_height_cm: 156
  note: "Forepaws reached the moth target after a left-side vertical chase."
```

## Calibration

Do not measure jump height with a single pixel-per-centimeter ratio. A review
camera usually sees the wall at an angle, so the image has projective distortion.
Use points that lie in the same wall plane as the projected target.

Recommended calibration:

1. Put four visible markers in the wall plane. One-meter cardboard angle
   profiles work well because they are long, straight, cheap, and easy to see.
2. Make at least one vertical one-meter segment whose bottom touches the floor.
3. Make at least one horizontal one-meter segment, preferably crossing a known
   point on the vertical segment.
4. Save a camera snapshot.
5. Pick image coordinates for at least four known wall points.
6. Assign wall coordinates in centimeters, with `wall_y_cm = 0` at the physical
   floor and positive `wall_y_cm` upward.
7. Save the calibration with `cat_tv_play.save_calibration`.

Example from a wall with a central vertical one-meter profile and a horizontal
one-meter profile crossing its top:

```yaml
service: cat_tv_play.save_calibration
data:
  calibration_id: living_room_wall
  points:
    - image_x: 616
      image_y: 719
      wall_x_cm: 0
      wall_y_cm: 0
    - image_x: 615
      image_y: 425
      wall_x_cm: 0
      wall_y_cm: 100
    - image_x: 469
      image_y: 427
      wall_x_cm: -52
      wall_y_cm: 100
    - image_x: 755
      image_y: 416
      wall_x_cm: 48
      wall_y_cm: 100
    - image_x: 269
      image_y: 719
      wall_x_cm: -128
      wall_y_cm: 0
    - image_x: 269
      image_y: 442
      wall_x_cm: -128
      wall_y_cm: 100
```

Measure a paw point from a review frame:

```yaml
service: cat_tv_play.measure_image_point
data:
  calibration_id: living_room_wall
  image_x: 253
  image_y: 220
response_variable: paw
```

The response contains:

```yaml
wall_x_cm: -134.7
wall_y_cm: 180.1
jump_height_cm: 180.1
```

That value is height above the physical floor, not height above the bottom of
the projected image.

## Stimulus Design

Cat TV Play does not ship copyrighted videos. Use your own clips or generated
assets.

Patterns that worked well in the pilot:

- high contrast prey on a bright wall;
- one target at a time;
- short dart, pause, dart motion;
- left-side vertical climbs when the cat naturally enters from the left;
- reachable steps before high targets;
- occasional low "catch" moments so the game does not become an unreachable
  laser pointer.

For higher jumps, start below the cat's known maximum and climb in short steps:

```text
120 cm -> 140 cm -> 155 cm -> 170 cm -> occasional 180 cm
```

Keep the landing area clear and stop before the cat is tired.

## Source Subtraction

When the clip shown on the projector is known, review tooling can warp that
source frame into the review camera and subtract it. This prevents projected
prey from becoming the main candidate when the real cat is lower, occluded, or
partly outside the projected rectangle.

Use a room background from frames known not to contain the cat, or update an
adaptive background through a mask that excludes the possible cat path. Do not
average the whole active clip: a tired cat that sits still can otherwise become
part of the background and disappear from detection.

See [docs/source-subtraction.md](docs/source-subtraction.md).

## Eye-Safety Overlay

For live projector sessions, keep video playback on the projector's native
Android player and use the safety server for overlay status only. The server
still samples the source clip for projector-content filtering, but it does not
encode or serve the playable video in this mode:

```bash
python3 scripts/cat_projector_safety_overlay_server.py \
  --source-video /config/www/cat-tv/current.mp4 \
  --camera-snapshot-url http://192.168.100.39:8081/shot.jpg \
  --host 0.0.0.0 \
  --port 8787 \
  --status-only
```

Then start the projector Android activity with the raw MP4 as `video_url` and
the safety server as `overlay_status_url`. This avoids the Python/ffmpeg HLS
path and preserves smooth native playback while black zones are drawn by the
Android overlay:

```bash
adb -s 192.168.100.39:5555 shell am start \
  -a by.openclaw.catprojectorcamera.DISPLAY_RECT \
  --es video_url "http://<ha-host>:8123/local/cat-tv/current.mp4" \
  --es overlay_status_url "http://<overlay-host>:8787/status.json" \
  --ei source_width 1280 --ei source_height 720
```

The server samples the projector camera, runs the local OpenCV MobileNet SSD
person detector, maps the padded eye band of any person overlapping the
projected wall back into source-video coordinates, and publishes the black
overlay polygons. `/status.json` includes the camera-space eye band and
source-space polygon; `/debug-camera.jpg` draws the detected person box plus the
black eye band on the camera frame so deployment can verify the polygon really
covers the face area. If the detector, camera, or geometry is unavailable, the
server reports `safety_overlay_unavailable`; production deployments must keep
the independent lamp watchdog active instead of trusting the player alone.

## Jump Tracking

Review tooling should transform measurement points into wall centimeters before
filtering. The modern path is segmentation-first: a detector returns a cat mask,
`measurement.py` extracts a robust top-of-mask point, calibration maps that point
to wall centimeters, and `custom_components/cat_tv_play/tracking.py` gates
impossible motion before peak extraction. The old bbox top remains only as a
low-trust debug fallback. `custom_components/cat_tv_play/review_overlay.py` can
hold the best three jump-peak crops in a right-side review panel for annotated
clips.

See [docs/tracking.md](docs/tracking.md) and
[docs/segmentation-pipeline.md](docs/segmentation-pipeline.md).

## Label Review UI

The repository also ships a local-only Sher jump-frame review tool at
`web/calibration-tools/projector-wall-calibrator.html`. It extends the
calibration surface with a `review` mode for browsing local
`cat-tv-learning` frames, sorting uncertain detector cases first, saving
video-player review decisions (`good`, `false_positive`, `missed_cat`,
`bad_geometry`, `unsure`), editing bbox/mask annotations on the current frame,
and running or queuing explicit retrain/rescore actions.

Reviewed masks can be exported for a Sher-specific YOLO segmentation model:

```bash
python3 scripts/export_cat_projector_yolo_segmentation.py \
  --output ~/.openclaw/state/cat-tv-learning/exports/sher-yolo-seg-$(date -u +%Y%m%dT%H%M%SZ) \
  --symlink
```

Validate the reviewed masks and train/evaluate from a local YOLO segmentation
base model:

```bash
make validate-sher-yolo-seg
make train-sher-yolo-seg \
  YOLO_EXPORT=~/.openclaw/state/cat-tv-learning/exports/sher-yolo-seg-20260523T013502Z \
  YOLO_BASE_MODEL=/path/to/local/yolo11n-seg.pt \
  YOLO_MODEL=~/.openclaw/state/cat-tv-learning/models/sher-yolo-seg.pt
make eval-sher-yolo-seg \
  YOLO_EXPORT=~/.openclaw/state/cat-tv-learning/exports/sher-yolo-seg-20260523T013502Z \
  YOLO_MODEL=~/.openclaw/state/cat-tv-learning/models/sher-yolo-seg.pt
```

Then rescore with a local segmentation model:

```bash
python3 scripts/cat_projector_active_learning.py \
  --detector-backend segmentation \
  --segmentation-model /path/to/sher-yolo-seg.pt
```

For live local rescore jobs, the same model can be provided with
`CAT_PROJECTOR_SEGMENTATION_MODEL=/path/to/sher-yolo-seg.pt`. Without a
configured segmentation model, the default `auto` backend falls back to the old
contrast/CatBoost detector so existing local jobs remain runnable. Pass
`--detector-backend segmentation` when you require the modern path and want a
missing model to fail loudly; pass `--detector-backend legacy` for explicit
fallback/debug runs.

The segmentation path is offline/local and does not train inside Home Assistant
runtime.

Run the review backend locally:

```bash
python3 scripts/cat_projector_label_review_server.py --host 0.0.0.0 --port 8790 --allow-live-jobs
```

Review mode is video-first: pick a recording, use Space/arrows to play or step
the local frame sequence, follow the suspicious-frame marks on the timeline,
save the current review decision, and jump to the next suspect. Previous model
rendering is an optional compare overlay; household frames remain local.

Run the local Segment Anything service separately, with official Meta
`segment-anything` dependencies and a local checkpoint:

```bash
CAT_PROJECTOR_SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth \
  python3 scripts/cat_projector_sam_service.py --host 127.0.0.1 --port 8766 --warmup

python3 scripts/cat_projector_label_review_server.py --host 0.0.0.0 --port 8790 --allow-live-jobs
```

By default it reads `CAT_TV_LEARNING_ROOT`, a repo-local
`datasets/cat-tv-learning`, or the house `tasks-loop` dataset if present, and
writes labels/masks/actions under
`~/.openclaw/state/cat-tv-learning/label-review/`. Household frames stay on the
local machine. The review backend uses
`http://127.0.0.1:8766/segment` by default for promptable masks; set
`CAT_PROJECTOR_SAM_ENDPOINT` only to override or explicitly clear that local
endpoint. If SAM is unavailable, manual boxes/polygons still work and the UI can
request the backend's degraded click-to-contour fallback.

After a review pass, run the repo-owned offline active-learning iteration:

```bash
python3 scripts/cat_projector_active_learning.py --jobs 16
```

It trains `~/.openclaw/state/cat-tv-learning/models/cat_projector_candidate_detector_v1.cbm`,
rescores original corpus frames, and writes the next uncertainty queue under
`~/.openclaw/state/cat-tv-learning/label-review/rescores/`.

See [docs/label-review.md](docs/label-review.md).

## Example Automation

```yaml
alias: Start Cat TV when cat asks
trigger:
  - platform: event
    event_type: cat_meow_confirmed
condition:
  - condition: state
    entity_id: sensor.cat_tv_play_session
    state: idle
action:
  - service: cat_tv_play.start_session
    data:
      media_url: "http://homeassistant.local:8123/local/cat-tv/jump-ladder.mp4"
```

## Development

```bash
python3 -m pytest
python3 -m compileall custom_components tests scripts
python3 scripts/cat_projector_label_review_server.py --fake-smoke
```

## Name

The name is deliberately **Cat TV Play**, not "trainer". The goal is an
interactive play loop: show prey-like video, observe behavior, measure jumps,
and choose better play sessions.
