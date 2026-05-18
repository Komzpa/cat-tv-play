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

See [docs/source-subtraction.md](docs/source-subtraction.md).

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
```

## Name

The name is deliberately **Cat TV Play**, not "trainer". The goal is an
interactive play loop: show prey-like video, observe behavior, measure jumps,
and choose better play sessions.
