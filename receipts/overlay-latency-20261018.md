# Overlay latency trace receipt — 2026-10-18

Read-only trace. No restarts, config edits, or physical changes. Host under test is
nucat (`192.168.100.74`); HA core at `192.168.100.128:8123`; projector Android
device (camera publisher + renderer) at `192.168.100.39`.

## Topology (confirmed from live process/unit/config inspection)

```
projector Android device .39
  camera.catprojector.publisher app
    http://192.168.100.39:8081/video?fps=25   (MJPEG producer, 25 fps)
    http://192.168.100.39:8081/shot.jpg       (snapshot endpoint)
        |
        +--> go2rtc (nucat :1984, stream `cat_projector`, mpjpeg producer)
        |       +--> RTSP restream nucat:8554/cat_projector
        |               +--> Frigate camera `Cat_projector_camera`
        |                       detect 1280x720 @ 5 fps, record+snapshots on,
        |                       live stream = main_stream `cat_projector`
        |                       (Frigate /api/Cat_projector_camera/latest.jpg,
        |                        HA camera.cat_projector_camera, Frigate live view)
        |
        +--> safety overlay service (nucat user unit
              cat-projector-safety-overlay.service, PID 5136,
              scripts/cat_projector_safety_overlay_server.py --status-only)
              --camera-snapshot-url http://192.168.100.39:8081/shot.jpg
              => overlay reads the DIRECT camera publisher, NOT Frigate.
              Publishes http://0.0.0.0:8787/status.json (+camera.jpg/debug-camera.jpg)
              No HLS rendering in this deployment (`--status-only`, hls_encoder_enabled=false).

projector Android device .39 renderer
  camera.catprojector.display / ProjectorDisplayActivity
    launched by HA script.cat_projector_play with
      --es video_url <input_text.cat_projector_video_url>  (HA local mp4, direct)
      --es overlay_status_url http://192.168.100.74:8787/status.json
    => renderer plays the source video ITSELF and applies blackout zones from
       overlay status.json. Renderer is a consumer of overlay status, not of
       the overlay's video (no stream.m3u8 is served in status-only mode).

cat-projector-safety-watchdog.service (nucat, PID 5138, 0.35 s poll)
  reads overlay status + HA session/display states;
  DANGER (safety_overlay_unavailable/starting/status_error/status_stale or
  camera_error) for >= 8 s while display active -> script.cat_projector_lamp_off
  (fail-closed at the LAMP). Clear requires 5 s of clean status.
  Status staleness threshold 1.5 s (fail-open at the RENDERER side: overlay
  itself expires stale `active` results after 0.9 s -> `no_person`).
```

Service/source ownership: overlay unit lives in
`/home/kom/.config/systemd/user/cat-projector-safety-overlay.service` (drop-in
dir empty); watchdog in same dir. Frigate camera block is live-owned in
`/srv/cold-storage/hass/frigate/config/config.yaml:156` (`Cat_projector_camera`,
input `rtsp://127.0.0.1:8554/cat_projector`, detect fps 5). HA package mirror:
`tasks-loop/ops/homeassistant/packages/cat_projector.yaml` (overlay status URL
`http://192.168.100.74:8787/status.json`, consistent with live bind).

## Measurements (windows stated per probe)

1. Live identity: Frigate `/api/version` = `0.18.0-77a66e7`, uptime 14967 s.
   `Cat_projector_camera`: camera_fps 5.1, process_fps 5.0, skipped_fps 0.0,
   detection_fps 1.9; detector inference_speed 16.9 ms.
2. Direct vs Frigate frame comparison, 20 s window, 0.2 s sampling,
   n=91 valid pairs each, 96x54 grayscale MAD cross-correlation of
   scene-change series:
   - direct `/shot.jpg`: frame-change MAD p50 0.366 (i.e. scene mostly static at
     probe time; projector showing low-motion content).
   - Frigate `latest.jpg`: p50 0.383.
   - Best alignment lag `-600 ms`, correlation 0.974 (peak unique within
     ±2 s; next-best 0.33). => Frigate snapshot path lags the direct camera
     publisher by ≈600 ms. (FPS alone was NOT used as latency proof.)
3. Overlay internal stages, 12 s window, 0.25 s status polling, n=48:
   - camera_read_ms (blocking urlopen of shot.jpg): p50 66.8 / p95 158.0 / max 192.3 ms
   - detector_ms (MobileNet-SSD): p50 9.8 / p95 29.0 / max 80.5 ms
   - source_filter_ms: 0.0 (disabled, no raw detections)
   - safety_worker_loop_ms: p50 80.8 / p95 175.4 / max 201.9 ms
   - status_interval_ms: p50 50.1 / p95 52.0 ms (20 Hz publication, healthy)
   - safety_result_age_ms at publication: p50 38.1 / p95 107.6 / max 305.5 ms
   - source_frame_age_ms (HA mp4 tracked at source_tracking_fps=5): p50 217.5 /
     p95 312.7 / max 421.2 ms
4. Consumer health: watchdog journal over sampled window shows only
   `no_action:no_person`, latch file lamp_held_off=false; overlay PID 5136 /
   watchdog PID 5138 running since 2026-10-13 (~428 ks), no restart loop.
5. End-to-end safety reaction budget (derived from measured values + code):
   camera appearance -> overlay publication ≈ camera_read (67 p50) + detector
   (10) + compute/tracking (~3) + status publish delay (50) ≈ 130 ms p50,
   ≈300 ms p95. Renderer blackout additionally depends on the display app's
   status poll interval (APK source not in repo — unmeasured, see hypotheses).
   Watchdog lamp-off path: status staleness 1.5 s + danger delay 8 s + HA/ADB
   command time ≈ 10 s worst case.

## Confirmed bottlenecks

- The safety overlay does NOT sit behind Frigate: it reads the direct publisher
  and its p50 reaction path is ~130 ms. The ≈600 ms Frigate lag (probe 2) is
  confined to Frigate consumers: Frigate live view, `camera.cat_projector_camera`
  snapshots, HA camera proxy, and anything else fed from `latest.jpg`.
- Within the overlay, the dominant stage is the per-sample blocking
  `urlopen(shot.jpg)` (p50 67 ms, p95 158 ms — it likely forces a fresh camera
  capture per request), i.e. ~50% of the worker loop. Detector is second.
- HA gate poll (`input_boolean.cat_projector_session`,
  `binary_sensor.cat_projector_display_active`) and HA source-video poll are
  synchronous HTTP inside the same worker thread (1 s cadence, 1 s timeout
  each): an HA stall can stretch a single safety sample by up to ~2 s. Not
  observed in this window (gate error null).

## Hypotheses (not confirmed; need APK/deployment evidence)

- The projector renderer app's own poll interval for status.json determines the
  last mile of blackout latency; source not available in repo, unmeasured.
- Frigate live-view perceived delay (user report) is dominated by the
  ~600 ms go2rtc/Frigate path plus live-view client buffering, not by the
  overlay.

## Safety invariants (must survive any latency fix)

1. Direct-camera read stays the overlay input (never route safety inference
   through Frigate).
2. Camera read failure => status `safety_overlay_unavailable`, watchdog lamp-off
   after 8 s; never silently `no_person`.
3. Stale `active` results expire to `no_person` after 0.9 s in the server, and
   the watchdog independently treats status age > 1.5 s as danger: the lamp
   fails closed even though the renderer blackout fails open.
4. `updated_at` must be produced by the safety worker's completion timestamp
   (monotonic-based age), so faster publication cannot mask stale detections.
5. Prediction/tracking only expands coverage; served overlays use
   `_people_with_current_frame_evidence` — predicted-only tracks must not be
   served as current evidence.

## Narrow fixes proposed (no change applied in this pass)

a. Persistent HTTP connection (or MJPEG stream read) for shot.jpg in
   `_read_camera_snapshot`, or subscribe once to `video?fps=N` and grab frames
   from the buffered stream: removes the per-sample capture handshake (est.
   saves ~60 ms p50 / ~150 ms p95 per safety sample). Keep the 2 s timeout and
   the `camera_error` fail-closed path.
b. Move the HA gate/source polls off the safety thread (cache result, refresh
   in background): bounds the worst-case sample at camera+detector+filter even
   when HA is slow. Fail-open gate semantics on HA error must be preserved.
c. For Frigate live-view delay (separate concern): lower go2rtc buffering for
   `cat_projector` / serve `latest.jpg` from the detect stream at 5 fps is
   already the floor; the remaining ~600 ms is mpjpeg->rtsp restream +
   Frigate snapshot pipeline. Do not touch this to fix overlay safety — it is
   not on the safety path.

## End-to-end oracle

1. Frame-lag oracle (no physical change needed): the MAD cross-correlation probe
   from measurement 2 must keep peak |lag| < 100 ms between `/shot.jpg` and
   Frigate `latest.jpg` if the Frigate path is fixed; today it is ~600 ms.
2. Overlay reaction oracle: poll `/status.json` at 4 Hz for 60 s while the
   projector plays moving content; `safety_result_age_ms` p95 must stay
   < 150 ms and `safety_worker_loop_ms` p95 < 120 ms after fix (baseline:
   108 ms / 175 ms recorded above), with `camera_error: null` throughout.
3. Fail-closed oracle: block the camera endpoint (e.g. firewall drop for 3 s in
   a maintenance window) => status must become `safety_overlay_unavailable`
   within one 2 s timeout, watchdog must latch lamp_off after 8 s, and lamp_on
   must require 5 s of clean status. (Destructive; run only in maintenance.)


## Exact patch points (line refs at trace time)

1. Camera read (biggest stage): `scripts/cat_projector_safety_overlay_server.py:280-283`
   `_read_camera_snapshot()` does a fresh `urlopen(shot.jpg)` per sample; called at
   `:1622` inside `_run_safety_worker`. Smallest fix: keep a persistent
   HTTP/1.1 connection (or read the buffered `video?fps=N` MJPEG stream and decode
   the latest frame), preserving the `timeout=args.camera_snapshot_timeout`
   contract and the `except Exception` -> `camera_error` fail-closed path at
   `:1686-1687`. Est. saving: p50 ~60 ms, p95 ~130 ms per safety sample.
2. Synchronous HA polls on the safety thread: `ProjectorActiveGate.active`
   (`:338-365`, urlopen at `:306`) and `ProjectorSourceVideo.source`
   (`:391-421`), both called from `_run_safety_worker`/`_run_status_only` loops.
   Smallest fix: refresh both caches in a daemon thread; keep fail-open-on-error
   semantics (`self._active = True` at `:356`) unchanged.
3. Frigate lag (separate path, do NOT couple to overlay): live config
   `/srv/cold-storage/hass/frigate/config/config.yaml:156-182`
   (`Cat_projector_camera`, input `rtsp://127.0.0.1:8554/cat_projector`,
   detect fps 5). The ~600 ms vs-direct lag lives in the go2rtc mpjpeg->rtsp
   restream + Frigate snapshot pipeline, not in this camera block's fps value.
4. Renderer last mile (unmeasured): HA `ops/homeassistant/packages/cat_projector.yaml:107,135,137,211,232`
   passes `overlay_status_url http://192.168.100.74:8787/status.json` to
   `ProjectorDisplayActivity`; the app's status poll interval is in the APK
   (no source in repo) — measurement gap, needs APK inspection or an
   on-device log line.

## Precise measurement gaps

- Renderer blackout latency: ProjectorDisplayActivity status-poll interval and
  draw-to-phosphor delay (APK source absent; would need adb logcat/screencap
  timing in a maintenance window).
- Frigate live-view client buffering: only the server-side ~600 ms snapshot-path
  lag was measured; the browser/WebRTC client leg not measured (needs a player
  probe).
- shot.jpg per-request capture cost on the Android publisher: inferred from
  p50 67 ms read + p95 158 ms; not directly instrumented on-device.


## Correction note (lead review of CaptureTrace receipt, 2026-10-18 02:0x)

CaptureTrace's publisher-lane receipt
(`receipts/capture-latency-20261018.md`) concluded the .39 publisher is
"cleared as the culprit" with capture+transport <= 35-70 ms. That conclusion is
NOT adopted here, on lead review:

- Matching `/status.json` constants (1280x720, q90, fps bounds) against repo
  source is consistency evidence, NOT proof the deployed APK equals git HEAD
  3ba40e5. Deployed-APK identity verification is CameraRepair's task.
- Steady 28 fps MJPEG and 12 distinct `/shot.jpg` hashes prove the callback
  path is advancing and not backlogged at the network edge; they do NOT bound
  sensor-exposure age. A stable seconds-old backlog would look identical in
  both measurements (uniform inter-arrival, all frames distinct).
- The publisher exposes no capture timestamp and no frame counter, so callback
  -> send time can be measured but sensor-exposure -> send age cannot be
  produced from the publisher today. That is an instrumentation gap, not a
  clean bill of health.

Impact on my lane: my ~600 ms direct-vs-Frigate lag and ~130 ms overlay-path
numbers are RELATIVE and survive (both branches consume the same publisher, so
publisher-side age cancels in the direct-vs-Frigate comparison and adds
identically to both consumers of shot.jpg). The absolute age of a shot.jpg
frame (which caps the overlay's true safety latency) remains UNBOUNDED above
the HTTP-read p50 of 67 ms until a capture timestamp exists. Proposed fix
(unchanged lane ownership): add `frame_count` + `last_frame_unix_ms` to the
publisher `/status.json` (CaptureTrace's fix 1) — accepted as the prerequisite
for any absolute freshness claim.


## Renderer last-mile constants (source read, 2026-10-18 02:04)

CaptureTrace located renderer source:
`/home/kom/proj/cat-projector-camera/src/camera/catprojector/publisher/ProjectorDisplayActivity.java`
(git Komzpa/cat-projector-camera HEAD 3ba40e5, package camera.catprojector.display).
Constants in `ProjectorDisplayView` (lines 298-301):
`OVERLAY_POLL_INTERVAL_MS=60`, `OVERLAY_ERROR_RETRY_MS=100`,
`OVERLAY_HTTP_TIMEOUT_MS=900`, `OVERLAY_FAIL_CLOSED_DELAY_MS=9000`.
Poll thread reads `/status.json` and redraws on signature change; on HTTP/parse
error it keeps last zones for 9 s, then forces full-screen blackout
(fail-closed at the renderer, matching the watchdog's 8 s lamp-off).

Proof status: SOURCE-LANE EVIDENCE ONLY. Same caveat as the publisher receipt —
file constants at HEAD are not proof the deployed APK on .39 contains them
(CameraRepair owns deployed-APK identity). Measured, not assumed: overlay
status.json publication cadence is 50 ms (my probe), so renderer poll of 60 ms
+ HTTP RTT (~2 ms on-LAN) makes renderer last-mile staleness ≈ 60-120 ms when
healthy, and the fail-closed path is time-bounded at 9 s per source. This
closes the previously named gap "renderer poll interval (APK)" to
source-constants level; absolute closure still gated on deployed-APK proof.
