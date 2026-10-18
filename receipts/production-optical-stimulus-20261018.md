# Production Optical Stimulus Availability — 2026-10-18

Gate receipt. READONLY live inspection; no projector lamp, playback, session, service
config, restart, or source edits performed. No new capture, test, or helper created.
All hardware OFF.

Relocated from an initial wrong-cwd write at
`/home/kom/Dropbox/Reemxy/tasks-loop/production-optical-stimulus-20261018.md`
(sha256 `b5b28842e74f2bc276a2622dc119691f8893dc71ccd12646932f430b7079afce`) into the
cat-tv-play receipts tree and corrected: scoped "not found" to the verified existing
tools, scoped the events-query conclusion to indexed person events, removed the unrun
'pass' test claim, and corrected the proof-tool characterization.

## Question

Can existing tools / fixtures / Frigate recordings provide a safe **autonomous** optical
stimulus through the REAL path
`camera -> production MobileNet -> source-subtraction -> tracking -> eye-band -> physical renderer`
without a person available, without a weakened filter or watchdog, and without human exposure?

## Answer

**Not found in the verified existing tools.** No autonomous stimulus method through the
full real path was FOUND in the existing tools, fixtures, and Frigate recordings checked
below (one bounded read pass). This is a scoped search result over verified existing
tools, not proof that no method can exist. The existing valid proof method
(`cat_projector_eye_safety_proof.py`) is person-gated by design, and the bounded event
query found no indexed real-person recording for this camera. The gap is an external
physical prerequisite, not a tooling gap.

## Evidence checked (one bounded pass)

1. **Durable repo**: `/home/kom/proj/cat-tv-play` (skill `cat-tv-play-ops` source of truth).
   Named scripts verified:
   - `scripts/cat_projector_eye_safety_proof.py` — the production coverage/snapshot proof
     tool. It reads the production service's own `http://127.0.0.1:8787/status.json`,
     requires `direct_detector_active` (rejects `residual_only_people`,
     `held_after_last_detection`), captures production `camera.jpg` /
     `debug-camera.jpg`, and reads the physical camera snapshot
     `http://192.168.100.39:8081/shot.jpg`. Its documented purpose is capture coverage
     of a DETECTED person. This preserves the full real path — but it is a
     coverage/snapshot proof for a person the production detector actually detects,
     NOT by itself a causal end-to-end latency measurement, and not a complete latency
     oracle. It cannot synthesize that detection. A future full production latency
     measurement additionally needs a time-correlated physical stimulus/capture.
   - `scripts/cat_projector_realtime_rect_proof.py` — known and excluded: direct bright
     marker to synthetic status, bypasses the production detector. Not valid for this
     question (already 20/20 p50 740.7 p95 965.6, replay-level proof only).

2. **Tests (read, not run in this gate; replay/synthetic fixture proof, not full physical loop)**:
   - `tests/test_projector_safety.py` (8), `tests/test_projector_safety_overlay_server.py`
     (36), `tests/test_review_overlay.py` (5) — read only, not executed in this gate; the
     tests use synthetic frames (`Image.new("RGB", (1280,720), ...)`) and monkeypatched
     status dicts. No recorded real-person fixture present at the named test locations.
   - `tests/test_detection_measurement.py`, `tests/test_source_subtraction.py`,
     `tests/test_tracking.py` — synthetic/tmp_path frames only (read only).

3. **Frigate recordings (bounded read API, one query)**:
   - `GET http://192.168.100.74:15000/api/events?camera=Cat_projector_camera&limit=20`
     (HTTP 200): all 20 recent events are label `cat_tv_session` (projected-content
     sessions), zero `person` events.
   - `GET ...&label=person&limit=2` -> `[]` (empty).
   - Conclusion: the empty `person[]` result proves there are no indexed `person` events
     for this camera in that query; it does not prove that all recordings are absent.
     And a projector-played person from a clip would be subtracted as expected source by
     source-subtraction, so it is not automatically a valid stimulus.

## Why no autonomous method was found in the verified existing tools

- The production detector only reports a person it sees through the real camera. None of
  the verified existing fixtures, tools, or recordings can inject a person into that
  detector through the real physical path without either (a) an independently visible
  person physically in frame, or (b) a physical actuator presenting a person-like
  stimulus — and option (b) via projector playback is removed by source-subtraction as
  expected source.
- Weakening the filter or watchdog, or faking status, is explicitly out of scope and
  would invalidate the proof anyway.

## Exact external prerequisite (for the lead)

To obtain a true full physical camera-loop eye-safety proof, exactly one of these is
required, in this order of preference:

1. **An independently visible person** physically present in the camera's view
   (a cooperating human in the room, visible to camera `Cat_projector_camera`), at which
   point the existing method is run unmodified:
   `python3 scripts/cat_projector_eye_safety_proof.py --out <dir>` (from
   `/home/kom/proj/cat-tv-play`; defaults read the production service on 127.0.0.1:8787
   and the physical camera at 192.168.100.39:8081). This preserves the full real path.
   That command alone is a coverage/snapshot proof for a detected person, not a complete
   latency oracle; a future full production latency measurement also needs a
   time-correlated physical stimulus/capture.
2. **A suitable physical actuator** presenting a person-like stimulus to the real camera
   that is not a projected image (a projected image would be subtracted as expected
   source and is not a valid stimulus).

Final R1/R2 acceptance remains **BLOCKED** on an independently visible physical
person/stimulus — an external physical prerequisite, not a user chore and not a planned
retest. The discovery row is DONE on actual evidence; the verification row stays open
pending true physical stimulus.

## What was NOT done (intentionally)

- No projector lamp, playback, or session commands; no human exposure arranged.
- No physical stimulus method executed (per instruction: return the method, do not run it).
- No new capture, test, or helper script, and no new oracle implementation.
- No git operations by the gate worker.

## Artifact identity

- This receipt: `receipts/production-optical-stimulus-20261018.md` in
  `/home/kom/proj/cat-tv-play`, relocated from the wrong-cwd write named above and
  included in the 2026-10-18 optical publication commit on
  `fix/dynamic-cat-tv-source`.
- Evidence sources: `/home/kom/proj/cat-tv-play` scripts and tests (named above,
  read-only); Frigate read API `http://192.168.100.74:15000` (bounded, read-only).
