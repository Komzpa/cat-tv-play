# Residual optical latency — corrected receipt (TraceOpticalResidual, 2026-10-18)

Supersedes the unmeasured attribution of the first version (ReduceOpticalResidual,
source-only analysis). This version carries the executed timing diagnostic, the
failed optimized-run chronology, the rollback decision and the honest residual.
Raw observations: `receipts/optical-residual-diag-20261018/` and
`receipts/optical-loop-optimized-20261018/`.

## 0. Corrections to the v1 claims (all three were unsupported)

1. `status_requests` 1008/78 s (~77 ms mean poll) was NOT a worst-case bound and
   NOT a draw timestamp. Measured now: renderer GET arrival gaps p50 63 /
   p95 100 / max 375 ms (709 GETs), server service p50 0.1 ms, and
   activation -> first active-status GET response 1.2-40.8 ms on 8/8 attempts.
2. MJPEG send p50 3 ms was NOT proof that downstream buffers were empty.
   Measured now: wire receive gaps p50 132 / p95 346 / max 743 ms with read
   spans p50 129 ms; one burst episode = 585 ms gap then a 0.1 ms read span
   (bytes sat in the socket while the receiver decoded); camera sequence
   advance per wire frame exactly 1.0; this does not establish whether a persistent
   FIFO exists downstream. The measured socket burst occurred while the receiver
   decoded; buffer location/age downstream is UNKNOWN. Camera-side observations
   include JPEG encode jitter 33-288 ms serialized behind the single preview callback buffer.
3. Subtracting unrelated medians to call ~500 ms a "hardware ceiling" was
   invalid. NO hardware ceiling is claimed in this receipt or anywhere in the
   lane output.

Input-count correction for the optimized run (below): 18 input JPEGs were
written, one per activation — that is NOT 18 valid trials. Valid trials = 0.

## 1. Baseline (canonical, unchanged)

`receipts/optical-loop-repaired-20261018/proof-summary.json`, SHA-256
`162ba02f6944de8b2b05c05cf8cd986419b0c0892570303c6538e6a30c2a44c2`.
20 valid / 0 invalid. Fixed-marker transport/render loop; bypasses production
detector + source-subtraction + tracking + eye-band geometry.

| metric | n | p50 | p95 | max |
|---|---|---|---|---|
| point_ms | 20 | 740.7 | 965.6 | 1591.0 |
| lower_ms | 20 | 619.6 | 833.6 | 1473.5 |
| upper_ms | 20 | 843.3 | 1159.7 | 1737.9 |

This remains the comparison of record. The <100 ms goal is NOT met and is not
claimed anywhere.

## 2. The executed timing diagnostic (v1's deferred section)

Three 45 s windows, one instrumented harness wrapper (reused accepted harness
semantics unchanged: same calibrated homography masks, same thresholds, same
state machine, same MJPEG parser; logging only).

- Runs 1-2 were CONTAMINATED and are not used for attribution. HA history
  proves the cause: `script.cat_projector_play` auto-fires ~7 s after
  session-on (01:43:17.6 Z, 14.2 s long; 01:51:43.6 Z, 24.6 s long) and issues
  its own display intents, replacing the diagnostic's DISPLAY_RECT mid-window
  (run 1: 2 GETs then silence; run 2: 24 GETs then a 3.06 s stall). Runs 1-2
  raw JSONLs were overwritten before the preservation instruction — process
  failure recorded in Retro; their observed counts live in session evidence and
  the HA history above (durable).
- Run 3 is the coherent trace: 709 status GETs / 47 s with zero stalls, 284
  wire frames, 227 publisher samples, 8 activations, 1 valid trial.
  Raw (permanent): `frames.jsonl` (EVERY received frame: sequence, read-start,
  receive, callback age, encode, JPEG bytes, calibrated input/output luma),
  `status-gets.jsonl` (EVERY GET: arrival/response, body, activation nonce),
  `publisher-status.jsonl`, `events.jsonl`, `diag-summary.json`,
  `diag-analysis.json`, `diag-NN-{input,output}.jpg` under
  `receipts/optical-residual-diag-20261018/`.

### 2a. When black decline begins vs the acceptance thresholds (raw frames)

Deltas from input-edge frame receive; thresholds unchanged
(`min_darkening=18`, `max_active_luma=95`).

| attempt | first decline>0 | first decline>=18 | first luma<=95 | settled black | accepted |
|---|---|---|---|---|---|
| 1 | 774 ms | 774 ms | never | 104.1 | no |
| 2 | 445 ms | 640 ms | never | 99.3 | no |
| 3 | 682 ms | 682 ms | never | 99.0 | no |
| 4 | 806 ms | 806 ms | never | 100.7 | no |
| 5 | 780 ms | 780 ms | never | 101.8 | no |
| 6 | 777 ms | 777 ms | never | 101.1 | no |
| 7 | 713 ms | 713 ms | 713 ms | 94.2 | YES (713.3 ms) |
| 8 | 640 ms | 640 ms | never | 100.4 | no |

Answer: the black decline begins 445-806 ms after the observed input edge. Seven
attempts crossed the decline threshold on the first observed decline; attempt 2
first declined at 445 ms and crossed the threshold at 640 ms (195 ms later). The
BINDING constraint is the absolute luma (<=95): under early
AE/content settling the painted black settles at 99-105 and is rejected; the
AE-settled attempt reached 94.2 and was accepted (point 713.3 / lower 573.2 /
upper 963.7 ms). The same rejection mode was measured BEFORE any publisher
change, so it is an environment/threshold-sensitivity limitation, not a
publisher regression (no threshold was retuned).

### 2b. Poll gaps vs bytes queue vs renderer scheduling

- Poll: activation -> first active GET 1.2-40.8 ms; gaps p50 63 / p95 100 /
  max 375 ms. At the ~700 ms scale this is not dominant.
- Bytes queue: consecutive sequence advance of 1.0/frame and send p50 2 ms do
  not establish that no persistent FIFO exists downstream; downstream buffered
  age is UNKNOWN. Socket bursts were observed while the receiver decoded.
  Camera-side observations: wire interval p50 132 ms (6.36 fps), encode 33-288 ms,
  callback age p50 59 ms (wire) / 146 ms (publisher poll), and single callback
  buffer capture stop-and-wait behind the encode.
- Renderer scheduling (poll -> draw -> vsync -> photons) remains UNOBSERVED.
  No screenshot or photon timestamps were taken anywhere; screenshot time is
  not photon time and none is claimed. Legacy Camera exposes no exposure
  timestamp (`timing_origin=preview_callback_entry_not_sensor_exposure`).

## 3. Publisher improvement attempt (minimal) and rollback

Candidate (Main's architecture, adopted): keep the single preview callback
buffer but return it immediately AFTER `CameraFramePipeline.transformNv21`
has copied raw NV21 into `transformedNv21` (the JPEG encode reads only the
copy), BEFORE `compressToJpeg`, with exactly-once `finally` handling when the
transform throws. This overlaps HAL capture with the ~53-58 ms encode without
queuing extra frames. Preserved: transform, JPEG_QUALITY 90, 1280x720, publish
storage, watchdog/overlay/geometry/thresholds. Identities: source
`CameraServerService-5d8f3015.java`, APK `camera-207210774e3a.apk`, both
archived under `receipts/optical-loop-optimized-20261018/candidate/`.

20-trial comparison (same harness SHA `6c09e0c8...` unchanged, same marker
`458741db...`, same defaults, new dir `optical-loop-optimized-20261018`):
**0 valid / 25 attempts** (attempts 1-18 `wait_output_timeout`, 19-24
`wait_input_timeout`, 25 `run_timeout`; `restore_status=off_restored`).
Input-count correction: the 18 `trial-NN-input.jpg` files are one per
ACTIVATION — valid trials are 0. Activegate check: 120/120
`projector_active_gate_active` true over 60 s.

First-failure chronology (startup passed; two post-start failures/hypotheses,
with the first cause unresolved):

1. Attempts 1-18 rejected with the same signature as the pre-change
   diagnostic: input edges found, renderer polled the harness status URL
   throughout (1607 GETs ≈ 15/s x 106 s display lifetime — correct URL, no
   hijack), no accepted output decline. The frozen harness records no
   per-frame output luma (the reason the diagnostic wrapper exists), so the
   in-run rejection cause is UNPROVEN; the measured precedent is run 3 (old
   publisher, pre-change): black settles 99.0-104.1 > 95 on 7/8 attempts.
   [INFERENCE] AE/ambient black-floor above the absolute threshold.
2. At **02:23:02.03 Z** an external `script.cat_projector_stop` — NOT the
   runner (whose cleanup is the separate 02:24:18.68 Z event) — turned
   session off (02:23:02.034) and display_active off (02:23:02.037), ~139 s
   after session-on. Initiator unknown. The stimulus died: attempts 19-24
   lost the input marker, 25 hit `run_timeout`. Run-3 also shows the display
   being replaced by the session-on-triggered `script.cat_projector_play`
   auto-cycle (~7 s after session-on, 14-25 s long) unless the harness starts
   after that cycle completes.

Per the evidence rule (restore known publisher unless an independent
measurement failure is proven AND a bounded correction can produce valid proof
now): no bounded correction can produce valid proof without a forbidden rerun
or threshold relaxation, and the 0/25 establishes no publisher regression.
**Rollback executed**: installed publisher RESTORED to
`82089b37b01e90d1097d4cd694a8a4ae88ff5419db8f1e6cd8c18654db25cb53`
(pull-back verified ~02:35 Z), workspace source reverted to original
(`6872031c...`), candidate preserved in `candidate/`. Baseline 20/20 remains
canonical.

Publisher-level observations (facts, not an optical claim): candidate lamp-ON
window camera sequence advance 529 -> 1184 over 60 s (≈10.9 fps) via the live
overlay's `camera_frame_sequence`; old build lamp-ON measured 6.36 fps wire
(~7.57 fps publisher in the baseline window). Lamp-OFF A/B
(`publisher-cadence-after.jsonl`, 45 s): 11.64 fps, callback age p50 83 ms,
encode p50 33 ms vs run-3 lamp-ON old build 6.36 fps / 146 ms / 61 ms —
content/lamp confounded (encode time depends on scene). The candidate's
cadence and callback-age differences were measured in different-content windows;
the candidate effect is not isolated and was never validated optically.

## 4. Identities (final state)

- Installed publisher: `82089b37...` (restored; rollback path proven).
- Installed renderer: `688be696...` (unchanged all session; pull-verified
  after both installs).
- Frozen, untouched: `ProjectorDisplayActivity.java` `63eb95fa...` and harness
  `scripts/cat_projector_realtime_rect_proof.py` `6c09e0c8...`
  (published by PublishFrozenOpticalSources; FREEZE RELEASED there).
- `CameraServerService.java`: reverted to `6872031c...`.
- Overlay server `b0fb0928...`; marker `458741db...`.
- Pre-candidate repo build `2d80be91...` was destroyed by `build.sh`'s
  `rm -rf build` (never installed; hash recorded before that).
- Build non-determinism note: rebuilt display APK hashes differ from installed
  `688be696...` byte-wise (zip timestamps) from identical source; the display
  APK was never installed by this lane.

## 5. Safety and operations ledger (this lane)

Physical operations after the explicit release from MeasureRepairedOptical:

| when (UTC) | operation | result |
| --- | --- | --- |
| 01:43-02:24 | `script.cat_projector_start` x3 (diagnostic windows + optimized run) | HTTP 200 each; auto play-cycle waited out before am-start from run 3 on |
| 01:43-02:24 | `script.cat_projector_stop` + `script.cat_projector_lamp_off` (safe-off after each window) | HTTP 200 each; verified blacklight=false |
| 02:23:02.03 | external `script.cat_projector_stop` (initiator unknown) | session/display off mid optimized run (chronology §3) |
| ~02:15 | `pm install -r` candidate publisher 207210774e3a (camera APK only) | verified installed SHA == build; renderer untouched |
| ~02:35 | `pm install -r` rollback 82089b37 | verified installed SHA == 82089b37 (pull-back) |
| ~02:36 | explicit physical release to MeasurePacedBrowser | accepted (paced-r1 proof owner) |

Untouched all session: watchdog (`cat-projector-safety-watchdog.service`,
MainPID 5138), overlay service (MainPID 1507351), Frigate/go2rtc config (other
lanes), all eye-safety/quality/geometry thresholds (1280x720 q90, homography,
`min_darkening=18`, `max_active_luma=95` unchanged — no retune), no git, no
reboot, no new service. Final safe-off readback (~02:35 Z): session off
(02:23:02 Z), display_active off (02:23:02 Z), blacklight false, 0
ProjectorDisplayActivity tasks.

## 6. Outcome and uncertainty (best-effort, no ceiling claim)

- The <100 ms goal is NOT met and is not claimed. Best measured round-trip:
  baseline p50 740.7 ms (canonical, 20/20) and the diagnostic's single clean
  trial 713.3 ms (bounds 573.2-963.7 ms).
- Measured stage facts (never summed into a total): activation 2.6 ms; poll
  wait 1.2-40.8 ms (gaps p50 63 / p95 100 / max 375 ms); status service
  0.1 ms; MJPEG send p50 2 ms; wire sampling interval p50 132 ms (±1 interval
  quantization per edge, bounds width ≈ 2 intervals by design); black-decline
  onset 445-806 ms after the observed input edge.
- Unobserved segments where the residual sits, explicitly NOT attributed to
  hardware: renderer poll->draw->vsync->present and photon emission (no photon
  or screenshot timestamps exist; screenshot time would not equal photon time),
  sensor exposure/readout (Legacy Camera exposes no exposure timestamp), and
  AE/ambient drift of the camera black floor (measured 94-105 across one 45 s
  window vs 62-82 in the baseline era) which binds the absolute `<=95`
  acceptance. No hardware ceiling is inferred, measured, or claimed.
- Encode p50 53-58 ms held capture stop-and-wait in the candidate window. The
  candidate publisher cadence and the old-build baseline cadence were measured
  in different-content windows; the descriptive ratio is not a causal estimate
  of candidate effect. The change was NOT validated on the optical loop and was
  rolled back per the evidence rule; the candidate remains unproven.

## 7. Retro

- Keep: per-frame/per-GET instrumentation with activation nonces localized the
  residual without touching thresholds; HA history identified both hidden
  actors (play auto-cycle, external stop); raw-trace discipline exposed the
  input-count mistake (18 activation JPEGs ≠ 18 valid trials).
- Failures: two contaminated diagnostic windows before the ordering fix; runs
  1-2 raw JSONLs overwritten before the preservation instruction (loss
  documented §2); the frozen harness lacks per-frame luma logging, so the
  optimized run's 18 rejections are in-run unattributable (the wrapper exists
  because of this gap); the run stretched past an unknown ~139 s session
  self-stop.
- Decisions: no threshold retune (correct — the 95 bound is safety-adjacent);
  no hardware ceiling (correct — unmeasurable with these instruments); rollback
  of an optically unproven candidate (correct per evidence rule).
- Inference discipline was insufficient: sequence advance and send duration do
  not establish absence of a downstream FIFO, and different-content windows do
  not isolate a candidate cadence effect. No new mechanism/skill campaign is
  warranted by this receipt.
- Next (if re-attempted, not this lane): run the 20-trial harness inside the
  session budget (start promptly after the play auto-cycle; total <139 s),
  with the per-frame luma wrapper active so rejection causes are in-run
  measurable; interpret `max_active_luma` against the measured AE/ambient
  black-floor drift; re-test the archived early-buffer-return candidate as the
  cadence improvement.

