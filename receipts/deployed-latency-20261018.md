# Deployed latency & safety verification — final integrated receipt (`deployed-latency`, cumulative)

## 0. Current-state summary (final integration, 2026-10-18)

This section is the current-state record for the whole night's latency and
safety work. Everything below it is cumulative lane history, preserved as
provenance; obsolete or unsupported claims inside that history have been
corrected in place and marked as corrections. All times are local +04 unless a
`Z` suffix is shown.

### 0.1 What is deployed now (every accepted source fix, oldest first)

The Android publisher `camera.catprojector.publisher` 0.20 (versionCode 20) is
installed as `base.apk` SHA-256
`82089b37b01e90d1097d4cd694a8a4ae88ff5419db8f1e6cd8c18654db25cb53`. It
introduced the frame metadata contract the overlay consumes: `X-Frame-Sequence`,
`X-Frame-Age-Ms` and `X-Frame-Encode-Ms` on MJPEG parts, plus `frame_sequence`,
`frame_callback_age_ms`, `frame_encode_ms` and the `last_send_*` fields in
`/status.json`, at 1280x720, JPEG quality 90, fps bounds 1..30. Its declared
timing origin remains `preview_callback_entry_not_sensor_exposure`, so no
sensor-exposure timestamp exists anywhere in the pipeline. This build is also
the known-good rollback point: the residual lane's candidate
`camera-207210774e3a.apk` (early return of the single preview callback buffer
after the NV21 transform, intended to overlap HAL capture with the encode) was
tested at 0 valid / 25 attempts, archived under
`receipts/optical-loop-optimized-20261018/candidate/`, and rolled back —
installed publisher restored to `82089b37…` and source `CameraServerService.java`
reverted to `6872031c…`.

The Android display renderer `camera.catprojector.display` 0.2 (versionCode 2)
is installed as `base.apk` SHA-256
`688be696125babafbc38baa346f0b061cd56e6a593e4c7d90ac6ec41a857ac0e` (current installed base.apk SHA-256). The
historical previous renderer was `cccc0a81ea574ce8ef514ae046def40fa8222dc54978579371e2fb762e148e6a`. The
renderer repair recorded in `renderer-activation-20261018.md` (bounded playback
retry, `ensurePlayback()` on same-URL `onNewIntent`/`onResume`, real error
logging in place of silent `catch (ignored)`) is what made the marker paint with
the lamp on after §10 isolated the pre-repair failure. The installed renderer
identity `688be696…` and `ProjectorDisplayActivity.java` source `63eb95…`
stayed unchanged through the residual and browser lanes.

The overlay unit `cat-projector-safety-overlay.service` (MainPID 1507351) carries
exactly one added argument, `--camera-stream-url
http://192.168.100.39:8081/video?fps=25`, with every other argument
byte-unchanged, and the watchdog `cat-projector-safety-watchdog.service`
(MainPID 5138, enabled since 2026-10-13) was never touched. The overlay's 38
unit tests (§1) and the 6 fail-closed parser fixtures (§12) stand unchanged and
were deliberately not retested for this integration. The frozen proof sources
are the accepted harness `scripts/cat_projector_realtime_rect_proof.py`
(`6c09e0c8…`) and marker `optical-marker.mp4` (`458741db…`).

Frigate keeps the original `cat_projector` MJPEG producer
(`http://192.168.100.39:8081/video?fps=25`) with `input_args:
preset-rtsp-restream-low-latency`, and adds the derivative
`cat_projector_h264: ffmpeg:cat_projector#video=h264#raw=-fps_mode:v
passthrough`; the selected live view uses the derivative, and the six other
cameras' detect and record blocks are byte-unchanged. The live config SHA-256 is
`649c5e97e08e639532d1dbee4ccfd7b2f542969ffb62edd3705ce3396f6be446`
(`frigate-pacing-20261018.md`). The pacing suffix replaced the CFR-duplication
failure (+9323 decoded frames and 99% drops over about 12 s, buffered lead
0.773 → 3.717 s) with +134 decoded frames over 11.8 s and +5 drops (3.7%), with
buffered lead bounded at 0.19 → 0.41 s. That is a short-window observation, not
a forever-buffer proof.

### 0.2 What is measured (three accepted measurements; no stage sums anywhere)

1. **Fixed-marker optical round trip** —
   `receipts/optical-loop-repaired-20261018/proof-summary.json`, accepted=true,
   20 valid / 0 invalid trials, recorded artifact identity `162ba02f…` (the
   file's own `artifact_sha256` field, cited as such by the residual receipt; a
   literal `sha256sum` of the file on disk reads
   `accfa31d34789aac7bfd6aa378149a2f1de694d22b6c68931e4d41cc1039f595`, so the
   accepted identity is the recorded field value, not the current byte hash).
   Point p50 740.7 / p95 965.6 / max 1591.0 ms; lower 619.6 / 833.6 / 1473.5 ms;
   upper 843.3 / 1159.7 / 1737.9 ms. This is a fixed-marker, same-camera,
   input-frame-to-output-frame measurement that uses the production `_mask`
   homography but bypasses the production MobileNet-SSD person detector, the
   source-subtraction filter, tracking, and eye-band geometry. It is therefore a
   transport/render closed loop and never a full production eye-safety latency.
   Common camera-delay cancellation between the input and output observations is
   assumed, not proven; exposure and readout are unknown; the lower/upper
   brackets cover adjacent-frame sampling only and are not strict photon-onset
   bounds.
2. **Instrumented residual diagnostic** — `residual-latency-20261018.md`, raw
   under `receipts/optical-residual-diag-20261018/`: 8 activations, 1 valid
   trial at 713.3 ms (bounds 573.2–963.7 ms). Activation to first active-status
   GET response was 1.2–40.8 ms; GET arrival gaps p50 63 / p95 100 / max 375 ms;
   status service about 0.1 ms. The black decline began 445–806 ms after the
   observed input edge, and the binding acceptance constraint was the absolute
   luma floor (`max_active_luma=95`) under measured AE/ambient black-floor drift
   (94–105 in that window versus 62–82 in the baseline era). Downstream queue
   age is UNKNOWN despite strictly consecutive sequence advance of 1.0 per
   frame; renderer draw, vsync, present and photon emission are unobserved; the
   sensor exposes no exposure timestamp. No hardware ceiling is inferred,
   measured, or claimed. Startup in those windows was HTTP 200 with the
   session-on play auto-cycle completed and the lamp verified true; an external
   `script.cat_projector_stop` at 02:23:02 Z (initiator unknown) ended the
   optimized run's stimulus.
3. **Browser-versus-direct playback delay** —
   `frigate-browser-marker-pairs-paced-r1-reanalysis-20261018.json` (SHA-256
   `49eceb6ef6c5dbf03d99090a63a5ef61e7b18e1dfdfae198e1cafc6868a89711`), a
   standalone offline schedule-ordinal reanalysis of the preserved raw capture
   `frigate-browser-marker-pairs-paced-r1-20261018.json` (SHA-256
   `064c1f4ba3cb022affd62b990695b9844b69533f5ebfce6d1d7fa3aa17108f11`, 1911
   browser + 1911 direct samples over 76.4 s, whose original zero-pair `stats`
   are preserved verbatim). 19/19 unique pairs: point p50 +271.2 / p95 +391.5 /
   max +391.5 / min −128.8 ms — Frigate MSE playback of `cat_projector_h264`
   relative to the direct camera MJPEG on one nucat browser clock. This is not
   the whole loop and is not a stage in any sum. The 40 ms sampling brackets
   give lower p50 +231.2 / upper p50 +311.2 ms, and all four controls pass
   (+350, +3350 without aliasing, dropped edge, periodic-ambiguity rejection).
   Identity rests on unique embedding into the full non-repeating 21-event
   schedule with a 500 ms tolerance, joined on ordinal (never nearest phase);
   the browser receipt's phrase "TAU < 500" is a wording slip — the reanalysis
   records `tau_ms: 500`. The interval distortion that defeated the original
   exact-quantized-key matcher is unproven beyond that exact-key mismatch.

### 0.3 What is explicitly not claimed (standing acceptance gaps)

- No sub-100 ms claim exists anywhere. The user's target was best-effort and it
  is missed: the best measured round trip is the fixed-marker p50 740.7 ms (and
  the single diagnostic trial at 713.3 ms), roughly seven times the target.
- No full production detector end-to-end latency claim. The accepted optical
  measurement bypasses the person detector, source-subtraction filter, tracking,
  and eye-band geometry, and no recorded real-camera detector fixture exists in
  the repo. No human was fabricated and no live fault injection was performed.
- No stage sums and no proven dominant stage. Per-frame costs (callback age,
  encode, poll waits, wire cadence) are observations only; downstream queue
  age, renderer draw/present and photon timing, and sensor exposure/readout all
  remain unobserved. No hardware ceiling is claimed.
- The browser figure is a relative Frigate-MSE-versus-direct-MJPEG playback
  delay, not an end-to-end latency figure.
- Absolute capture age stays unbounded because the publisher's timing origin is
  the preview callback entry, not sensor exposure.

### 0.4 Safety state at close

The documented `script.cat_projector_stop` and `script.cat_projector_lamp_off`
returned HTTP 200 at 02:43:39 / 02:43:44 Z, with an independent re-read at
about 07:05 +04: `input_boolean.cat_projector_session=off`,
`binary_sensor.cat_projector_display_active=off`, `blacklight=false`, zero
`ProjectorDisplayActivity` tasks, publisher serving `ok 1280x720` at quality 90
with `frame_sequence` advancing, and watchdog MainPID 5138 plus overlay MainPID
1507351 active and unchanged. The paced browser lane closed its own managed
service (`marker-serve-paced-r1-8796`) and left the older 8799 service
untouched. No reboot, no watchdog change, no threshold retune, no git.

### 0.5 Retro — the actual acceptance failures (honest list; no new campaign)

- The first LatencyProof lane executed zero causal trials: identity verification
  and Home Assistant gate discovery consumed the window before the first
  harness call (§1–§8).
- The first optical harness run produced 0 valid / 18 attempts; the renderer
  never painted the marker and the display activity died mid-test (§9–§10).
- The residual diagnostic runs 1–2 were contaminated by the session-on
  `script.cat_projector_play` auto-cycle replacing the diagnostic display intent
  mid-window, and their raw JSONLs were overwritten before the preservation
  instruction — a process failure, retained as documented loss.
- The publisher candidate `207210774e3a` reached 0 valid / 25 attempts; 18
  activation JPEGs were initially miscounted as 18 valid trials (corrected to
  0), an external stop at 02:23:02 Z (initiator unknown) killed the stimulus,
  and the candidate was rolled back rather than retuned.
- Diagnostic run 3 accepted only 1 of 8 activations, bound by the absolute
  `max_active_luma=95` threshold under AE/ambient drift; no threshold was
  retuned (the bound is safety-adjacent), which was the correct call.
- The browser baseline capture was invalid on three counts: a
  seconds-versus-milliseconds window filter silently matched zero samples, the
  asserted ROI had zero overlap with the true homography mask, and the periodic
  marker carried no cycle identity. The paced-r1 first attempt then failed
  before sampling with `ReferenceError: CAM_W is not defined` (four frozen
  config constants missing after the sampler patch) and was restored exactly.
  The paced-r1 raw capture itself recorded 0 pairs against the exact
  quantized-key matcher; the accepted 19/19 comes from the standalone offline
  reanalysis, with the raw left byte-for-byte unedited.
- Attribution and wording failures corrected by this integration: the deploy
  lane's §6 blamed the 03:06–03:07 lamp-ON state on the proof lane (the actor is
  unknown and that lane issued zero controls then), and the R4/R10 sections
  below once treated a mean 77 ms poll cadence as proof that polling was not
  dominant and camera cadence as the proven dominant term. Both attributions
  were unsupported and are withdrawn in place.

These are execution and inference failures, not newly discovered product
defects. No new skill, prompt, or mechanism campaign is warranted by them.

---

# Historical record below (sections 1–15: first LatencyProof + RunOpticalLoop lanes; R1–R10: OpticalLoopRepaired lane)

Preserved as provenance. Superseded or unsupported claims inside it were
corrected in place and marked; the current state is §0 above.

As-of 2026-10-18 03:55 +04 (all times +04 unless `Z`). Lane: LatencyProof (omp
worker). Companion receipts: `deploy-latency-20261018.md` (deploy lane, read),
`capture-latency-20261018.md`, `overlay-latency-20261018.md` (earlier lanes).

**Historical outcome of this first lane only, superseded for current state by
§0 above and the appended runs in §9 onward: the optical proof never ran in
this lane. ZERO causal trials were executed and no latency number of any kind
is claimed by this section.** First failing proof: this lane did
not launch the `.venv` harness before the lead's safety stop order (03:51);
elapsed lane time was consumed by deployed-identity verification and Home
Assistant gate/session semantics discovery. This was a pacing failure of this
lane, not a device, harness, or calibration failure. The lead preserved the
full original optical acceptance for a fresh bounded owner after control
release (below).

## 1. Correction to `deploy-latency-20261018.md` §6 (attribution)

That receipt's §6 attributes the 03:06–03:07 lamp-ON / display-focused state to
"the proof lane". **Actor/source of that earlier ON state is UNKNOWN** and this
lane does not claim or repeat that attribution. What this lane can attest: its
first device probe (~03:08) already showed launcher foreground
(`com.zeasn.launcher/.ui.MainActivity`) and no display task; this lane issued no
`am start`/lamp/session command at any point before the safe-stop in §3. The
companion `deploy-latency-20261018.md` §6 has now been corrected accordingly
(final integration, 2026-10-18); this section remains the record of correction.

Two lead clarifications recorded verbatim in substance:

- The prior lamp-OFF low-light hypothesis for the publisher age numbers is
  **unproven**; only a lamp-ON measurement may support it.
- Metrics must not be reported from an inactive overlay gate. The deploy
  receipt's overlay single-read at 03:07:42 was `safety_worker_idle=true,
  idle_reason=projector_inactive` (session/display off) — an inactive-gate
  sample, not detector/loop timing.

## 2. Exact installed identities (verified this lane, read-only)

| item | identity | verification (method, as-of) |
| --- | --- | --- |
| publisher `camera.catprojector.publisher` 0.20 (versionCode 20) | installed `base.apk` SHA-256 `82089b37b01e90d1097d4cd694a8a4ae88ff5419db8f1e6cd8c18654db25cb53` = `build/cat-projector-camera.apk` | `adb shell pm path` + `adb pull` of installed base.apk + `sha256sum`, 03:2x; `dumpsys package` lastUpdateTime 2026-10-18 02:53:33 |
| display `camera.catprojector.display` 0.2 (versionCode 2) | installed `base.apk` SHA-256 `cccc0a81ea574ce8ef514ae046def40fa8222dc54978579371e2fb762e148e6a` = `build/cat-projector-display.apk` | same pull+hash, 03:2x |
| overlay `cat-projector-safety-overlay.service` | MainPID **1507351**, `--status-only`, explicit `--camera-stream-url http://192.168.100.39:8081/video?fps=25` present in ExecStart | `systemctl --user status`, 03:08 and 03:53 |
| watchdog `cat-projector-safety-watchdog.service` | MainPID **5138**, `enabled`/`active` since 2026-10-13 02:47:56, untouched | `systemctl --user status/is-active`, 03:08 and 03:53 |
| harness `scripts/cat_projector_realtime_rect_proof.py` | SHA-256 `6c09e0c870fdc8021069553d3bac67a9aa6bcd23db71b678a9c76173f5f75238` (matches assigned accepted source) | `sha256sum`, 03:2x |
| harness runtime `.venv` | Python import smoke: numpy 2.4.6, PIL 12.2.0 present | 03:1x |
| publisher live contract | `/status.json` `ok 1280x720`, jpeg_quality 90, fps bounds 1..30, `timing_origin=preview_callback_entry_not_sensor_exposure`; `/shot.jpg` 200 | single reads 03:10 and 03:53 |

Installed-APK pull copies remain at `/tmp/installed-publisher.apk`,
`/tmp/installed-display.apk` (transient; hashes above are the durable proof;
deploy lane already archived the pre-change rollback pair).

## 3. Physical mutation / job ledger (complete and exhaustive)

**Mutations issued by this lane: three commands, all after the lead's stop
order. Before them: zero mutations.**

| # | when | command | result |
| --- | --- | --- | --- |
| 1 | 03:49:5x accepted | HA REST `POST /api/services/script/turn_on` `entity_id=script.cat_projector_stop` | HTTP 200; entity completion 03:50:37 (`last_changed` readback) |
| 2 | 03:49:5x accepted | HA REST `POST /api/services/script/turn_on` `entity_id=script.cat_projector_lamp_off` | HTTP 200; entity completion 03:49:58 (`last_changed` readback) |
| 3 | 03:55 | `adb shell input keyevent HOME` | UI navigation only, to verify launcher-foreground baseline |

Everything else in this lane was read-only (`curl` GET, `systemctl status`,
`dumpsys`/`getprop`, `adb pull` of installed base.apk, `sha256sum`, file reads,
file writes confined to `receipts/` and `/tmp`). Background jobs started were
all read-only evidence probes (grep/sed/find/health-readback series); the single
mutating job was `bg_92` = the documented safe-stop pair above. No watchdog
change, no config change, no git, no restarts, no repository mutation.

## 4. Safe baseline OFF readback (03:53–03:55, after §3)

| check | value |
| --- | --- |
| `input_boolean.cat_projector_session` | `off` |
| `binary_sensor.cat_projector_display_active` | `off` |
| `getprop vendor.newlink.blacklight.enable` | `false` (lamp off) |
| `mResumedActivity` | `com.zeasn.launcher/.ui.MainActivity` (launcher foreground, after HOME) |
| `ProjectorDisplayActivity` in `dumpsys activity activities` | 0 occurrences (no display task) |
| publisher `CameraServerService` | `isForeground=true` (camera left active and healthy) |
| watchdog / overlay units | `active` / `active` |
| Frigate `GET /api/stats` | HTTP 200 |

Device `.39` adb / HA lamp / playback control is **RELEASED** to the lead for
reassignment to a fresh bounded proof owner. The device is in the clean
documented OFF baseline.

## 5. Measurements actually taken (single reads only — NOT latency proof)

- Publisher `/status.json` single reads: 03:10 `frame_sequence 7475,
  frame_callback_age_ms 48, frame_encode_ms 29, latest_jpeg_bytes 57666`;
  03:53 `frame_sequence 37253, frame_callback_age_ms 78, latest_jpeg_bytes
  57684`. Units ms / bytes / count, lamp state at capture not recorded. Frame
  sequence advanced (not frozen). **These are not windows and support no
  latency claim.**
- One camera frame 03:12: 1280x720, mean luma 138.22/255, max 255. Lamp state
  at capture **unknown** (this lane never switched the lamp); retained honestly
  as `receipts/latency-proof-20261018/camera-frame-0312-lampstate-unknown.jpg`
  (SHA-256 `a6ed63a2d3f4e03acb926cb300e5bf70c5133b762a7ccda49ab749dc31da6b4a`).
  Not calibration evidence.
- No stage sums were formed or reported anywhere (forbidden by assignment).

## 6. Missing evidence (honest, everything the acceptance requires)

1. **≥20 valid causal optical trials** (point / lower / upper p50/p95/max and
   actual adjacent-frame sampling intervals, raw traces, timestamp
   limitations) — NOT RUN.
2. Input/output trial JPEGs and actual input/output ROI luma checks (marker
   visibly reaching calibrated camera ROI before output status activation) —
   NOT RUN.
3. Lamp-ON camera cadence, JPEG encode, and production `/status.json`
   detector/loop timing 60 s windows (with the overlay gate **active**:
   session+display on) — NOT RUN. Only lamp-OFF single reads exist elsewhere.
4. Parser stale/duplicate behavior and production fail-closed behavior via the
   existing safe disposable fixture (`/tmp/overlay-loopback-smoke.py` or test
   suite) — NOT RUN by this lane (deploy lane has its own loopback smoke on
   record).
5. Real person detector proof on an existing recorded real-camera
   frame/video fixture — NOT RUN.
6. Frigate browser player identity (jsmpeg/WebRTC/MSE/still) and visual
   current-stream proof — explicitly delegated by the lead to the independent
   read-only owner `MeasureFrigatePlayer`; out of this lane's scope.
7. Consequently: **no <100 ms claim, no measured dominant stage, and no
   upstream fix recommendation from this lane.** The full round trip (video →
   camera → nucat → overlay → render of overlay) remains unmeasured.

## 7. Exact blocker

No hardware or software fault: adb transport healthy, publisher serving
sequenced MJPEG, overlay and watchdog active, harness source verified and its
runtime importable. The blocker was lane execution order — deployment-identity
verification plus Home Assistant gate/session semantics discovery consumed the
window before any harness invocation, and the lead's correct safety stop
(03:51, after row P95) froze projection experiments before the first trial.
A fresh bounded owner can start immediately from the clean baseline in §4.

Handoff notes for that owner (from lead + this lane's verified facts):

- Run exactly: `.venv/bin/python scripts/cat_projector_realtime_rect_proof.py
  --out-dir <durable dir> --trials 20 --max-attempts 25 --source-size 1280x720
  --rect 520,260,760,420 --input-rect 120,120,320,220 --restore-off`.
- Physical lamp only via the documented HA path (`script.cat_projector_lamp_on`
  / `script.cat_projector_lamp_off`); never disable the watchdog; `--restore-off`
  returns display baseline; then re-verify §4 row-by-row.
- For production detector/loop timing, the overlay gate must be **active**
  during projection (documented HA session state on + display active);
  `projector_inactive` idle samples do not count.
- Do not fabricate a human for detector proof; fixed-marker transport proof is
  not production safety proof and must be labeled so.

## 8. Retro (short)

1. Launch the primary proof first; identity checks and config reading should
   have been interleaved around it, not front-loaded.
2. The overlay's own `debug` fields (`safety_worker_idle`,
   `projector_active_gate_states`) answer the gate question directly — read
   those instead of walking HA package YAML.
3. Identity verification itself paid off (installed `base.apk` hashes matched
   the deploy lane's build SHAs exactly) and belongs in the first minutes, as a
   bounded check with a hard stop.

---

# Optical-loop execution receipt (RunOpticalLoop lane, 2026-10-18 04:05-04:50 +04)

Single writer: RunOpticalLoop gate agent. This section APPENDS measured outcomes
to §1-§8 above; the earlier zero-trial history is preserved as provenance, not
overwritten. Companion artifacts: `receipts/optical-loop-20261018/` (all files
named below), browser-lane negative proof in
`receipts/frigate-browser-latency-20261018.md` (owner MeasureFrigatePlayer).

## 9. What actually ran (first real harness execution, ever)

Exact ordered command, cwd `/home/kom/proj/cat-tv-play`, harness SHA256 verified
before run as `6c09e0c870fdc8021069553d3bac67a9aa6bcd23db71b678a9c76173f5f75238`:

```
.venv/bin/python scripts/cat_projector_realtime_rect_proof.py \
  --out-dir receipts/optical-loop-20261018 --trials 20 --max-attempts 25 \
  --source-size 1280x720 --rect 520,260,760,420 \
  --input-rect 120,120,320,220 --restore-off
```

Exit status 1, wall ~202 s. `proof-summary.json`: **accepted=false, valid
trials n=0** (point/lower/upper all n=0, p50/p95/max null). 18 attempts:
#1 `wait_output_timeout`, #2-#17 `wait_input_timeout` (16x), #18
`run_timeout`. `restore_status=off_restored`, no restore error. Marker
`optical-marker.mp4` H264 1280x720@30, 300 s, SHA256
`458741db4f0517ffa009511c22660734b52a4fc5a305636bf6e3dc31144b68df`;
offline decode proof: input_rect luma alternates uniform 128 <-> uniform 255
(frac>200 = 0.0 vs 1.0), i.e. the source marker is correct.

Actual-frame evidence (PIL/numpy on saved artifacts, not stage sums):
`trial-01-input.jpg` overall mean 228, output_rect mean 254.0 std 0.0
(pure saturated white), input_rect mean 212 std 13.3 with NO 128/255 marker
structure — trial 1's "brightening" was a saturated startup transient, not the
marker. No causal trial counted.

Fixed-marker caveat (harness's own, retained): the harness bypasses the
production MobileNet-SSD person detector, source-subtraction, tracking and
eye-band geometry. Even n>0 would be transport/render closed-loop only, never
whole-model safety proof.

## 10. Root failure pinned: renderer never paints the marker, then dies

Two bounded marker-render diagnostics (production `_mask` homography, same
geometry as the overlay server; source rectangles `input-rect 120,120,320,220`,
`output-rect 520,260,760,420`):

- Lamp OFF session (`marker-render-diag.json`): 10 samples/6.4 s, input ROI
  120.3-120.6 (spread 0.3), output 169.1-169.9 stable.
- Lamp ON session (`marker-render-diag-lamp-on.json`): 24 samples/10 s (>3
  marker periods), input ROI 113.0-113.8 (spread 0.8), output 165.6-167.1
  stable. No 128<->255 transition ever reached the camera.

Activation facts that rule out the easy causes:
- Marker source delivery proven twice: marker HTTP service (name
  `marker-http-8799`, pid 2676129, port 8799, retained for later proof) access
  log shows `192.168.100.39 ... GET /optical-marker.mp4 ... 200` at 04:27:56
  and 04:42:47. Port reachable from the projector.
- DISPLAY_RECT intent correct and accepted: `am start` exit 0, stdout
  `Starting: Intent { act=camera.catprojector.publisher.DISPLAY_RECT cmp=camera.catprojector.display/...ProjectorDisplayActivity (has extras) }`,
  stderr empty; `mResumedActivity` = ProjectorDisplayActivity (token 57bcabf).
- Lamp verified ON in the recovery session: documented
  `script.cat_projector_start` HTTP 200 -> session on, display_active on
  (~25 s), `vendor.newlink.blacklight.enable=true` (first lamp-ON proof of this
  lane). Note: live script rejects `confirmed_play_request` (HTTP 400) — that
  field from the repo YAML is not deployed on the live script; plain start used.
- Hardware decoder was running (HISI_OMXVDEC FBD/FTB/ETB flow, ~33 ms cadence,
  no decode errors), yet renderer screencaps x4 during lamp-ON marker playback
  were **pure black (mean 0.0)** for 3 frames then jumped to launcher wallpaper
  (53.2) — `renderer-screen.png`, `renderer-screen-lamp-on.png`.

Then the display app died mid-test (durably saved
`android-crash-buffer.log`, 756 lines; `android-lifecycle-events.log`,
91 lines, 46 lifecycle/crash matches):
- `04:44:49.731 WindowManager: WIN DEATH: Window{2b11ef9 u0
  camera.catprojector.display/camera.catprojector.publisher.ProjectorDisplayActivity}`
- `04:45:13.498` / `04:45:16.061` launcher records force-removed,
  "app died, no saved state" (com.zeasn.launcher).
- Repeatable third-party crash (not ours, occurred at 04:09:13, 04:15:34 and
  04:45:15): `FATAL EXCEPTION: main Process: com.zhiying.powerservice`
  `RuntimeException: Unable to start service PowerService ... IllegalStateException:
  View android.widget.RelativeLayout{... 0,0-1920,1080} has already been added
  to the window manager` at `PowerService.onStartCommand(PowerService.java:89)`.
  This powerservice window also held `mCurrentFocus` throughout.
- Post-death: `blacklight=false`, resumed activity `com.zeasn.launcher`,
  camera.catprojector.display process gone.

**Conclusion:** with lamp ON, source delivery proven, intent accepted and
decoder running, the marker still never paints (black screencaps) and the
display activity then dies (WIN DEATH) amid a powerservice crash-loop. The
full-loop measurement is blocked upstream at renderer surface paint +
activity lifetime. The evidence (calibrated no-toggle, black captures,
process death) isolates the renderer/activation investigation for the repair
owner; the root cause itself remains UNPROVEN, and OEM screencap semantics
(black frames under secure/protected surfaces are possible) are unknown, so
this is not yet attributable to a specific defect. Not fixable from this gate
lane (no source edits allowed).

## 11. 60 s status collection (labeled INACTIVE gate)

`status-collect-60s.jsonl`, 111 samples (0.5 s cadence, 59.6 s):
`projector_active_gate_active` **0/111** (session had been unlatched by the
HA stop path mid-window), overlay `status=no_person` for all samples
(`idle_reason=projector_inactive`). Per §7 rule this does NOT count as
production detector/loop timing; those numbers remain UNMEASURED.

Camera-publisher stats from the same window (valid regardless of gate, they
measure the .39 publisher, not the overlay):
- `frame_callback_age_ms` p50 88 / p95 241 / max 306
- `frame_encode_ms` p50 32 / p95 65 / max 71
- `last_send_callback_age_ms` p50 33 / p95 66 / max 71; `last_send_duration_ms`
  p50 1 / p95 4 / max 57
- achieved cadence p50 91.1 / p95 136.7 / max 193.1 ms/frame (~11 fps actually
  delivered vs requested `fps=25`; frame_sequence +657 in 59.6 s)
- `timing_origin=preview_callback_entry_not_sensor_exposure` — capture
  timestamp/frame-counter absent, so exposure/readout uncertainty stays
  unbounded; adjacent-frame interval and callback age are the only quantified
  sampling terms. This is a partial budget (capture→overlay→display→
  re-capture), not absolute capture age.

## 12. Fail-closed fixtures (existing only, no live fault injection)

`pytest tests/test_projector_safety_overlay_server.py -k "stale or duplicate or missing or partial"`
-> **6 passed, 32 deselected, 0.39 s** (covers stale active-result expiry,
duplicate-frame evidence non-renewal, camera-slot duplicate skip/reset-only-
after-reconnect, partial MJPEG frame rejection, missing-window expiry).

Real-detector recorded fixture: **no such fixture exists in the repo**
(`tests/fixtures/` absent; no recorded-detector asset found in a bounded
search). Exact evidence gap: production detector path on recorded real-camera
media remains UNPROVEN — no fabricated human, no live fault injection.

## 13. Final safety state (verified readback, 04:48-04:49 +04)

Documented `script.cat_projector_stop` and `script.cat_projector_lamp_off`
both HTTP 200. Readback: `input_boolean.cat_projector_session=off` (since
00:44:46Z), `binary_sensor.cat_projector_display_active=off`,
`binary_sensor.cat_projector_shading_requested=off`,
`input_boolean.cat_projector_shading_confirmed=off`,
`blacklight=false`. Camera publisher healthy (`/shot.jpg` 200),
overlay 8787 healthy (`status=no_person`, gate_active=false,
idle=projector_inactive), watchdog PID 5138 untouched and active throughout
(only ever emitted `no_action:no_person` / `session_off`; it was NEVER the
stop actor). Physical control released to RepairRendererActivation at 04:50 +04 (not the user; handoff via agent message with the §10-§12 facts); marker
HTTP service retained (§10) for the repair owner's proof reuse.

Session-stop provenance (bounded, per lead's order no broad discovery):
`script.cat_projector_stop` fired at 00:08:50Z during session #1 with no
logbook message; watchdog journal shows no stop decision. Actor not pursued
further; repair owner should re-check it when touching activation.

## 14. Exclusions and honest labels

- **No <100 ms claim** — this remains true in §0.3 (target missed at p50
  740.7 ms). Historical note: at this lane's close n=0 and no latency
  distribution existed; the accepted 20/20 fixed-marker distribution arrived
  later (R3, summarized in §0.2). User
  target stays best-effort and unverified.
- Production safety-model latency UNPROVEN (detector bypassed by design of the
  harness; recorded-detector fixture does not exist).
- Production overlay detector/loop timings UNMEASURED. Historical note: §11's
  collection had the gate never active; R5 later ran an ACTIVE-gate collection
  (112/112 samples) and the overlay exports no loop-timing/age fields at all,
  so production loop timing stays unmeasured as an export gap, not a sampling
  failure.
- Browser-versus-direct playback delay is now MEASURED (paced-r1, recorded in
  `frigate-browser-oracle-20261018.md` §6), superseding this bullet's earlier
  "UNMEASURED" state: 19/19 unique identity pairs, Frigate MSE playback of
  `cat_projector_h264` versus direct camera MJPEG on one nucat browser clock,
  point p50 +271.2 / p95 +391.5 / max +391.5 / min −128.8 ms (nearest-rank).
  The 40 ms sampling brackets give lower p50 +231.2 / upper p50 +311.2 ms, and
  all four controls pass (+350, +3350 without aliasing to +350, dropped edge,
  periodic-ambiguity rejection). This is a relative playback delay against the
  direct MJPEG edge, NOT a whole-loop latency and not part of any stage sum.
  Identity rests on unique embedding into the full non-repeating schedule with a
  500 ms tolerance, joined on ordinal (never nearest phase) — note the browser
  receipt's "TAU < 500" phrase is a wording slip; the reanalysis records
  `tau_ms: 500`. The interval distortion that defeated the original exact-key
  matcher is unproven beyond that exact-key mismatch. The earlier MSE-specific
  finding stays scoped to its lane (go2rtc MSE cannot packetize cat_projector
  MJPEG; the H.264 derivative is that repair,
  `frigate-browser-latency-20261018.md`). A prior ~600 ms latest.jpg
  cross-correlation figure is NOT a browser measurement and must not be
  presented as one.
- Exposure/readout uncertainty unbounded (publisher has no sensor timestamp).

## 15. Retro (this lane)

1. The old zero-trial failure (§1) was lane execution order: verification
   consumed the window. This lane inverted the order — harness FIRST — and got
   real command output (18 attempts, artifacts) within minutes. The lesson
   holds: run the primary proof before paperwork.
2. Raw-rect luma inspection was misleading (saturated startup frame); the
   production `_mask` homography plus renderer screencap was what pinned the
   failure. Next time compute calibrated-ROI stats from the first frame.
3. `script.cat_projector_lamp_on` returning HTTP 200 proved nothing about the
   lamp — only `vendor.newlink.blacklight.enable` readback during a
   `cat_projector_start` session did. Script success != physical state.
4. The remaining gap for full acceptance: renderer/activation repair (marker
   must paint with lamp ON and the activity must survive), then rerun the
   same command, plus a gate-ACTIVE 60 s collection, plus a real-detector
   recorded fixture which does not exist yet.

---

# Optical-loop REPAIRED run — first accepted causal measurement (OpticalLoopRepaired gate, 2026-10-18 05:12–05:26 +04)

Single writer: OpticalLoopRepaired gate agent. Appends to §1–§13 above; nothing
overwritten. Fresh artifact dir: `receipts/optical-loop-repaired-20261018/`
(42 files, new dir, prior `optical-loop-20261018/` untouched).

## R1. Documented start (visible marker, real flags)

- 05:12:04 +04: documented plainstart — HA REST service call
  `script.cat_projector_start`, HTTP 200 (live script rejects the canonical
  `confirmed_play_request` field, hence plain `{}` payload, consistent with §10).
- Flag readback during run: `input_boolean.cat_projector_session=on`
  (since 01:12:04Z), `binary_sensor.cat_projector_display_active=on`,
  `vendor.newlink.blacklight.enable=true`, `mCurrentFocus=ProjectorDisplayActivity`.
  Lamp-ON established BEFORE any trial; blacklight readback, not script-200, is
  the evidence.

## R2. Harness execution (exact assigned command, no discovery repeats)

cwd `/home/kom/proj/cat-tv-play`; harness SHA-256 re-verified before run as
`6c09e0c870fdc8021069553d3bac67a9aa6bcd23db71b678a9c76173f5f75238`:

```
.venv/bin/python scripts/cat_projector_realtime_rect_proof.py \
  --out-dir receipts/optical-loop-repaired-20261018 --trials 20 --max-attempts 25 \
  --source-size 1280x720 --rect 520,260,760,420 \
  --input-rect 120,120,320,220 --restore-off
```

Exit 0, wall 78 s (≈05:12:45–05:14:15 +04). `proof-summary.json`:
**accepted=true, valid_trials 20, invalid_trials 0**, calibration transition
succeeded, `restore_status=off_restored`, no failed_error/restore_error.
Artifact SHA-256 `162ba02f6944de8b2b05c05cf8cd986419b0c0892570303c6538e6a30c2a44c2`.
Marker `optical-marker.mp4` SHA-256 `458741db…` (identical to §9's marker).
All 20 trial input/output JPEG pairs saved.

## R3. Measured round trip (fixed-marker, same camera) — NO <100 ms claim

| bound | p50 | p95 | max | n |
| --- | --- | --- | --- | --- |
| point | 740.7 ms | 965.6 ms | 1591.0 ms | 20 |
| lower | 619.6 ms | 833.6 ms | 1473.5 ms | 20 |
| upper | 843.3 ms | 1159.7 ms | 1737.9 ms | 20 |

Per-trial point list (ms): 1591, 676, 760, 811, 812, 747, 732, 654, 734, 680,
933, 692, 675, 803, 682, 805, 754, 815, 665, 665. Bracket widths
point−lower p50 123.5 ms / upper−point p50 121.2 ms (= one adjacent-frame
interval, consistent with sampling quantization). Clock limitations as recorded
by the harness in `proof-summary.json` (nucat monotonic receive times of actual
neighboring frames; exposure/readout and pre-callback delay unknown; per-frame
`X-Frame-Age-Ms` retained, never summed; common-delay cancellation assumed, not
proven). **Result is ~7.4x above the 100 ms best-effort target; reported as
measured.**

## R4. Dominant-delay evidence from THIS run's causal traces (no stage sums)

- Input→output frame-sequence gap per trial: p50 7 frames (range 4–11).
- `point / seq_gap` = ~111 ms per elapsed camera frame (p50), min 73 max 163 —
  i.e. the point latency is almost exactly "N camera frames × delivered
  cadence". corr(seq_gap, point) = 0.455.
- Achieved camera cadence during trials: adjacent-frame intervals in_cad p50
  121.2 / p95 196.6; out_cad p50 123.5 / p95 282.4 ms. Publisher callback age
  p50 42/49 ms, encode p50 41/48 ms — these are per-frame costs, never summed.
- Harness made 1008 `/status.json` requests over the 78 s run (~13/s, ~77 ms
  mean request cadence). **Integration correction: a mean poll cadence is not a
  worst-case bound and not a draw timestamp, so this bullet withdraws the claim
  that "renderer HTTP/status polling is NOT the dominant term" and the claim
  that the camera cadence is the dominant term — both were unsupported
  attributions.** Specific observations instead: the residual lane later
  measured activation → first active-status GET response 1.2–40.8 ms on 8/8
  attempts and GET arrival gaps p50 63 / p95 100 / max 375 ms; the camera stream
  delivered ~110–135 ms/frame (7.5–9 fps delivered versus fps=25 requested)
  across the median 7-frame input→output span. Specific unknowns: downstream
  buffer/queue age is UNKNOWN despite strictly consecutive sequence advance
  (a socket burst was observed while the receiver decoded, which does not
  locate or age any FIFO); renderer poll → draw → vsync → present and photon
  emission are UNOBSERVED; sensor exposure/readout is unknown. No dominant
  stage is proven here or anywhere in this receipt. Overlay debug exposes no
  loop-timing/age fields (key audit in R5), so finer sub-frame attribution
  (detector inference versus display paint) remains UNPROVEN and is not
  guessed.

## R5. First ACTIVE-gate 60 s production collection

`status-collect-60s-active.jsonl`, 112 samples (0.5 s cadence, 60.2 s,
≈05:16:30–05:17:30 +04, lamp ON, session+display on):
`projector_active_gate_active` **112/112** (vs §11's 0/111 — the gate was
genuinely active this time), overlay `status=no_person` for all samples, no
`idle_reason`. Overlay debug field audit: no age/loop/latency/frame-timing keys
exist (only counts, physics-track counters, gate states) — production loop
timing is not exported by the overlay server; that is an evidence gap, not a
measurement.

Camera publisher (valid regardless of overlay gate): `frame_callback_age_ms`
p50 128 / p95 264 / max 400; `frame_encode_ms` p50 53 / p95 82 / max 128;
`last_send_callback_age_ms` p50 54 / p95 84; `last_send_duration_ms` p50 3 /
p95 14; achieved cadence p50 134.0 / p95 263.5 / max 529.0 ms/frame;
frame_sequence +452 in 59.7 s (**7.57 fps delivered**). `timing_origin`
unchanged: `preview_callback_entry_not_sensor_exposure`.

## R6. Browser-observer marker window (coordinated, read-only observer)

- Marker HTTP service from §10 had died (pid 2676129 gone); restarted 05:22 +04
  (python http.server, pid 3599544) serving the same
  `optical-marker.mp4` from the fresh dir; reachable from .39 (projector GET
  `/optical-marker.mp4 HTTP/1.1 200` at 05:23:59 logged).
- `script.cat_projector_play` via `/api/services/script/turn_on` wrapper with
  `video_url` → HTTP 400 (live script does not accept wrapper-passed fields);
  direct service call `/api/services/script/cat_projector_play` with
  `video_url=http://192.168.100.74:8799/optical-marker.mp4` → HTTP 200 at
  05:23:50 +04.
- Marker observation window: **05:23:50–05:25:45 +04 (115 s)**, exact START/STOP
  delivered to observer `MeasureBrowserAfterRepair` (read-only); observer
  confirmed in-window pairing on direct MJPEG vs Frigate browser player
  (post-ApplyFrigateH264 H264 player; JSMPEG baseline has no in-window pairs
  because no marker existed before 05:23:50). Observer's numbers are its own
  receipt.
- Honest limitation: this gate's own 8 spot camera probes (1 s spacing) over
  the output rect [520,260,760,420] during plain play showed stable mean luma
  196.2–197.0 with no 128↔255 alternation — expected, because that rect is the
  harness-loop's painted position; during plain play the alternation lives at
  the mapped source-rect position which this gate did not probe. The window's
  transition evidence is the observer's paired data plus the projector GET log,
  not this gate's spot probes.

## R7. Causal ordering note (H264 lane interaction)

ApplyFrigateH264's corrective frigate restart landed 05:18:07 +04 — after both
the harness run (ended 05:14:15) and the 60 s collection (ended ≈05:17:30), and
it touched no .39 control. No measurement in R2–R5 overlaps its restart window.

## R8. Final safety state (documented stop, verified readback, 05:25–05:26 +04)

Documented `script.cat_projector_stop` + `script.cat_projector_lamp_off`
(direct service calls) both HTTP 200 at ~05:25:45, immediately after the
observer STOP message. Readback:

| check | value |
| --- | --- |
| `input_boolean.cat_projector_session` | off (since 01:25:17Z) |
| `binary_sensor.cat_projector_display_active` | off (since 01:25:17Z) |
| shading flags | off / off |
| `vendor.newlink.blacklight.enable` | **false** |
| `ProjectorDisplayActivity` occurrences | 0 |
| `mCurrentFocus` | com.zhiying.powerservice (launcher-adjacent OEM window, pre-existing) |
| camera publisher `camera.catprojector.publisher` | `ok 1280x720`, frame_sequence 92495 advancing, cb_age 110 ms — healthy |
| watchdog `cat-projector-safety-watchdog.service` | active, MainPID **5138**, ActiveEnterTimestamp 2026-10-13 02:47:56 — never touched |
| overlay `cat-projector-safety-overlay.service` | active, MainPID 1507351, since 02:58:42 |

Fail-closed 6-test suite and detector-fixture search deliberately NOT rerun
(§12 evidence stands unchanged, per assignment).

## R9. Limitations (unchanged, explicit)

The fixed-marker harness bypasses the production MobileNet-SSD person detector,
source-subtraction/filter, tracking, and eye-band geometry: this is a
transport/render closed-loop measurement with the production `_mask`
source→camera homography, never a production eye-safety latency proof. No
stage sums were formed. Exposure/readout uncertainty stays unbounded. No
<100 ms claim is made or supported.

## R10. Retro

1. The repair that unblocked the run: with lamp ON and renderer fixed by the
   repair lane, the harness calibrated on the first attempt and all 20 trials
   were valid — §10's blocker is gone end-to-end.
2. **Integration correction:** delivered cadence (7.57 fps versus requested 25)
   and callback age p95 264 ms are measured per-frame costs, not a proven
   dominant term. The one optimization attempt — the residual lane's early
   single-buffer-return publisher candidate — ended at 0 valid / 25 attempts
   and was rolled back to `82089b37…`. Renderer draw/present and photon
   segments and downstream queue age remain unobserved, so no stage is
   designated dominant here and no optimization target is asserted.
3. The wrapper-vs-direct service-call asymmetry (`turn_on` 400 vs direct
   service 200 for `cat_projector_play` with `video_url`) is a live-vs-canonical
   drift worth folding into the canonical-behavior docs — recorded, not fixed
   here (no source edits in this gate).
