# Latency deploy receipt — 2026-10-18 (`deploy-latency`)

Lane: LatencyDeploy (omp worker). Scope: build and deploy the frozen latency
fixes (Android publisher + display, overlay explicit stream arg, Frigate
`input_args` one-liner) with verification and rollback surfaces.

**Explicit non-claim:** this receipt does NOT measure or claim the full
round-trip (video -> camera -> nucat -> overlay -> render of overlay) and does
NOT claim any sub-100 ms end-to-end figure. Optical exposure age, lamp-ON
publisher cadence/encode, overlay metric windows, and browser/live-view lag are
unproven here. Proof is the next lane's deliverable (see §9).

All times local `+04` unless a `Z` UTC suffix is shown.

## 1. Frozen sources: tests and build (run once each)

- Android `/home/kom/proj/cat-projector-camera`:
  - `./checks/run-camera-frame-oracle.sh` → `PASS: NV21 equivalence/reuse;
    latest-only sequence, cadence and stalled-frame age`
  - `./build.sh` → OK (only JDK/deprecation/desugar warnings)
  - Built APK SHA-256:
    - `82089b37b01e90d1097d4cd694a8a4ae88ff5419db8f1e6cd8c18654db25cb53`  build/cat-projector-camera.apk (0.20 / 20)
    - `cccc0a81ea574ce8ef514ae046def40fa8222dc54978579371e2fb762e148e6a`  build/cat-projector-display.apk (0.2 / 2)
- Overlay `/home/kom/proj/cat-tv-play` (installed project runtime `.venv`):
  - `.venv/bin/pytest -q tests/test_projector_safety_overlay_server.py` →
    `38 passed in 1.35s`
- Loopback smoke of the new overlay parser against NEW publisher metadata and a
  stale stream (disposable fixture `/tmp/overlay-loopback-smoke.py`, local fake
  publisher only; no live device/service touched) → 6/6 PASS:
  1. snapshot metadata parsed (sequence/age/encode present) and evidence age
     honored (`evidence_at < received_at`)
  2. MJPEG parts with metadata accepted latest-only; 8 sequences strictly
     advancing on one connection (`[3,4,5,6,8,9,10,11]`; skipped slot = dropped
     overwritten frame, never a FIFO)
  3. stalled stream → explicit `camera stream frame stale`, retained frame kept
     as stale evidence
  4. negative control: part without `X-Frame-Sequence`/`X-Frame-Age-Ms` rejected
     (`camera frame requires X-Frame-Sequence and X-Frame-Age-Ms`)

## 2. Target identity before mutation (two mappings each)

- Projector `.39`: adb transport `192.168.100.39:5555` (`HiDPTAndroid_Hi3751V350`,
  product `NL5H00X`) + HTTP `192.168.100.39:8081` body `Cat Projector Camera`.
  Pre-install on-device APKs pulled and hashed = documented rollback pair:
  `bcc33cfa68ea4505eb30233fab9334475f2582d1f773f057e06bea8f1e26a19d`
  (publisher 0.19/19) and
  `324ca818fa39eeb5d4dc405dd2f7ac0c02f109b43db820121f33405c531fb829`
  (display 0.1/1).
- Overlay unit: `systemctl --user show` `FragmentPath=/home/kom/.config/systemd/user/cat-projector-safety-overlay.service`,
  `DropInPaths=` (empty). The full pre-change `ExecStart` is not stored in this
  receipt: it lives in git (`git show HEAD:ops/systemd/cat-projector-safety-overlay.service`;
  the lane's edit is uncommitted at receipt time); §4 shows only the changed
  fragment.
- Frigate: container `frigate-nucat` (`ghcr.io/blakeblackshear/frigate:0.18.0`,
  healthy) + live config `/srv/cold-storage/hass/frigate/config/config.yaml`;
  API `:15000`, go2rtc `:1984`. Pre-change live config SHA-256
  `7fafe74f2899196e15c4763ba6bcf82736229d6341edb5675be8adf03579e7e5`.

## 3. Android install and publisher verification

Maintenance: documented `script.cat_projector_lamp_off` called once via HA REST
(HTTP 200, 02:52), `blacklight=false` confirmed; watchdog left enabled/active
throughout. Install used the README's explicit commands (NOT `install.sh`, which
launches an activity):

```text
adb -s 192.168.100.39:5555 install -r build/cat-projector-camera.apk   -> Success
adb -s 192.168.100.39:5555 install -r build/cat-projector-display.apk  -> Success
adb -s 192.168.100.39:5555 shell pm grant camera.catprojector.publisher android.permission.CAMERA
```

Installed identity readback (read-only pull of installed `base.apk`):

| package | version | installed `base.apk` SHA-256 | == built |
| --- | --- | --- | --- |
| camera.catprojector.publisher | 0.20 (versionCode 20) | `82089b37b01e90d1097d4cd694a8a4ae88ff5419db8f1e6cd8c18654db25cb53` | yes |
| camera.catprojector.display | 0.2 (versionCode 2) | `cccc0a81ea574ce8ef514ae046def40fa8222dc54978579371e2fb762e148e6a` | yes |

Foreground camera service (deviation recorded):

- README's explicit `adb shell am start-foreground-service -n
  camera.catprojector.publisher/.CameraServerService` is rejected on this image:
  `Error: Requires permission camera.catprojector.publisher.START_CAMERA`
  (shell uid 2000 cannot hold the app's `prot=signature` permission; `su` absent).
- The publisher's `MY_PACKAGE_REPLACED` receiver started the service at install.
  Readback: `ServiceRecord camera.catprojector.publisher/.CameraServerService`,
  `isForeground=true foregroundId=1`, `startRequested=true`, notification channel
  `cat_projector_camera`. `/status.json` and headers serve continuously.
  Residual gap: an explicit shell-side restart command does not exist on this
  image; receiver-path start is the working idempotent mechanism.

Publisher metadata / geometry (new contract verified live):

- `/status.json` new fields present: `frame_sequence`, `frame_callback_age_ms`,
  `frame_encode_ms`, `last_send_sequence`, `last_send_callback_age_ms`,
  `last_send_duration_ms`, `timing_origin=preview_callback_entry_not_sensor_exposure`.
- `/shot.jpg` and MJPEG parts carry `X-Frame-Sequence` / `X-Frame-Age-Ms` /
  `X-Frame-Encode-Ms`; verified advancing across 1 s (sequence 677 -> 686).
- Geometry `1280x720`, `jpeg_quality 90`, fps bounds 1..30 (default 30).

### 60 s `/status.json` metrics — **lamp-OFF sample, NOT a playback baseline**

Low-light exposure may contribute; no sensor-exposure timing was measured.
Lamp-ON cadence/encode is collected by proof lane. Window 60.033 s, 590 samples
@10 Hz, 0 errors (2026-10-18 03:00-03:01):

| field (ms unless noted) | min | p50 | p95 | max |
| --- | --- | --- | --- | --- |
| frame_callback_age_ms | 37 | 130 | 294 | 605 |
| frame_encode_ms | 32 | 55 | 88 | 174 |
| last_send_callback_age_ms | 32 | 57 | 97 | 454 |
| last_send_duration_ms | 0 | 2 | 11 | 35 |
| latest_jpeg_bytes | 39724 | 40070 | 40770 | 41245 |

- `frame_sequence` 3275 -> 3732: span 457 frames over 60.033 s (~7.6 fps
  lamp-OFF), 396 strictly-advancing sampled transitions, 193 same-sample
  repeats, **0 regressions**.
- All timings are preview-callback-entry based, never sensor-exposure age; no
  cross-host monotonic comparison was made.

## 4. Overlay unit: one added arg, one restart

Canonical owner: tasks-loop `ops/systemd/cat-projector-safety-overlay.service`
(operational owner = the installed user unit
`~/.config/systemd/user/cat-projector-safety-overlay.service`; identical edit to
both). Change = exactly one added argument, all other args byte-identical. Changed
fragment shown (the full before/after command is not stored here; it lives in
git — see §7):

```text
before: ... --camera-snapshot-url http://192.168.100.39:8081/shot.jpg --host 0.0.0.0 --port 8787 --status-only ...
after:  ... --camera-snapshot-url http://192.168.100.39:8081/shot.jpg --camera-stream-url http://192.168.100.39:8081/video?fps=25 --host 0.0.0.0 --port 8787 --status-only ...
```

(Live publisher preceded the stream consumer: publisher installed and verified
in §3 before this restart.)

- One restart at 2026-10-17T22:58:40Z (`systemctl --user daemon-reload` +
  `restart`). New MainPID 1507351, `active/running`; `ExecStart` readback
  contains the new arg and every old arg unchanged.
- Immediate journal + health cycle: overlay journal clean (HTTP 200s from local
  watchdog and `.39` renderer polls); `/status.json` `status=no_person`,
  `camera_error=null`, `updated_at` advancing.
- Watchdog `cat-projector-safety-watchdog.service` untouched: PID 5138
  unchanged, `enabled`/`active`, journal `{"action":"none","reason":"no_action:no_person"}`.
  Lamp fail-closed protection stayed live the whole window.

### Overlay `/status.json` exact read (single sample, 03:07:42) — 60 s window **MISSING**

No overlay metric window was sampled in this lane (lead scope: "no extra
samples now"); the proof lane collects overlay p50/p95. Exact single read:

```text
status=no_person  camera_error=null  source_size=[1280,720]  updated_at=1792278462.5377603
performance: safety_sample_index=15551  status_target_interval_ms=50.0
  inactive_status_interval_ms=1000.0  status_interval_ms=1000.1  status_loop_ms=0.0
  safety_result_age_ms=50.6  max_active_overlay_age_ms=900.0  status_only=true
  source_video_entity=input_text.cat_projector_video_url  source_video_error=null
debug: safety_worker_idle=true  idle_reason=projector_inactive
  projector_active_gate_states={session: off, display_active: off}  gate_error=null
```

Note for the proof lane: at that instant the overlay's HA active-gate read
`session=off / display_active=off` (idle, 2 s inactive cadence) while the
display activity was focused on `.39`; recorded verbatim, playback/HA state is
the proof lane's ownership.

## 5. Frigate: one line, one restart, full readback

- Supported validation contract checked first: API exposes `/api/config/schema.json`
  (GET), `/api/config/save` (POST), `/api/config/set` (PUT), `/api/restart` (POST).
  Candidate validation used the deployed image's own config parser before
  restart: `docker exec -i frigate-nucat python3` → `FrigateConfig.parse_yaml`
  on candidate and live = `VALID`, `Cat_projector_camera inputs=[('rtsp://127.0.0.1:8554/cat_projector','preset-rtsp-restream-low-latency')]`,
  all six other cameras' inputs byte-unchanged. (`docker compose up` deliberately
  not used — it is not a reload.)
- Backup (documented path), byte-exact pre-change state (SHA-256 equals the
  recorded pre-change live hash `7fafe74f…`):
  `/srv/cold-storage/hass/frigate/backups/cat-projector-lowlatency-20261017T2302UTC/config.yaml`
- Change: exactly one line, `Cat_projector_camera` block
  (`input_args: preset-rtsp-restream` -> `preset-rtsp-restream-low-latency`).
  Proof: `diff` backup vs live = `168c168` one line; `/api/config/raw` diff
  before vs after = the same one line.
- Single restart `docker restart frigate-nucat` at 2026-10-17T23:04:10Z; API up
  after 7 s; version unchanged `0.18.0-77a66e7`.
- Health cycle + readbacks (settled, ~2 min post-restart):
  - `/api/stats`: `Cat_projector_camera` fps 5.0 / process 5.0 / skipped 0.0 /
    detect 4.9 / reconnects 0 / excellent. Other cameras match pre-change
    baseline: Living 5.1/5.1/0/0, Bedroom 5.0/5.0/0/0, Hallway 5.1/5.1/0/0,
    Xiaomi_balcony 5.1/5.1/0/0, Tab70 3.0/3.0/0/0 fair (baseline identical),
    Xiaomi_doorbell 0/0/0/0 unusable (baseline identical — disabled/sleepy).
  - go2rtc `cat_projector`: exactly one producer (mpjpeg,
    `http://192.168.100.39:8081/video?fps=25`) + one consumer (rtsp+tcp,
    `FFmpeg Frigate/0.18.0-77a66e7`) after restart.
  - `/api/Cat_projector_camera/latest.jpg` two reads 1 s apart differ
    (`830fbb49…` vs `e96a2982…`) — advancing, not frozen.
  - Container logs since restart: clean startup (FastAPI started, capture
    processes started, go2rtc healthcheck started), no restart loop. First stats
    sample 40 s post-restart showed expected ramp values; the settled sample
    above is the health result.
- Live config identity after: SHA-256
  `a87689de0a2acb6af46b83f2bbc9f3f422d037ad2ef752574b80c78b0b1a1c64`.

## 6. Playback / lamp state

- Captured before maintenance (02:43-02:47): no active playback (launcher
  foreground, display package stopped), `input_boolean.cat_projector_session=off`,
  `binary_sensor.cat_projector_display_active=off`, `blacklight=false`.
  Recorded playback intent: URL `http://192.168.100.128:8123/local/cat-tv/cats-tv-real-crazy-flies-T0SgA3miIxo-skipintro-30fps.mp4`
  (`input_text.cat_projector_video_url`), source `1280x720`, overlay URL
  `http://192.168.100.74:8787/status.json`, `start_position_ms 0` for this URL
  (the 30000 rule applies only to `cats-tv-october-compilation-fMiAfGtt9rc-seed30.mp4`).
  No active playback existed to resume, so "restore original playback" = keep
  baseline off until the proof starts it deliberately.
- At control handoff (03:06-03:07): a lamp-ON / display-focused state was
  observed — `blacklight=true`, focus
  `camera.catprojector.display/camera.catprojector.publisher.ProjectorDisplayActivity`.
  **Integration correction (2026-10-18): the actor of that state is UNKNOWN and
  it must NOT be attributed to the proof lane.** That lane issued zero adb/HA
  lamp/playback commands before its safe stop, and its first device probe
  (~03:08) already showed launcher foreground
  (`com.zeasn.launcher/.ui.MainActivity`) with no display task. This is likewise
  not a deploy-lane mutation: after the transfer timestamp LatencyDeploy issued
  zero adb/HA lamp/playback commands. Playback restore semantics belong to the
  accepted proof source (`6c09e0c8`, `--restore-off` / `--restore-video-url`).

## 7. Rollback paths

1. **Android**: exact pre-repair APKs + key archived at
   `/home/kom/proj/cat-projector-camera-rollback-20261018` (hashes in §2). The
   README's frozen rollback procedure repackages the archived dex with higher
   version codes (21 / 3) and the archived signing key, then `adb install -r`
   both. Keep the lamp off during rollback; restore the captured display intent
   only after camera + safety checks.
2. **Overlay unit**: remove `--camera-stream-url http://192.168.100.39:8081/video?fps=25`
   from the canonical and installed unit. The full pre-change command is the
   git source (`git show HEAD:ops/systemd/cat-projector-safety-overlay.service`,
   uncommitted edit at receipt time) — §4 shows only the changed fragment;
   `systemctl --user daemon-reload && systemctl --user restart cat-projector-safety-overlay.service`.
   Watchdog needs no change.
3. **Frigate**: restore
   `/srv/cold-storage/hass/frigate/backups/cat-projector-lowlatency-20261017T2302UTC/config.yaml`
   (byte-exact, `7fafe74f…`) — or change only `Cat_projector_camera`'s
   `input_args` back to `preset-rtsp-restream` preserving concurrent edits —
   validate with `FrigateConfig.parse_yaml`, then one `docker restart frigate-nucat`
   and the same readbacks.
4. **Publisher service start**: shell-side explicit start is permission-blocked
   (§3); use the package-replacement receiver path. HA
   `script.cat_projector_restart_camera_publisher` launches `MainActivity` —
   avoid during playback.

## 8. Doc state

`tasks-loop/ops/frigate-nucat/README.md` updated to deployed-and-read-back state
with outcome kept `partial` (browser/live proof + MAD freshness re-probe not
run). Not committed (per instruction). The cat-tv-play README was not touched
(optical worker owns it).

## 9. Exclusive-control handoff — OpticalProbe / LatencyProof

- From ~03:05 (2026-10-18) ALL `.39` adb / HA lamp / playback control is
  exclusively **LatencyProof's** (accepted by Main). LatencyDeploy performs no
  further adb or HA lamp/playback mutations; its remaining surface was read-only
  Frigate HTTP + this receipt/README.
- Harness history: the first Optical harness attempt was withdrawn by lead
  review (wrong package/action; absolute-dark OR could count an unchanged dark
  baseline as a response). Replacement source `6c09e0c8` (correct package/action,
  actual luma falling edge, bounded run, `--restore-off`, local H264/parser
  negative smoke passed) is accepted; LatencyProof executes it.
- Handed-over inputs: production restore URL (§6), overlay status URL
  `http://192.168.100.74:8787/status.json`, camera stream
  `http://192.168.100.39:8081/video?fps=25` (now explicit in the overlay unit),
  publisher header contract (§3), stale-frame and lamp-protection semantics
  (overlay rejects frames older than 900 ms; watchdog fails the lamp closed after
  >=8 s danger and requires 5 s clean to clear).
- Still unproven and owned by the proof lane: full round-trip latency and any
  <100 ms figure; optical exposure age; lamp-ON publisher cadence/encode;
  overlay p50/p95 windows; browser/live-view lag. **No such claim appears in
  this receipt.**
