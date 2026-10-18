# Renderer activation repair — 2026-10-18

## Observed failure (before)
- Normal HA `cat_projector_play` / harness DISPLAY_RECT reached
  `ProjectorDisplayActivity` (am exit 0, mResumedActivity), lamp
  `blacklight=true`, yet wall-camera input ROI flat (~113, spread 0.8/10s)
  and `adb screencap` pure black (mean 0.0) while lamp on.
- `WIN DEATH ProjectorDisplayActivity` 04:44:49 (no app crash stack);
  vendor `com.zhiying.powerservice` FATAL `RelativeLayout already added
  to window manager (PowerService.java:89)` at 04:09:13 / 04:15:34 /
  04:45:15 — chronic since 10-15, AFTER the 04:44 death. Vendor crash
No causal link was established between the vendor crash and the app
death — the cited later vendor crash cannot explain the earlier death
by timestamp alone (no evidence; separate processes).
- Vendor overlay window held `mCurrentFocus` while idle
  (`mCurrentFocus=...com.zhiying.powerservice` with launcher as focused
  app) — observed live 04:54 pre-probe.

## Root cause (recovery gap identified; exact code-level cause NOT
established)
`ProjectorDisplayActivity` (singleTop) treated a repeat DISPLAY_RECT
with the SAME video URL as a video no-op (`onNewIntent` same-URL branch
never touched MediaPlayer), while every video failure was silent
(`setOnErrorListener → true`, `catch (ignored)`). Identified recovery
gap: a sameURL zero-position request did not repair a failed player;
an existing surface-available callback already restarted playback and
positive-position sameURL handling already existed. Candidate playback
was demonstrated after the repair. The original black screen and the
later window death were NOT causally reproduced or explained by this
receipt; the exact code-level root cause of the pre-repair failure
remains unproven.

## Fix (canonical owner: /home/kom/proj/cat-projector-camera)
`src/camera/catprojector/publisher/ProjectorDisplayActivity.java` only:
- `Log` diagnostics: `prepareAsync`, `prepared; starting`, `playback
  error what/extra/url`, deferred-surface and retry-budget lines
  (tag `ProjectorDisplay`).
- Bounded retry: ≤5 attempts, 1s apart, generation-guarded, reset on
  `onPrepared`.
- Same-URL `onNewIntent` and `onResume` call `ensurePlayback()`: restart
  when player is null / not playing / in illegal state; never disturb a
  playing player or an in-flight prepare.
- Untouched: overlay polling + 9s fail-closed blackout, STOP_DISPLAY,
  wakelock, camera server, lamp ON/OFF scripts, vendor power service,
  camera pipeline, watchdog.

## Proof (after, device .39, user-authorized)
- Installed `build/cat-projector-display.apk` 04:54:38 (adb install -r,
  Success). Before-APK pulled to this receipt dir (rollback).
- Normal `script.cat_projector_play` with
  `video_url=http://192.168.100.74:8799/optical-marker.mp4` (reused
  marker-http-8799, pid 2676129): HTTP 200, `blacklight=true`,
  `mCurrentFocus=...ProjectorDisplayActivity`, media_player app
  `camera.catprojector.display`.
- Screencap toggles with marker period: t1 130.7/255 → t2 127.9/128 →
  t3,t4 130.7/255 (white-box high phase vs gray low phase; t2↔others
  meandiff 2.756, 2.17% pixels changed). Earlier identical captures were
  same-phase coincidence, NOT a freeze — correction supersedes any
  freeze claim.
- Wall camera `/shot.jpg` input ROI (x120-320 y120-220) over 70s:
  138↔184 luma toggling at ~3s period, amplitude ~45, sustained whole
  window (n=65) — worker-reported; raw probe code and samples were not
  preserved, and the source rectangle (120,120,320,220) is NOT camera
  coordinates, so calibration is unverified. The rendered high/low
  phases are established instead by the saved screenshots
  `screen-t1.png` / `screen-t2.png` (see Artifacts).
- Sustained ≥60s: activity focus + lamp held from 04:55:01 through probe
  end; watchdog PID5138 alive throughout, untouched; session stayed off
  (direct play path; display_active off is the expected template output
  with session off).
- Normal `script.cat_projector_stop` → HTTP 200, `blacklight=false`,
  focus returned to vendor standby window, session off, display off.
  Lamp OFF behavior preserved.

## Artifacts / hashes (sha256)
- Fixed display APK: `688be696125babafbc38baa346f0b061cd56e6a593e4c7d90ac6ec41a857ac0e`
  (`/home/kom/proj/cat-projector-camera/build/`).
- Before (rollback): `cccc0a81ea574ce8ef514ae046def40fa8222dc54978579371e2fb762e148e6a`
  (`display-apk-before.apk` in this dir).
- Screens high phase: `7ab6acc3383b5d5df25db2bb8ec400eca37add2c98c5e12ac682b6c674b42152`
  (start/sustain/t1/t3/t4 identical pixels); low phase:
  `04bdc1710b37f62bbeb5c83bb6ea70b48d9194f2a6c84c3af95080bdec1e7fad` (t2).
- Marker reused: `receipts/optical-loop-20261018/optical-marker.mp4`
  (523352B, H264 1280x720 30fps, sha 458741db… in proof-summary.json).
- Prior-run evidence: `receipts/optical-loop-20261018/`
  (android-crash-buffer.log, android-lifecycle-events.log,
  proof-summary.json n=0/18 invalid, marker-render-diag*.json).
- Rollback: `adb -s 192.168.100.39:5555 install -r display-apk-before.apk`.
  No git performed (git-pr-owner publishes).

## Live HA drift (not changed, noted)
Live `script.cat_projector_start` rejects `confirmed_play_request`
(HTTP 400); plain start and `cat_projector_play` work. Canonical YAML
declares the field — reconcile live-vs-canonical before callers depend
on it. Probe used `cat_projector_play` + `cat_projector_stop` only.

## Normative requirement / oracle
REQ: normal start paints the SAME marker bytes visibly on camera with a
calibrated low/high toggle; activity+lamp persist ≥60s with healthy
watchdog; normal stop leaves lamp OFF.
ORACLE: wall-camera input ROI toggles ≥25 luma at marker period for the
full window; screencap max 255 in high phase / 128 in low phase;
`blacklight=true` during run, `false` after stop; focus enters and exits
`ProjectorDisplayActivity`.

## Retro
- Silent error swallowing + same-input no-op is the worst repair-hostile
  combination: nothing to observe, retries change nothing. Log every
  external failure at the point of occurrence; make repeat intents
  converge (ensure-state, not assume-state).
- Two pixel-identical screencaps minutes apart looked like a freeze but
  were same-phase captures of a 3s loop — always sample across the
  signal period before claiming stall.
- No causal link was established between the vendor crash and the app
death; the cited later vendor crash cannot explain the earlier death
by timestamp alone. Checking clock order before theorizing saved a
wrong fix (disabling the power service).
