# Frigate browser oracle — frozen 2026-10-18 (CalibrateBrowserOracle)

## 1. Baseline status: INVALID (do not use for latency claims)

- Raw: `receipts/frigate-browser-marker-pairs-baseline-jsmpeg-20261018.json`
  (9014 browser + 9020 direct samples, decoder actually selected-mse-video;
  run tag `baseline-jsmpeg` is a stale label). Old raws preserved, not overwritten.
- Reanalysis: `receipts/frigate-marker-pairs-analysis-20261018.json` → `valid: false`,
  `n_paired: 0`. Its own note admits the sampler's window filter compared ms sample
  `t` against **seconds** window bounds (`/tmp/frigate-marker-window.txt` held
  `1792286630 / 1792286745`), filtering out every sample.
- Geometry defect (worse): the sampler's ROI `[520,260,760,420]` was asserted
  (lines 8–10) to be the production-homography image of source rect
  `[120,120,320,220]` **without recorded evidence**. Proven wrong — see §2.
- Interpretation guard: the 5 direct-side edges are NOT marker evidence, and the
  browser-side constant `229.16–229.25` is codec noise on the wrong ROI, NOT a
  whole-player freeze and NOT a marker. No freshness claim is made here.

## 2. Proven mapping (sole oracle: unmodified production `_mask`)

Computed via `scripts/cat_projector_realtime_rect_proof.py:81-87`
(`overlay_server._source_polygons_to_camera_mask`, `DEFAULT_PROJECTOR_POLYGON`),
repo `.venv`, no live access:

- `_mask((1280,720),(1280,720),(120,120,320,220))` → area **13811 px**,
  bbox **(144,168,306,256)** (162×88), interior sample `mask[211,225] = 1`.
- Old `CAM_RECT [520,260,760,420]` vs true mask: **intersection 0 px, IoU 0.0**.
  Zero overlap is the confirmed defect; no further manual homography re-derivation.
- Frozen interior ROI: **`[170,185,280,240]`** (110×55), measured fill **1.0**
  inside the mask. Both pages sample this same camera-space ROI.

## 3. Patched sampler (temporary, `/tmp`, not a framework, no permanent test)

- File: `/tmp/frigate-latency-pairs.mjs`
- Current SHA256: `de135ad8f1499237c43b70acff1f1ccf01174dc18468dde0a7435847fa80c875`
  (superseded for the paced-r1 capture by `35ddc26bd84b6cf67cc23497051c24bbd5129c9944b05d8e753d88e8f671ba73`
  = this + 4 missing config constants restored from the frozen baseline config; see §6.1).
- Source harness (`scripts/`) untouched. No git. ROI/unit/progress fixes retained.

What changed (retro — old bug → fix):

1. **ROI identity gap** — hardcoded `[520,260,760,420]` with a false provenance
   comment → `MASK_BBOX [144,168,306,256]` + verified interior ROI
   `[170,185,280,240]`; old rect documented as zero-overlap, never reused.
2. **Window-unit bug** — seconds bounds silently matched zero samples → strict
   **milliseconds-explicit contract**: values `< 1e12` throw
   (`...look like SECONDS, not ms...`), `end <= start` throws. See §4.
3. **`new new Date()`** progress-line syntax bug → `new Date().toString()`.
4. **Arbitrary threshold** (`(p20+p80)/2`, fires on noise) → per-side `calibrate()`
   (p10/p90, contrast floor `MIN_CONTRAST = 20`; true 128/255 marker ≈ 127,
   observed wrong-ROI noise ≈ 0.09). Below-floor sides are noise, NEVER marker.
5. **Absolute identity correction:** the previous half-period cap was not evidence;
   +3350 ms on the periodic signal could pass as +350 ms. The current matcher requires
   unique eight-interval edge signatures, at least four uniquely identified matches,
   and reports unmatched counts. Periodic-only evidence is invalid; no lag bound or
   nearest-edge assumption remains.
6. **Visibility preflight** (12 s): both sides must show calibrated contrast and
   ≥2 transitions; identity is checked against the full capture, not prematurely
   gated during this short preflight. Periodic-only captures still fail final validity.
7. Preserved: raw drops (`no-source`/`no-dims`), `currentTime`, buffered lead,
   `totalVideoFrames`/`readyState`, per-sample source identity
   (`selected-mse-video` / `selected-jsmpeg-canvas` / `direct-mjpeg-img`), ROI
   crops+hashes, screenshots, one monotonic browser clock (cross-checked vs node).

## 4. Usage / window-ms contract (for the final capture owner)

```
node /tmp/frigate-latency-pairs.mjs --seconds 60 --tag paced \
  --window-file /tmp/frigate-marker-window.txt \
  --preflight-seconds 12
```

- Window file: **two wall-clock MILLISECOND numbers** (`Date.now()` domain),
  `start end` whitespace-separated, `end > start`, each `>= 1e12`.
  Seconds values are rejected with an explicit error — never zero-matched.
- Clock domain: page `t = performance.timeOrigin + performance.now()` ≈ wall ms
  on both pages (same browser process); offsets kept explicit (`OFF_A=OFF_B=0`).
- The existing marker (3 s cycle, 1 s gray / 2 s white) has no cycle identity and
  MUST be reported invalid. A usable run requires a distinctive, non-repeating
  transition-interval sequence long enough to yield at least four unique signatures.
- Capture the marker's exact media duration/SHA and sequence schedule, start it once
  through the physical owner, and retain the schedule as identity ground truth.
- Receipt JSON carries `stats.valid`; invalid evidence is never paired by proximity.

## 5. Model-free checks and physical handoff

The earlier §5 fixture had browser transitions before direct transitions despite
claiming browser +350 ms; its negative delta invalidates that result. The old
half-period pairing also admitted a +3350 ms periodic phase alias as +350 ms.
Neither is a valid absolute-delay oracle.

Frozen-head syntax command: `node --check /tmp/frigate-latency-pairs.mjs` → exit 0.

Synthetic matcher proof (in-memory extraction of the exact `pairEdges` function;
no live browser, camera, projector, or source harness):

```
node -e 'const fs=require("fs"),vm=require("vm"),s=fs.readFileSync("/tmp/frigate-latency-pairs.mjs","utf8"),f=s.slice(s.indexOf("function pairEdges("),s.indexOf("\nconst browser ="))+"\nthis.pairEdges=pairEdges;",c={};vm.runInNewContext(f,c);const intervals=[1000,2000,3000,1000,2000,4000,1000,3000,2000,5000,1000,2000,3000,1000,4000,2000,3000,1000,2000,5000];const gen=(offset=0,drop=-1)=>{let t=0;const out=[];for(let i=0;i<intervals.length;i++){if(i!==drop)out.push({dir:i%2?"falling":"rising",tMid:t+offset,tLo:t+offset-20,tHi:t+offset+20});t+=intervals[i];}return out;};for(const lag of [350,3350]){const d=gen(),b=gen(lag),p=c.pairEdges(d,b),ds=p.filter(x=>x.paired).map(x=>x.delta_point_ms);console.log(JSON.stringify({lag,n:p.length,paired:ds.length,deltas:ds,valid:ds.length>=4}));}const p=c.pairEdges(gen(),gen(350,4));console.log(JSON.stringify({droppedTransition:{pairs:p.length,paired:p.filter(x=>x.paired).length,deltas:p.filter(x=>x.paired).map(x=>x.delta_point_ms)}}));'
```

Exit 0 for the synthetic matcher command above. Observed:

- +350 ms: 12 uniquely identified pairs, each `delta_point_ms = +350`.
- +3350 ms: 12 uniquely identified pairs, each `delta_point_ms = +3350`.
- Dropped browser transition: 7 uniquely identified pairs remain; each delta is
  +350 ms, with no cycle-shift in subsequent matches.

Separate actual-function controls (extracted `calibrate`, `pairEdges`, and `nr`;
wall-window predicate exercised with the same threshold contract): exit 0.
Periodic +3350 ms fixture: **0 paired / 12 unpaired**; constant 229.2 luma:
contrast 0, below the 20 floor; seconds window: `seconds rejected`; ms window:
`accepted ms`. Exact checks: `node --check /tmp/frigate-latency-pairs.mjs` (exit 0)
and `sha256sum /tmp/frigate-latency-pairs.mjs` →
`de135ad8f1499237c43b70acff1f1ccf01174dc18468dde0a7435847fa80c875`.

The existing periodic asset cannot distinguish cycles. Physical owner: create a
1280×720 clip that alternates ROI luma 128/255 with successive transition intervals
in milliseconds `[1000,2000,3000,1000,2000,4000,1000,3000,2000,5000,1000,2000,3000,1000,4000,2000,3000,1000,2000,5000]`;
loop only after this sequence if needed. Save exact duration and SHA-256 plus the
ordinal schedule as identity ground truth. Run the §4 sampler during playback using
the measured wall-ms window; report all unmatched transitions. No live playback from
this lane. The browser sampler has a 12 s visibility preflight and 60 s capture;
extend `--seconds` to cover the complete distinctive schedule and at least four
identity-bearing matches. A periodic-only capture remains invalid.

## 5.1. Calibrated non-repeating marker — FROZEN (EncodeBrowserIdentityMarker)

Preparation complete. This is the only missing piece; the sampler
(`/tmp/frigate-latency-pairs.mjs`, SHA `de135ad8f1499237c43b70acff1f1ccf01174dc18468dde0a7435847fa80c875`) is untouched and its homography is unchanged.

 Asset dir: `receipts/browser-identity-marker-20261018/`
 File: `receipts/browser-identity-marker-20261018/marker.mp4`
 SHA256: `beca7148aa58bb442427e61c2903d287b3f1b9c47abdb351f671ad3fad2f410f`
 ffprobe: `h264`, `1280x720`, `30/1` fps, `duration 120.000000`, `nb_frames 3600`, size 210516 B.
 Schedule: `receipts/browser-identity-marker-20261018/transition-schedule.json`
  (21 ordinal events, rising/falling + exact ms times).

Design (reused, unchanged): uniform gray128 field + source-space rectangle
`[120,120,320,220]` filled white when active — same as the accepted optical
marker in `scripts/cat_projector_realtime_rect_proof.py` (`generate_marker`).
Only the enable schedule differs: the old generator's `mod(t, period)` periodic
enable was **not** used. The new schedule is non-repeating and deterministic.

Schedule: initial gray hold 5000 ms, then alternating rising/falling edges with
successive intervals (ms)
`[1000,2000,3000,1000,2000,4000,1000,3000,2000,5000,1000,2000,3000,1000,4000,2000,3000,1000,2000,5000]`.
After ordinal 21 (rising @ 53000 ms) the white state is held until 120 s. No
interval repeats within the video; the last state is held to the end, leaving
~67 s of headroom for the 12 s preflight + 60 s capture. The consumer may start
while the initial 5 s gray is showing.

Offline pixel proof (actual decoded file, not generator math) — command and
result retained:

```
python3 /tmp/probe_marker.py
```

 Decoded 3600 frames from `marker.mp4` via ffmpeg rawvideo rgb24 passthrough.
 All 3600 frames within ±12 luma of the scheduled state: OK.
 Representative probes (first/middle/last): ordinal 1 rising @5000 ms
  128.0→255.0; ordinal 11 rising @29000 ms 128.0→255.0; ordinal 21 rising
  @53000 ms 128.0→255.0. Directions and states correct.
 Edge-timing: 21 measured crossings vs 21 scheduled frames, all delta 0
  (within 1 frame). No missed or extra edges; no repeat.

Exit 0. This makes every edge identity unique: the eight-interval signature
is non-periodic, so the §4 matcher's unique-signature contract is satisfiable
and the old +3350 ms phase-alias failure mode is closed.

Serve/play instructions: the physical owner serves this file from the existing
server (no server started or played by this lane). During playback run:

```
node /tmp/frigate-latency-pairs.mjs --seconds 75 --tag paced \
  --window-file /tmp/frigate-marker-window.txt \
  --preflight-seconds 12
```

with the measured wall-ms window. The 12 s preflight + 60 s capture fits
inside the 120 s asset; extend `--seconds` to cover the full schedule and at
least four identity-bearing matches. A periodic-only capture remains invalid.

## 6. paced-r1 FINAL actual capture + schedule-identity reanalysis (MeasurePacedBrowser, 2026-10-18)

Status: **ACTUAL browser playback delay measured** (§6.5). §1 baseline stays
INVALID and untouched; no physical rerun was used for the identity repair —
it is an offline reanalysis of the preserved raw capture.

### 6.1 Physical ownership, marker serving, runs

- Explicit release from `TraceOpticalResidual` at 06:35 +04 (publisher restored
  to known-good `82089b37...`, renderer unchanged `688be696...`, safeOFF
  verified before release). Before that: zero .39/HA/lamp/playback commands
  from this lane.
- New marker served ONLY from my own fresh uniquely-named managed service
  `marker-serve-paced-r1-8796` (port 8796, cwd
  `receipts/browser-identity-marker-20261018/`, served bytes SHA `beca7148...`
  verified). Old 8799 (pid 3599546, cwd `optical-loop-repaired-20261018`)
  confirmed NOT exposing the new asset (404) and was not touched.
- Driver: `/tmp/paced-r1-drive.py`, ledger
  `receipts/paced-r1-driver-20261018.json` (SHA
  `b26eea9ee05f961c9c778fa9bccdf374e4ad025d74e9757c70ea60957e4c42c3`).
- **Run-1 (06:37:51–06:38:29) FAILED before sampling**: sampler FATAL
  `ReferenceError: CAM_W is not defined` at `samplerOpts`
  (`/tmp/frigate-latency-pairs-paced-r1-progress.json`, 06:38:24) — the §3 retro
  patch referenced `CAM_W/CAM_H/CROP_EVERY_TICKS/MAX_CROPS` (lines 273/304/411)
  without defining them. Fixed by restoring exactly the FROZEN baseline-run
  config values (`CROP_EVERY_TICKS 500, MAX_CROPS 24, CAM_W 1280, CAM_H 720`
  from `frigate-browser-marker-pairs-baseline-jsmpeg-20261018.json` config);
  `node --check` exit 0. **Capture sampler SHA:
  `35ddc26bd84b6cf67cc23497051c24bbd5129c9944b05d8e753d88e8f671ba73`**
  (= §3/§4 `de135ad8...` + those 4 constant definitions only; §3's "Current
  SHA256" line is superseded by this one for the paced-r1 capture).
- **Run-2 (actual capture, 06:41:41–06:43:46 +04)**: plain
  `script.cat_projector_start {}` -> start script completed AND automatic
  `script.cat_projector_play` cycle observed on->off (32.1 s) AND
  `blacklight=true` readback -> accepted `harness._start_android`
  (`video_url=http://192.168.100.74:8796/marker.mp4`, status URL
  `http://192.168.100.74:8787/status.json`, source 1280x720) at 06:42:15 ->
  window file `1792291330246 1792291480246` (both >= 1e12, ms-explicit) ->
  exact sampler
  `node /tmp/frigate-latency-pairs.mjs --seconds 75 --tag paced-r1 --preflight-seconds 12 --window-file ...`
  -> exit 0, wall 84.1 s. Start was prompt: capture began inside the marker's
  initial gray/early-schedule region (transitions 1-2 elapsed before sampler
  arm, hence 19 of 21 events observed — see 6.4).

### 6.2 Capture evidence (raw preserved byte-for-byte)

- Preflight PASSED (fail-closed gate): contrast 68.69 browser / 59.10 direct
  (floor 20), 4+4 edges, alternating order — visible calibrated marker on both
  sides. Screenshots `frigate-selected-player-paced-r1-20261018.png`,
  `frigate-selected-final-paced-r1-20261018.png`; ROI crops
  `frigate-marker-roi-{browser,direct}-paced-r1-00..02.png`.
- Raw: `receipts/frigate-browser-marker-pairs-paced-r1-20261018.json` SHA
  `064c1f4ba3cb022affd62b990695b9844b69533f5ebfce6d1d7fa3aa17108f11`:
  1911 browser + 1911 direct samples, 76.4 s span each; 19 browser + 19 direct
  edges; window filter ms-explicit `1792291330246..1792291480246`.
- Actual renderer/transport identity: decoder kind **`selected-mse-video`**
  (1280x720 blob, readyState 4, bufferedLead 0.046 s) fed by
  `ws://192.168.100.74:15000/live/mse/api/ws?src=cat_projector_h264`,
  firstText `video/mp4; codecs="avc1.640029"` (H264), 709 binary frames /
  6.41 MB during the capture — i.e. the paced `cat_projector_h264` restream
  under live config `649c5e97...` (`#raw=-fps_mode:v passthrough`), not jsmpeg,
  not a still tile. Clock cross-check vs node: 19.4 ms max initial, 0.4/0.1 ms
  at close (single browser clock domain).
- ORIGINAL capture `stats` retained verbatim in the raw JSON:
  `valid=false`, `invalid_reason=insufficient-or-ambiguous-transition-identity`,
  `n_paired=0` — not edited, not overwritten.

### 6.3 First cause of the identity rejection (one offline inspection, as ordered)

Frozen matcher (lines 166-191) pairs only on exact equality of a quantized
8-gap signature. First eligible signature (browser edge index 3):

- direct gaps ms `[3200.05,720,2480,4000,842.5,2757.45,2120.05,4800]` ->
  bins `20:5:16:25:5:17:13:30`
- browser gaps ms `[3119.90,920.65,2359.40,3840,1084.85,2875.15,2160,4880]` ->
  bins `19:6:15:24:7:18:14:31`

**All 8 bins differ; the second gap alone differs 200.65 ms** — this is not
40 ms bucket-boundary noise. Root cause: edge midpoints are quantized at 40 ms
sampling and camera-space transitions are gradual/compressed, so each side's
measured intervals deviate from the schedule by up to **359.4 ms (browser) /
480.0 ms (direct)**; exact quantized-key equality cannot survive that jitter.
Missing stimulus is ruled out (6.2: contrast, alternating order, 19/19 edge
counts, ROI crops). This is the identity-quantization sibling of §1's
noise-threshold defect — same class (over-exact matching), different layer.

### 6.4 Offline fix: schedule-grounded unique ordinal embedding (no new physical run)

Script: `receipts/frigate-paced-r1-reanalysis-20261018.py` SHA
`458172fbf60c7cc1d7d777d626085ee6edc1d934593d252de801b268c8df77e2`.

- For EACH side independently: enumerate monotone ordinal embeddings of its
  observed edge sequence into the frozen 21-event schedule
  (`transition-schedule.json` SHA `7d49ef59...`), requiring every inter-edge
  interval to equal the sum of spanned schedule intervals within **TAU=500 ms**
  (schedule intervals are multiples of 1000 ms with minimum separation 1000 ms,
  so TAU < 500 keeps single-interval matches unique; multi-interval bridges are
  guarded by whole-sequence uniqueness).
- Sides are then **joined on the ordinal** — never same-index, never
  nearest-phase/nearest-time. Cross-side check: browser->direct offset 0 is the
  only valid offset.
- Result: **both sides embed uniquely (1 candidate each)**, ordinals 3..21
  (0-based 2..20), i.e. 19/19 pairs, 0 unmatched on either side. Events 1-2
  (5000/6000 ms) elapsed between display start 06:42:15 and sampler arm
  (~06:42:22) — an explained, non-destructive head, not a dropped edge.
- Controls, ALL PASSED:
  - positive lag +350 ms -> 21 pairs, every delta exactly +350
  - delayed +3350 ms -> 21 pairs, exactly +3350, never aliased to +350
  - one dropped browser edge -> 20 pairs exactly +350, exactly 1 unmatched
    direct ordinal, no cycle shift afterwards
  - partial periodic capture -> browser embeds at >= 2 offsets -> rejected,
    0 pairs (periodic-only evidence remains invalid)

### 6.5 Final actual metric (paced-r1)

Output: `receipts/frigate-browser-marker-pairs-paced-r1-reanalysis-20261018.json`
SHA `49eceb6ef6c5dbf03d99090a63a5ef61e7b18e1dfdfae198e1cafc6868a89711`.
`valid=true` (unique per-side identity + contrast >= 20 both sides +
alternating order + 19 >= 4 identity pairs).

Browser-vs-direct observed edge delay, n=19 pairs, nearest-rank:

| metric | point ms | 40 ms-bracket bounds |
| --- | --- | --- |
| p50 | **+271.2** | lower p50 +231.2 / upper p50 +311.2 |
| p95 | **+391.5** | per-pair envelope over all pairs [-168.8, +431.2] |
| max | **+391.5** | bracket p50: browser 40 / direct 40 |
| min | -128.8 | |

Positive = the browser edge is observed later than the direct camera edge on
the same nucat clock.

Interpretation guards (unchanged from §1/§5): this is the delay of the Frigate
dashboard MSE playback of `cat_projector_h264` versus the direct camera MJPEG,
both observed on nucat in one browser clock domain; brackets cover frame
sampling only and are NOT exposure/readout/photon bounds; no stage sums; **no
`<100 ms` claim**; early ordinals 3..8 scatter (-128.8..+231.2 ms) while
ordinals 9..21 concentrate +191..+391.5 ms — reported as measured, not smoothed.

### 6.6 Safe OFF, closure, ownership

- Documented direct `stop` + `lamp_off` HTTP 200 at 02:43:39.4 / 02:43:44.2 Z;
  readback `input_boolean.cat_projector_session=off`,
  `binary_sensor.cat_projector_display_active=off`, `blacklight=false`,
  **0** `ProjectorDisplayActivity` tasks. Independent re-readback ~07:05 +04:
  `blacklight=false`, 0 display tasks, publisher `/status.json` `ok 1280x720`
  quality 90 with `frame_sequence` advancing.
- Watchdog + overlay untouched by this lane (release note: PID 5138 /
  1507351 active and unchanged all session).
- Own resources closed: sampler Chromium closes in-process (`finally`);
  my managed server `marker-serve-paced-r1-8796` stopped after the proof.
  Old 8799 and all other services left as found. No git, no Frigate
  config/camera/source change.
- Physical ownership released back to Main/`TraceOpticalResidual` with the
  receipt SHA + raw paths + safeOFF report.
