# Frigate browser live-view — selected-camera receipt (corrected, v2)
asof: 2026-10-18 04:22:29–04:22:49 Asia/Tbilisi (+04) = 2026-10-18 00:22:29–00:22:49Z
lane: FrigateBrowser (read-only browser observer). No device/HA/service/source/git/config mutations.
target: Frigate http://192.168.100.74:15000 camera `Cat_projector_camera` (go2rtc `cat_projector` <- MJPEG `http://192.168.100.39:8081/video?fps=25`)
status: prior receipt (this file, v1) is REJECTED as baseline; verbatim copy kept in Appendix A.

## 1. Selected-camera surface — PROVEN by actual click (frontloaded)
Method (no guessed hash route): opened the dashboard `http://192.168.100.74:15000/`
(title `Live - Frigate`, 7 tiles), then CLICKED `div[data-camera="Cat_projector_camera"]`
(tile box x=876 y=346 388x218). After the click:
- URL `http://192.168.100.74:15000/#Cat_projector_camera`, title `Cat Projector Camera - Live - Frigate`,
  single target tile remains (header strip 1205x30) + the expanded player surface.
- Screenshots (evidence):
  * `frigate-dashboard-baseline-v2.png` — 7-camera grid BEFORE click (the surface v1 mistook for the result).
  * `frigate-selected-after-click-v2.png` — selected surface at +2.5 s (player still loading; Back/History header).
  * `frigate-target-crop-v2-1..7.png` — target-crop screenshots at ~1.5 s cadence; all 7 SHA-256 distinct
    (431202, 432639, 442929, 435248, 441840, 439087, 432466 B) => the image INSIDE the target crop advances.
    (`frigate-target-crop-v2-0.png` is the 1513 B loading strip, pre-player.)
  * `frigate-selected-final-v2.png` — full screenshot at end.
- "Live" control: NOT present on the selected surface. Visible controls are `Back` (aria "Go back"),
  `History`, and icon-only header buttons; no button/link/aria matches "Live" (the only "Live" is the
  app nav route, already active). Nothing to exercise — recorded, not assumed.
Evidence file: `frigate-click-select-v2.json` (requests/responses/websockets/console/samples).

## 2. Real player websocket + response (the actual cause, measured)
On the click, the REAL player (not a bare probe) did all of the following, in order:
1. Opened `ws://192.168.100.74:15000/live/mse/api/ws?src=cat_projector` (handshake OK) and received an
   explicit go2rtc REFUSAL frame — this is the supported-codec contract violation, stated by go2rtc itself:
   `{"type":"error","value":"mse: streams: codecs not matched: video:JPEG => video:H264, audio:AAC,
   audio:PCMA, audio:PCMU, audio:S16B, audio:S16L, audio:OPUS"}` — then 0 binary frames; socket closed.
2. Console: `cat_projector - MSE error 'startup': Error opening MediaSource. See the documentation: …`
   (2026-10-18T00:22:37.347Z).
3. Fallback chain proven with timestamps: MSE socket closed 00:22:37.350Z, jsmpeg socket
   `ws://…/live/jsmpeg/Cat_projector_camera` first frame 00:22:37.618Z (+268 ms), then
   **57 binary frames / 891 872 B over 11.36 s (~5.0 fps)** through 00:22:48.978Z.
4. Target `<video>`: `blob:` MediaSource attached but `readyState=0`, `currentTime=0`,
   `getVideoPlaybackQuality().totalVideoFrames=0`, `display:none` — the MSE/`<video>` path is dead;
   the renderer is the jsmpeg CANVAS, and it plays (crop screenshots above show the live scene).
Endpoint contract cross-check in the SAME session: control sockets `?src=bed_main` received
`{"type":"mse","value":"video/mp4; codecs=\"avc1.640029,flac\""}` + 8 binary frames and
`?src=xiaomi_balcony` opus-MSE + 106 frames — the identical MSE endpoint serves H.26x streams and
refuses JPEG with the contract string above.
CONCLUSION (precise cause, proven by the selected player's own handshake, not inferred from silence):
the target's go2rtc producer is MJPEG (`video:JPEG`, go2rtc `/api/streams?src=cat_projector`:
producer id 20 `mpjpeg`, receiver `mjpeg`); go2rtc's MSE consumer accepts `video:H264 (+listed audio)`
only, so the browser falls back to jsmpeg and plays ~5 fps. "Browser delay" is therefore a
jsmpeg-canvas latency, not an MSE latency.

## 3. Corrections to the rejected v1 baseline
1. Timestamp/timezone: v1 "asof 2026-10-18T00:05 (Asia/Tbilisi +04; UTC ~2026-10-17T20:05Z)" mislabeled a
   UTC clock reading as Tbilisi local. Actual v1 run window: 2026-10-18 03:58–04:06 +04
   (= 2026-10-17 23:58Z – 2026-10-18 00:06Z). This receipt states +04 and Z explicitly.
2. "The `.39` publisher exposes no capture timestamp or frame counter" — stale/false now. Measured:
   every MJPEG part carries `X-Frame-Sequence`, `X-Frame-Age-Ms`, `X-Frame-Encode-Ms`
   (e.g. seq 51333, age 155 ms, encode 72 ms) and `/status.json` returns
   `frame_sequence` 51333→51883 (monotonic), `frame_callback_age_ms` 118–155,
   `timing_origin: preview_callback_entry_not_sensor_exposure`. Capture age IS observable.
3. "Full-page PNG screenshot hashes differ => the target image advances" — invalid inference
   (any widget/timer changes full-page hashes). Replaced by: fixed target-element crop hashes
   (§1, all distinct) + the jsmpeg frame counter (§2). Full-page hashes are retained as artifacts only.
4. "MSE websocket never opens for the target; the browser renders only a refreshed still;
   codec refusal inferred from source config + cross-camera behavior; smart-streaming ruled out by
   simultaneous multi-camera streaming" — all superseded. The selected player itself opens the target
   MSE socket and receives go2rtc's explicit codec-mismatch error (§2), then plays via jsmpeg fallback.
   v1 probed the DASHBOARD grid via the guessed route `#/live/Cat_projector_camera`, where the target
   tile is never expanded, so it never saw the fallback path and wrongly concluded "still-only".
   No cross-camera inference and no smart-streaming theory is needed or claimed.
5. v1's "browser-side marker comparison impossible because browser shows a still" — replaced: the
   browser plays jsmpeg canvas (~5 fps), so a browser-side transition timestamp IS obtainable; only
   the marker itself was missing (§4).

## 4. Missing measurement + prepared command (optical direct-vs-browser)
NOT measured: the >=10-pair direct-vs-browser marker latency. RunOpticalLoop's first optical run
(`scripts/cat_projector_realtime_rect_proof.py --trials 20 … --rect 520,260,760,420 --restore-off`,
PID 2500088) ended during this lane's session with n=0 usable pairs, and no marker process is running.
This lane does not touch .39/HA/lamp, so it does not restart the marker. No latency number is claimed. RunOpticalLoop's own artifacts: `receipts/optical-loop-20261018/` (`proof-summary.json`, `harness-run.log` 04:18).
Browser side is READY (jsmpeg canvas, measurable). Preserved proof method / exact command for the next
RunOpticalLoop marker run (node, playwright-core, one browser, two pages):
1. Page A: open `http://192.168.100.74:15000/`, click `div[data-camera="Cat_projector_camera"]`,
   wait for jsmpeg canvas in the target container (fallback socket `/live/jsmpeg/Cat_projector_camera`).
2. Page B: open `http://192.168.100.39:8081/` (same-origin) and poll `/shot.jpg` + `/status.json`
   every <=200 ms; direct-frame timestamp = Date.now() at receipt - `frame_callback_age_ms` - `frame_encode_ms`.
3. Both pages: draw the frame (jsmpeg canvas / decoded shot.jpg) to an offscreen canvas each tick,
   mean-luma of rect `520,260,760,420` (proportional mapping on page A), stamped with Date.now().
4. Detect same-direction luma edges (midpoint threshold), pair browser edges to direct edges within 3 s,
   require >=10 pairs; report per-pair delta, median, IQR, and the sampling interval used.
5. Rejection rule: any browser tick whose source is the still `latest.webp` (not the canvas) is counted
   and excluded; the receipt must state the exclusion count.
Trigger: coordinate with RunOpticalLoop to start the marker, then run this command before the marker
batch ends (this lane's scripts: `/tmp/frigate-click-select.mjs` for selection; a dual-page
luma-edge sampler per the spec above for the marker batch).

## 5. Retro — how the dashboard was mistaken for the selected view, and the fix
Cause: v1 navigated directly to the guessed hash route `#/live/Cat_projector_camera` and screenshotted
the WHOLE page. In Frigate 0.18 that route renders the 7-camera grid (the target tile is just one of
seven); the page `<title>` still says "Cat Projector Camera - Live", so the title check passed while the
screenshot showed the grid. From grid-level artifacts (full-page hashes, dashboard-level websockets) v1
read "target plays/still-only" without ever having the target expanded — and the real selected-view
fallback (MSE refusal -> jsmpeg playback) lives ONLY in the expanded surface, so it was missed.
Prevention (now standard for this lane): (a) reach the surface by CLICKING the DOM tile
`div[data-camera=<name>]` and assert URL + title + single-tile layout afterwards; (b) all visual claims
come from fixed target-element crops (hash sequence) and in-element frame counters, never full-page
hashes; (c) the player verdict comes from the real player's own websocket frames (error/codec strings
quoted verbatim), never from the absence of a socket on a bare probe.

## Appendix A — rejected v1 baseline (verbatim, retained as history only)
<details><summary>v1 receipt (REJECTED 2026-10-18 04:30 +04)</summary>

# Frigate browser live-view latency — read-only observer receipt
asof: 2026-10-18T00:05 (Asia/Tbilisi +04; UTC ~2026-10-17T20:05Z)
lane: FrigateBrowser (read-only browser observer). No device/HA/service/source/git mutations.
target: Frigate http://192.168.100.74:15000 camera `Cat_projector_camera`
go2rtc: 1984 `cat_projector` <- MJPEG `.39:8081/video?fps=25` -> RTSP8554 -> Frigate detect 5fps

## 0. Method / browser hygiene
- Isolated headless Chromium driven directly via `playwright-core` (the playwright-cli
  daemon did not persist between shell invocations, so a single-process Node driver was
  used). Fresh `newContext()` per run; NO real browser profile touched. Tabs muted
  (`video.muted=true`, `--autoplay-policy=no-user-gesture-required`). No audio played.
- Runs: `frigate-browser-observer`, `frigate-map-cams`, `frigate-still-cadence`,
  `frigate-ws-contrast`. All closed their own browser in a `finally { browser.close() }`.
  The surviving `/opt/google/chrome` PIDs seen in `ps` carry user-data-dir `dcg-pw-1000`
  (a different lane's Playwright workspace) — NOT mine; left untouched.

## 1. Correct camera + advancing image — VERIFIED
- Page URL `http://192.168.100.74:15000/#/live/Cat_projector_camera`;
  title `/live/Cat Projector Camera - Live - Frigate`.
- DOM contains the projector cell: `div[data-camera="Cat_projector_camera"]` wrapping a
  `<video>` (evidence JSON `mapping[].chain`).
- The rendered projector image ADVANCES (not frozen):
  * Browser screenshot taken 1.2 s apart -> distinct SHA-256
    (`8e639b38…` != `4cedf3d2…`), bytes 351919 -> 351936.
  * Frigate-served still `/api/Cat_projector_camera/latest.webp?height=360` fetched
    twice ~2.5 s apart -> distinct SHA-256, HTTP 200, 10962 -> 24366 bytes.
  So the correct camera's picture is live-updating in the browser, at still-image cadence.

## 2. Protocol identification — MSE (go2rtc) is the live path; target renders as STILL
Live-view is MediaSource Extensions, not jsmpeg/WebRTC:
- `assets/MsePlayer-PsOKtS3P.js` loaded; `window.MediaSource === present`;
  live websockets of form `ws://…/live/mse/api/ws?src=<stream>`.
- NOT jsmpeg: `window.JSMpeg` absent; no streaming `<canvas>`.
- NOT WebRTC: `window.RTCPeerConnection` global exists but no webrtc/ICE socket opened
  for any live view; only `/ws` (Frigate event WS) + `/live/mse/api/ws?src=…` (MSE).
- For the TARGET the effective rendered output is a periodically-refreshed still
  `<img src=/api/Cat_projector_camera/latest.webp?...>`, because the MSE video path
  delivers zero bytes (see §3).

## 3. Target player state vs control — the measured browser bottleneck
Real-player observation (map-cams + still-cadence runs, 5–17 s each):

| camera        | go2rtc source codec | MSE ws opened by player | video readyState | plays? |
|---------------|---------------------|-------------------------|------------------|--------|
| cat_projector (target) | MJPEG (`mpjpeg`, codec `mjpeg`, `video recvonly JPEG`) | never opened | 0 | NO  |
| xiaomi_balcony (ctrl)  | H.265 (`rtsp`, codec `hevc`) | yes, `?src=xiaomi_balcony`, 265 rx frames | 4 | YES |
| living_main (ctrl)     | H.26x | yes, `?src=living_main`, 289 rx | 4 | YES |
| bed_main (ctrl)        | H.26x | yes, `?src=bed_main`, 427 rx | 4 | YES |

Target `<video>` detail:
- still-cadence run: `src=""`, `readyState:0`, `paused:true`, `display:none` for all 17 samples.
- map-cams run: a `blob:` MediaSource WAS assigned (`src=blob:…`, `paused:false`) but
  `readyState` stayed `0`, `videoWidth/Height=0` — the SourceBuffer never received an
  initialization segment, so the player never decoded a frame.

CONFIRMED bottleneck (measured, not inferred from source alone):
- go2rtc `cat_projector` producer is **MJPEG** (`format_name:mpjpeg`, `codec_name:mjpeg`,
  `medias:["video, recvonly, JPEG"]`, url `http://192.168.100.39:8081/video?fps=25`).
  go2rtc's MSE endpoint serves fMP4/H.264/H.265; it cannot packetize MJPEG into MSE, so
  the projector's MSE stream carries no data. Control H.264/H.265 cameras stream over the
  same `/live/mse/api/ws` endpoint. This is why the projector live view degrades to a
  still image in-browser.

### codec-refusal vs motion-based smart streaming (explicitly separated)
Codec refusal, NOT a "stream only the active camera" policy:
- In one run the player opened MSE for TWO cameras simultaneously (`living_main` + `bed_main`);
  in another, `xiaomi_balcony`. Frigate is not limiting to a single active camera — it
  streams whichever cameras can be served, and never serves the projector.
- In the map-cams run the player ATTACHED a MediaSource blob to the target `<video>`
  (`paused:false`) yet received zero bytes (`readyState:0`). A motion/smart-streaming
  policy would leave the video untouched; attaching-but-receiving-nothing is a
  transport/codec-level inability, i.e. MJPEG cannot ride MSE.

Honest instrument caveat: a bare in-browser `new WebSocket(/live/mse/api/ws?src=…)`
probe returned 0 frames for BOTH target and control, so it is NOT a valid codec oracle
(a bare socket does not perform the player's MSE handshake; the real player demonstrably
receives frames for the control cameras). That bare-probe result is therefore excluded
from the conclusion above. The conclusion rests on real-player behavior + go2rtc source
codec, which are directly observed.

## 4. Live-lag / buffering indicators (control camera only — NOT the target)
Because the target never decodes, no browser playback latency figure exists for it.
For a WORKING MSE camera (xiaomi_balcony) sampled 6×/500 ms:
- `currentTime` advanced 7.171 -> 9.695 s over ~2.5 s wall (playing, not stalled).
- browser MSE buffering lag proxy `buffered_end − currentTime` ≈ 0.229–0.268 s
  (p50 ≈ 0.25 s). This is the go2rtc/MSE player's inherent buffer, characteristic of
  that H.265 camera, and is NOT a projector figure.

## 5. What is NOT measured here (honest missing clauses)
- No end-to-end browser playback latency number for `Cat_projector_camera` can be produced:
  in this state the browser does not render it as video, only as a refreshed still.
  A "browser delay" is undefined for a still image (no continuous timeline to correlate).
- No direct-camera-vs-browser optical comparison was run: the optical marker was NOT
  running at observation time (see §6). The earlier ~600 ms figure (OverlayTrace) was
  latest.jpg cross-correlation sampled at 200 ms, not a browser measurement; do not
  attribute it to the browser.
- Absolute capture age remains UNKNOWN (the `.39` publisher exposes no capture timestamp
  or frame counter — see capture-latency receipt). This lane adds no new absolute-age claim.
- Headless SwiftShader rendering may differ from a headed hardware browser in decode
  timing; protocol identification and the MJPEG/MSE refusal are transport-level and are
  not expected to differ.

## 6. Optical marker status + prepared observation method (for RunOpticalLoop)
Marker NOT running at observation time: no marker/optical process present;
overlay `/status.json` shows `projector_active_gate_active:false`,
`input_boolean.cat_projector_session=off`, `binary_sensor.cat_projector_display_active=off`,
`idle_reason:"projector_inactive"`. This lane never starts the projector.

PREPARED method (execute once RunOpticalLoop runs the marker; read-only browser side):
1. Run the accepted marker harness (`6c09e0c8`, luma falling-edge, `--restore-off`) so a
   known transition hits the projector wall. Owner: RunOpticalLoop — NOT this lane.
2. This lane opens the isolated muted browser on `#/live/Cat_projector_camera` and samples
   the target `<video>`/still; concurrently polls the direct camera `.39` `/shot.jpg`
   (or `/video?fps=25` frame grab) — both timestamped from the same wall clock.
3. Per observation, record the marker-transition timestamp as seen (a) direct-camera and
   (b) in-browser. Collect >=10 independent transitions spanning >=10 s apart where possible.
4. Report per-observation deltas, median, MAD/IQR and sampling uncertainty (browser sample
   interval stated explicitly). Flag any observation where the browser shows a still rather
   than video — a still cannot yield a transition timestamp and must be excluded and counted.
5. CRITICAL precheck before spending marker cycles: the browser currently CANNOT render this
   camera as MSE video (MJPEG source, §3). Unless an H.264 restream of `cat_projector` is
   exposed to go2rtc, browser-vs-direct optical latency is not measurable — the browser
   side will be a still. Confirm an MSE video path exists for `cat_projector` first, or the
   comparison will return only "no video" observations.

## 7. Remaining measured browser bottleneck (single line)
`cat_projector`'s go2rtc source is MJPEG; go2rtc MSE/fMP4 cannot carry it, so the browser
MSE player attaches a MediaSource but receives zero bytes and the live view degrades to a
refreshed still — while H.264/H.265 cameras stream over the identical MSE endpoint. The
fix lives upstream (expose an H.264 restream of `cat_projector` to go2rtc), not in this
read-only lane.

## 8. Evidence artifacts
- `frigate-browser-latency-20261018.json` — full summary + request/response/websocket logs
  (98 requests, 96 responses, 2 websockets `/ws` + `/live/mse/api/ws?src=xiaomi_balcony`,
  1 benign `net::ERR_ABORTED` on a balcony still refresh).
- `frigate-browser-latency-20261018.png` — browser screenshot of the live view.
- go2rtc source configs captured in-run (`cat_projector`=mpjpeg/mjpeg; `xiaomi_balcony`=rtsp/hevc).

</details>

Prior v1 evidence artifacts retained as rejected-baseline material:
`frigate-browser-latency-20261018.json`, `frigate-browser-latency-20261018.png`.
