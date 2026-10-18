# Frigate H264 restream frame-pacing repair — 2026-10-18

Lane: ApplyPacingPassthrough. Scope: correct the confirmed CFR-duplication
pacing regression on the `cat_projector_h264` go2rtc derivative by appending
the supported `#raw` FFmpeg option to that derivative only, validate with the
deployed image's own config parser, apply target-only, restart once, and prove
the actual encoder command plus consumer pacing.

**Explicit non-claim:** this receipt does NOT measure any end-to-end latency
(video -> camera -> nucat -> overlay -> render) and does NOT claim a
sub-100 ms figure. `bufferedLead` is labeled as such and is NOT latency.
`framesPerSecond` below is decoder/WebSocket delivery cadence, not latency.

All times local `+04` unless a `Z` UTC suffix is shown.

## 1. Inputs and frozen context

- **Live config before change:** `/srv/cold-storage/hass/frigate/config/config.yaml`
  SHA-256 `767cb9935204ffe45ef33b5ed1521669347d1ec0997a055dfb73cc7120f3d51b`
  (the settled state from the prior `ApplyFrigateH264` lane).
- **Baseline backup (DO NOT REPLACE):**
  `/srv/cold-storage/hass/frigate/backups/cat-projector-h264-20261018/config.yaml`
  SHA-256 `a87689de0a2acb6af46b83f2bbc9f3f422d037ad2ef752574b80c78b0b1a1c64`
  — the pre-H264-repair state. Retained byte-for-byte, not touched.
- **Bad baseline receipt:** `frigate-h264-20261018.md` in this directory and
  `frigate-click-select-v2.json` (SHA-256
  `11c04020663fcb79e5a0432d8f5a6482bc151e5118528d5b116787b9dbd246cf`) —
  retained as the bad baseline, NOT rewritten.
- Container `frigate-nucat` `ghcr.io/blakeblackshear/frigate:0.18.0`, version
  `0.18.0-77a66e7`; API `:15000`, go2rtc `:1984`, RTSP `:8554`; embedded
  go2rtc `1.9.13`, FFmpeg `8.0`.
- Camera publisher `.39` `http://192.168.100.39:8081/video?fps=25` unchanged.

### Bad proof (frozen, from the prior lane)

`frigate-click-select-v2.json` (retained byte-for-byte):

```text
decoder frame deliveries:  +9323 totalVideoFrames over 11945 ms wall = ~780 fps
dropped:                   +9234 of +9323 = 99.0%
WS binary frames:          15268 over 14523 ms = ~1051 fps
source rate:               ~8 fps (MJPEG publisher)
bufferedLead:              0.773 s -> 3.717 s (grew ~2.94 s, unbounded)
media clock:               9.43 s over 11.945 s wall = 0.79x realtime
```

Consistent with FFmpeg CFR duplication on the `ffmpeg:cat_projector#video=h264`
leg. No additional bad proof was needed.

## 2. Applied change (target-derivative only)

Exactly one line changed; all other content byte-identical to the pre-change
live config (`767cb993...`):

```text
go2rtc.streams (line 56):
-  cat_projector_h264: ffmpeg:cat_projector#video=h264
+  cat_projector_h264: ffmpeg:cat_projector#video=h264#raw=-fps_mode:v passthrough
```

The `cat_projector` source stream (line 55) is EXACT unchanged:
`http://192.168.100.39:8081/video?fps=25`.
`Cat_projector_camera.live.streams.main_stream` stays `cat_projector_h264`
(line 164) — unchanged by this repair.
No arbitrary `-r` cap; no codec/quality/resolution/safety change.

**Supported option basis:** official go2rtc
`internal/ffmpeg/README.md` documents `raw` — "You can use `raw` param for any
additional FFmpeg arguments (ex. `#raw=-vf transpose=1`)" — and
`-fps_mode:v passthrough` is a standard FFmpeg output option that preserves
input timestamps (no CFR duplication).

## 3. Backup

Current (pre-change, `767cb993...`) config copied to a UNIQUE path:

```text
/srv/cold-storage/hass/frigate/backups/cat-projector-pacing-pre-20261018/config.yaml
  SHA-256 767cb9935204ffe45ef33b5ed1521669347d1ec0997a055dfb73cc7120f3d51b
```

The baseline `a87689de...` backup was NOT replaced.

## 4. Validation (deployed image's own parser, via stdin)

`docker exec -i frigate-nucat python3` feeding the live YAML through
`FrigateConfig.parse_yaml` (as instructed — no `docker cp` retries):

```text
VALID
main_stream: cat_projector_h264
input path: rtsp://127.0.0.1:8554/cat_projector
input_args: preset-rtsp-restream-low-latency
detect: True 1280 720 5
record enabled: True snapshots enabled: True
cat_projector_h264: ffmpeg:cat_projector#video=h264#raw=-fps_mode:v passthrough
cat_projector: http://192.168.100.39:8081/video?fps=25
other cameras: Living_room_camera, Bedroom_camera, Hallway_camera,
  Xiaomi_balcony_camera, Xiaomi_doorbell_camera, Tab70_camera
```

### Semantic compare vs pre-change backup — 6/6 PASS

```text
PASS | cat_projector EXACT unchanged | http://192.168.100.39:8081/video?fps=25
PASS | cat_projector_h264 ONLY diff is raw suffix | ('ffmpeg:cat_projector#video=h264#raw=-fps_mode:v passthrough', 'ffmpeg:cat_projector#video=h264')
PASS | all other go2rtc streams equal | 19
PASS | cat camera block EXACT unchanged | main_stream=cat_projector_h264
PASS | 6 other cameras byte-equal | Living, Bedroom, Hallway, Xiaomi_balcony, Xiaomi_doorbell, Tab70
PASS | top-level config equal | classification, detectors, face_recognition, ffmpeg, lpr, mqtt, record, semantic_search, snapshots, version
OVERALL: PASS
```

The ONLY semantic difference is the `#raw=-fps_mode:v passthrough` suffix on
the target derivative. Live config SHA after apply:
`649c5e97e08e639532d1dbee4ccfd7b2f542969ffb62edd3705ce3396f6be446`.

## 5. Restart and post-restart readback

- Notified `TraceOpticalResidual` before the corrective restart (its direct
  camera is independent; no overlap).
- One `docker restart frigate-nucat` (exit 0, 18 s). Container `Up (healthy)`,
  API `0.18.0-77a66e7` after 18 s. No blind restart loop, no `docker compose up`.
- `GET :1984/api/streams?src=cat_projector_h264` producer:
  `ffmpeg:cat_projector#video=h264#raw=-fps_mode:v passthrough` — the arg landed.
- No `#raw` consumer while idle (transcoder is on-demand; "no consumer => no
  encoder" was not re-run as an idle probe).

### Settled `/api/stats` after fix

| camera | fps | process | skipped | reconnects | quality |
| --- | --- | --- | --- | --- | --- |
| Living_room_camera | 5.0 | 5.0 | 0.0 | 0 | excellent |
| Bedroom_camera | 5.0 | 5.0 | 0.0 | 0 | excellent |
| Hallway_camera | 5.0 | 5.0 | 0.0 | 0 | excellent |
| Xiaomi_balcony_camera | 5.0 | 5.0 | 0.0 | 0 | excellent |
| Xiaomi_doorbell_camera | 0 | 0 | 0 | 0 | unusable (baseline identical) |
| Cat_projector_camera | 5.0 | 5.0 | 0.0 | 0 | excellent |
| Tab70_camera | 3.0 | 3.0 | 0.0 | 0 | fair (baseline identical) |

All cameras match their pre-change baseline; no regression.

## 6. First causal active-command result (bounded RTSP consumer)

A 10 s bounded RTSP consumer on `rtsp://127.0.0.1:8554/cat_projector_h264`
activated the on-demand transcoder; its exact encoder cmdline was captured
from container `/proc/<pid>/cmdline` (first causal proof of the actual
argument):

```text
/usr/lib/ffmpeg/8.0/bin/ffmpeg -hide_banner -v error -fflags nobuffer -flags
low_delay -timeout 5000000 -user_agent go2rtc/ffmpeg -rtsp_flags prefer_tcp
-i rtsp://127.0.0.1:8554/cat_projector?video&source=ffmpeg:cat_projector%23video%3Dh264%23raw%3D-fps_mode%3Av+passthrough
-fps_mode:v passthrough
-c:v libx264 -g 50 -profile:v high -level:v 4.1 -preset:v superfast -tune:v
zerolatency -pix_fmt:v yuv420p -an -user_agent ffmpeg/go2rtc -rtsp_transport
tcp -f rtsp rtsp://127.0.0.1:8554/72c605d43d11454da1a59f2ef79011ae
```

**Verification the exact argument actually landed:** the output option
`-fps_mode:v passthrough` is present as a standalone FFmpeg argument between
the input and `-c:v libx264`, and the go2rtc source URL carries
`%23raw%3D-fps_mode%3Av+passthrough` (the `#raw=...` fragment go2rtc parsed).
The consumer's own input leg also reported
`packets read` 123 over 10 s and the encoder process was cleaned up after the
consumer disconnected (PID 4371 gone).

### Source/output frame PTS/cadence during the active window

The bounded consumer's own progress/`out_time` (no arbitrary `-r` cap,
`-c copy`/`-fps_mode:v passthrough` on the consumer leg to preserve capture
timestamps):

```text
frame count over the 10.05 s output window: 121 frames muxed, 123 packets read
cadence: frame deltas ~13-17 frames per ~0.4-0.6 s slice => ~14-26 fps,
  settling to the source cadence; NOT ~780 fps, NOT ~1000 fps
media output time: 00:00:10.05 over the 10 s window (pacing tracks wall,
  no unbounded growth)
```

## 7. Browser consumer proof (pacing-r1, isolated copy, fresh output)

Script: `/tmp/frigate-click-select-pacing.mjs` (isolated copy of
`/tmp/frigate-click-select.mjs`; `-v2` outputs NOT overwritten). Run
2026-10-18 02:11:08 Z (`06:11 +04`). Output:
`frigate-click-select-pacing-r1.json` + `frigate-*-pacing-r1.png`.

Selected-stream WebSocket after clicking
`div[data-camera=Cat_projector_camera]`:

```text
ws://192.168.100.74:15000/live/mse/api/ws?src=cat_projector_h264
  firstText {"type":"mse","value":"video/mp4; codecs=\"avc1.640029\""}
  recvBin 151 frames, recvBytes 1916488, firstBinAt->lastBinAt advancing
  (151 binary WS messages over ~12.9 s => ~12 fps — WS delivery cadence)
codecNoise: []  (0 codec/MSE/JSMPEG errors — codec negotiation still accepted)
failedRequests: 3 = clips/previews mp4 net::ERR_ABORTED (unrelated tile)
```

Decoded video evidence (32 samples, wall span 11804 ms, in
`#Cat_projector_camera`):

```text
samplesWithVideo 32, samplesPlaying 32 (media time advancing)
firstCt 0.596 s -> lastCt 12.178 s (media advance 11.582 s)
decoder frames: total 11 -> 145 (+134) over 11804 ms => ~11 fps
                dropped 2 -> 7 (+5) => ~3.7% dropped (was 99.0%)
                rendered (total-dropped) 9 -> 138
bufferedLead:   0.192 s -> 0.411 s (grew ~0.22 s over the run; bounded)
                vs bad baseline 0.773 -> 3.717 s
readyState 4, paused false, 1280x720, display block, src blob:
```

### 7a. Acceptance outcome (A/B vs frozen bad baseline)

| metric | bad `frigate-click-select-v2.json` | fixed `pacing-r1` |
| --- | --- | --- |
| decoder frames over window | +9323 (~780 fps) | +134 (~11 fps) |
| dropped | +9234 / 99.0% | +5 / 3.7% |
| WS binary cadence | ~1051 fps | ~12 fps |
| bufferedLead | 0.773 -> 3.717 s (unbounded) | 0.192 -> 0.411 s (bounded) |
| media clock vs wall | 0.79x | 11.58 s / 11.80 s = 0.98x |

**Verdict: PASS on the pacing repair row.** Total/dropped fps are now near the
actual source cadence (~8 fps MJPEG source; ~11 fps decoded, ~12 fps WS), NOT
1000. Buffered lead no longer grows unbounded (bounded 0.19-0.41 s). Media
clock advances sensibly vs wall (0.98x vs the prior 0.79x). Codec negotiation
remains accepted (MSE `avc1.640029`, no JSMPEG fallback, `codecNoise: []`).
No arbitrary `-r` cap, no codec/quality/resolution/safety change was made.

**Raw samples + screen names (new, `-v2` retained):**
`frigate-click-select-pacing-r1.json`,
`frigate-dashboard-baseline-pacing-r1.png`,
`frigate-selected-after-click-pacing-r1.png`,
`frigate-selected-final-pacing-r1.png`,
`frigate-target-crop-pacing-r1-0..7.png`.

**WS binary messages are not decoded frames** — the WS `recvBin` count is
delivery cadence only; decoded-frame counts come from the `totalVideoFrames`
samples above.

## 8. Rollback

Rollback if the option had failed (it did not): restore the byte-exact
pre-change config
`/srv/cold-storage/hass/frigate/backups/cat-projector-pacing-pre-20261018/config.yaml`
(SHA `767cb993...`), validate with `FrigateConfig.parse_yaml`, one
`docker restart frigate-nucat`, read back `/api/config/raw` +
`/api/go2rtc/streams` + `/api/stats`. No new pipeline was invented; no
rollback was needed.

## 9. Controls and ownership

- No `.39`/HA control, no physical projector mutation, no git, no integration
  latency claim (`<100` not proven). Scoped smoke only (existing browser reuse
  with fresh output); no full-repo or sandbox campaign.
- `TraceOpticalResidual` notified before the one corrective restart.
- Old `frigate-h264-20261018` receipt retained as the bad baseline; linked here,
  not rewritten; raw `-v2` samples untouched.

### Oracle (pacing row)

Actual oracle for the pacing repair = the `pacing-r1` numbers in §7/§7a:
decoded frames near source (~11 fps), dropped ~3.7% (not 99%), bufferedLead
bounded (0.19-0.41 s), media 0.98x wall, plus the captured encoder command in
§6 showing `-fps_mode:v passthrough` actually landed.
