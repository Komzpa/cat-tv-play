# Capture & Transport Latency — camera -> publisher HTTP/MJPEG ingress (read-only)

Scope: camera sensor -> publisher ingress only. Downstream (go2rtc remux ->
Frigate decode -> detect -> overlay render -> projector display) is OverlayTrace's
lane. No restart, no config, no source mutation. Only this receipt written.
Run window 2026-10-18 ~01:42-02:00 +04. cwd /home/kom/Dropbox/Reemxy/tasks-loop.

## Source / deployed owner mapping (HTTP-constant compatible; APK identity not established)

- Canonical source: /home/kom/proj/cat-projector-camera/src/camera/catprojector/publisher/CameraServerService.java
  (git git@github.com:Komzpa/cat-projector-camera.git, HEAD 3ba40e5), package
  camera.catprojector.publisher.
- Deployed = live device 192.168.100.39:8081, HTTP body "Cat Projector Camera".
  HTTP constants match the live probe: PORT=8081, 1280x720, JPEG_QUALITY=90,
  DEFAULT/FPS bounds 30/1/30, ROTATE_180=true, FLIP_HORIZONTAL=true, endpoint
  /status.json. Live /status.json returned {"status":"ok 1280x720","width":1280,
  "height":720,"jpeg_quality":90,"default_mjpeg_fps":30,"min_mjpeg_fps":1,
  "max_mjpeg_fps":30,"latest_jpeg_bytes":40485}. These HTTP constants establish a
  compatible source; they do NOT establish exact APK identity. Camera source
  identity and the remaining capture-age gap are owned by the running CameraRepair
  worker.
- NOT the tablet-mesh publisher. 192.168.100.46:8081 ("Home Mesh", /status,
  quality 85, 20fps cap) runs
  /home/kom/proj/home-audio-mesh/deploy/android/tablet-mesh-source/
  (package mesh.homeaudio.tabletsource) = Tab70 kitchen-fridge tablet, a
  separate camera. Earlier detour corrected.
- Renderer (OverlayTrace lane, source located here for handoff): same repo,
  /home/kom/proj/cat-projector-camera/src/camera/catprojector/publisher/ProjectorDisplayActivity.java
  (package camera.catprojector.display; plays source mp4 + pulls blackout zones
  from http://192.168.100.74:8787/status.json).

## Transport format — no inter-frame encoder exists

Publisher is MJPEG: one independent software JPEG per frame
(YuvImage.compressToJpeg, quality 90, 1280x720, ~39-40 KB/frame). There is NO
H.264/H.265 hardware encoder, therefore NO GOP, NO B-frames, NO reference-frame
queue anywhere in capture->ingress. Those latency terms are structurally N/A,
not merely small. go2rtc confirms codec_name=mjpeg, RTP JPEG/90000.

## Measured values (count / window / unit)

1. MJPEG arrival cadence, direct /video?fps=30 from .39:
   169 frames / 6.00 s = 28.17 eff fps. Inter-arrival ms: min 4.0, p50 35.5,
   mean 35.6, p95 54.3, max 143.3, stdev 14.2. Low jitter = steady, no backlog.
2. Frame-advancement age proxy, /shot.jpg polled 12x / ~1.8 s:
   12 of 12 distinct SHA-256, bytes 39258-39539. Capture advancing, not frozen.
3. Network ping .39: 15 tx / 15 rx, 0% loss, RTT min/avg/max/mdev =
   3.279/13.920/42.294/9.265 ms. Clean LAN.
4. go2rtc ingest stability (nucat:1984/api/streams?src=cat_projector), two
   samples ~4 s apart: producer_id=21 STABLE (no reconnect churn); recv_packets
   344469 -> 344576 (+107/4s ~ 26.8 pps); bytes_recv 21998663966 ->
   22002871662 (+4.22 MB/4s ~ 1.05 MB/s); 1 consumer (rtsp+tcp, FFmpeg
   Frigate/0.18.0). Live, flowing, no backlog.

## Confirmed bottleneck vs hypotheses (in scope)

CONFIRMED minor (CPU, not a queue):
- Per-pixel Java transformNv21() (180 rotate + h-flip) runs on the SINGLE camera
  callback buffer, then software JPEG encode, before publish. CPU transform
  dominance and the <=1-frame latest-frame age bound are UNMEASURED here; the
  single callback buffer structure is inferred from source, and a slow frame
  drops an internal frame rather than queuing. Effective fps measured at 28.17
  (see Measured values #1); whether the transform is the dominant per-frame cost
  is not established by these probes.
- latestJpeg volatile slot + writeMjpeg sleep loop: a consumer reads the newest
  completed frame; inherent staleness <= 1 frame interval (~35 ms @30fps,
  ~40 ms @go2rtc fps=25). Structural to latest-frame MJPEG.

NOT a bottleneck (evidence):
- Transport: 0% loss, 13.9 ms avg RTT, stable producer id=21, ~27 pps steady,
  low inter-arrival jitter. The 15 pings sampled only RTT and loss; they do not
  measure one-way delay or capture-side timing, so no latency-accumulation claim
  in capture->ingress is made here.

Absolute capture age is UNKNOWN. The publisher exposes no capture timestamp and
no frame counter, so device-side capture->ingress wall-clock age cannot be
produced from this lane. OverlayTrace's independent trace reports: branch A
(Frigate snapshot/live) lags direct camera ~600 ms (n=91, corr 0.974); the
safety-overlay branch B is direct-camera ~130 ms p50 and shares no stage with A
beyond the publisher. The ~130 ms is a partial budget, not end-to-end. Whether
the ~600 ms is downstream of this lane, or includes an unmeasured capture-side
component, is not established by these probes; the remaining gap is owned by the
running CameraRepair worker.

## Precise age-proxy gap

The .39 publisher publishes NO capture timestamp and NO frame counter:
/status.json exposes only status/width/height/jpeg_quality/fps-bounds/
latest_jpeg_bytes. (Contrast tablet-mesh /status, which has frame_count +
last_frame_unix_ms.) A true device-side capture->ingress wall-clock age CANNOT
be produced from this publisher. Defensible proxies used above: (1) distinct
snapshot hashes prove advancement; (2) steady low-jitter inter-arrival and 12
distinct snapshots cannot bound sensor age and cannot rule out a stable
backlog — they show the transport path is delivering frames steadily, not that
capture-side age is small.

## Narrow fixes (in scope, proposed - NOT applied; read-only phase)

1. Close the age-proxy gap: add frame_count + last_frame_unix_ms to
   /status.json (mirror tablet-mesh). Makes capture->ingress age measurable
   instead of inferred. Owner: cat-projector-camera repo.
   Oracle: /status.json carries two advancing ints; age = read_clock -
   last_frame_unix_ms, expected tens of ms given cadence above.
2. Cut per-frame CPU: replace per-pixel transformNv21 with bulk System.arraycopy
   row-reversal (180 = reverse row order + reverse within row). No fps promise is
   made here; whether this raises effective fps is unmeasured and is not a claim
   of this receipt. Oracle: re-run /video?fps=30 probe -> eff fps 28.17 ->
   outcome unmeasured.
3. Only if end-to-end must drop further after OverlayTrace bounds downstream:
   this MJPEG path cannot shed latest-frame staleness; low-latency H.264
   hardware encode is the structural change (out of this lane; flagged for
   overlay/display owners).

## Commands / exit status

Probes: curl 192.168.100.39:8081 {/,/status.json,/shot.jpg,/video?fps=30};
ping -c 15 192.168.100.39; curl 192.168.100.74:1984/api/streams?src=cat_projector;
bounded Python socket/urllib timing loops. All probes returned measured data; no
failures. Resource warnings: none — read-only, bounded single probes, no desktop
use, no credentials emitted.
