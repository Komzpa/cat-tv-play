# Cat Projector TikTok Render Requirements

The TikTok/Telegram jump render is a product video, not a detector-debug clip.
Future jump alerts and manual publish renders must satisfy these invariants.

## Visual Story

- Start from a full-room/full-projector view so the viewer understands the wall,
  floor, cat, and jump context.
- Smoothly zoom toward the active jump only after the motion starts.
- The zoom target must cover the whole cat trajectory for the rendered jump, so
  the camera settles on one composition instead of chasing the cat frame by
  frame.
- Once the close crop is selected for a jump, keep that trajectory crop fixed
  for every rendered source frame in the clip.
- Keep the cat, height line, and leaderboard inside TikTok safe zones: do not
  depend on content near the top app chrome or the bottom caption/action bar.
- Do not render duplicated or torn-looking background layers. Padding or blurred
  fill must never look like a second unsynchronised video.
- Render a longer clip when there is enough source material. If the layout shows
  three record slots, the video should include up to three distinct jump moments
  rather than one jump plus two decorative empty slots.

## Leaderboard

- Keep the image-card leaderboard style: visible slots with thumbnails from the
  beginning of the clip.
- During a jump, the current slot updates with the best frame seen so far while
  the cat is rising.
- After the peak, that jump slot freezes on the best peak frame.
- Slots show viewer-facing jump facts only: rank, height, and short status.
  Detector internals such as model type, probability, chunk id, frame id, or
  file names belong in sidecar JSON/review metadata, not in the TikTok video.
- Do not show internal counters such as per-hold observation counts.

## Cat Tracking

- Track and draw the cat per source frame. Never reuse the peak-frame contour or
  box across unrelated frames.
- The TikTok overlay should prefer the cat contour/mask when available. A box is
  only an explicit fallback for frames where a contour is absent.
- The current model/review output must use the same source-subtracted context as
  live detection.

## Height Line

- The displayed height line is a current-best-so-far wall height line, not the
  final maximum drawn from frame zero.
- The line must be computed from the same calibrated candidate that drives the
  leaderboard and the caption height.
- At the peak frame, the height line must pass through or above the measured cat
  top. It must never appear below the cat for the displayed height.
- Keep the max/current-best line visible once a meaningful best frame exists.

## Frame Rate And Enhancement

- Output FPS must match real motion cadence. Do not duplicate low-FPS review
  frames and label the result as 30 or 40 FPS.
- Prefer actual source-video frames for timing. Review extracted frames are
  acceptable for detector evidence, not as the only motion timeline.
- Publish renders must record the exact source chunk and frame window used for
  the video in sidecar metadata.
- Neural enhancement must run when the local tools and models are installed.
  Install missing local binaries/models instead of silently reporting
  `neural_upscale: unavailable`.
- Frame interpolation must not create ghost overlays. Run interpolation on raw
  source frames before overlays, or regenerate overlays for every output frame.
- Burned overlays may be upscaled after rendering, but that is visual polish; it
  does not fix fake frame cadence.

## Telegram Contract

- Future Telegram jump alerts must use this same publish renderer.
- The sidecar metadata should record the model, source-subtracted/legacy path,
  probability, threshold, neural tools, source FPS, output FPS, and selected jump
  candidates.
- The video itself remains clean and viewer-facing.
