# Source Subtraction

When the projected clip is known, review tooling should subtract that clip from
the camera frame before proposing cat candidates. Otherwise the detector wastes
capacity on projected prey, shadows from the clip, or screen-edge artifacts.

The portable helper in `custom_components/cat_tv_play/source_subtraction.py`
does this:

1. Warp the current source-video frame into the camera projector polygon.
2. Match brightness between the expected projection and the camera frame.
3. Subtract the expected projector signal from the camera frame.
4. Optionally subtract a residual baseline for static wall/projector mismatch.
5. Return remaining dark residual components as candidates.

Those candidates are not final cat detections. A downstream model or human
review still decides whether each candidate is the cat.

Do not build the room background from arbitrary active-session pixels. If the
cat sits still for most of the clip, median or percentile aggregation will learn
the cat as part of the background and erase it.

For live review, keep an adaptive background but update it only through a mask
that excludes every pixel where the cat could plausibly be. Draw this exclusion
generously: the floor line, stool/chair area, lower wall, entry side, and the
observed jump corridor. Pixels outside that path can keep adapting to IR
exposure and projector brightness changes; pixels inside the path keep their
last no-cat value until a reviewed no-cat frame refreshes them.

## Inputs

- `camera_frame`: the review-camera frame.
- `source_frame`: the frame currently being played by the projector.
- `projector_polygon`: the four camera-image corners of the projected source
  plane.
- `room_background`: optional camera background built from no-cat frames; this
  is what keeps a static sitting cat visible outside the projected rectangle.
- `update_mask`: optional background-learning mask. `True` means a pixel may
  update from the current frame; keep possible-cat pixels `False`.
- `residual_baseline`: optional percentile residual image computed over nearby
  frames to remove stable calibration and brightness mismatch.

## Why This Matters

For Sher's pilot, the old detector sometimes boxed the projected toy or a table
object while missing the visible cat. Source subtraction changes the candidate
proposal problem from "find dark things in the whole camera image" to "find
physical things that are not explained by the known projector frame".

That still needs model training. The model must see positive and negative
examples from source-subtracted candidates, especially occluded lower-wall cats,
stool/table false positives, and projected prey leakage.

## Eye-Safety Overlay

The live eye-safety path uses the same projector-plane geometry in the opposite
direction. `custom_components/cat_tv_play/projector_safety.py` takes person
detections in projector-camera pixels, intersects the padded upper head band
with the projected wall polygon, maps that overlap back into source-video
coordinates, and returns black overlay polygons.

`scripts/cat_projector_safety_overlay_server.py` is the runtime wrapper. It
loops the source video, samples the projector camera, renders the black zones,
and serves `/stream.m3u8` plus `/status.json`. The default policy is fail-open:
if the detector, camera, or calibration is unavailable, playback continues
unchanged and the status becomes `safety_overlay_unavailable`.
