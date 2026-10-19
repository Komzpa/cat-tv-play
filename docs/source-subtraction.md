# Source Subtraction

Source explanation is candidate-local evidence, not a way to manufacture a
source clock or refit the current camera frame until its detections disappear.
Ordinary camera-only proposals and their model scores run first. Missing,
invalid, or inapplicable projection evidence leaves those detections intact.

`custom_components/cat_tv_play/source_subtraction.py` owns the response model
and `explain_projected_candidate` operation. Its result separates:

1. `structural_valid`: native camera/source correspondence is supported.
2. `photometric_valid`: independently validated response applies to this capture,
   geometry and candidate color domain, with physical **unclipped** predictions.
3. `explained`: every surviving native source alternative explains the actual
   candidate pixels within the frozen disjoint-validation error.

A real physical object can have valid correspondence and calibration while
remaining **unexplained**. This returns `photometric_valid=true` and
`explained=false`, retaining candidate error, per-family errors and reasons.
Only `explained=true` can remove an ordinary cat candidate. A near-white or blank
wall fit alone never explains colored prey, and choosing the lowest-residual
source family or fitting disputed candidate pixels is forbidden.

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

- `camera_frame`: the RGB pixels of the exact native camera identity, not an
  unbound JPEG or a nearest nominal frame.
- `correspondence`: `FrameCorrespondence` from the canonical
  `cat_tv_play_source_correspondence` module, including native identities,
  geometry, camera-only object support and every surviving source family.
- `candidate_mask`: the nonempty native-camera proposal mask, or the entire
  native bounding box when no pixel mask is available. It must intersect the
  verified camera-only structural support, but may include white background
  and holes outside that dark-component support. Evaluate every pixel of the
  original candidate for native-projection coverage, color coverage, unclipped
  prediction and residual error; never intersect/crop it to structural support
  to exclude an occluder or missing projection/calibration coverage.
  Support/hash/geometry mismatches cannot authorize subtraction.
- `photometry`: optional frozen `PhotometricEvidence`. It binds camera media,
  source media, geometry, fit/validation RGB identities, exclusion/support mask
  hashes, zero candidate-overlap counts, independently covered color bins,
  unclipped prediction ranges and measured validation error. Fit and validation
  frames are disjoint; neither includes the assessed camera frame.

The canonical model is **`projector-rgb10-v1`**, the existing ten-feature RGB
response (R, G, B, max, min, mean, sRGB luma, saturation, square-root max and
bias). The diagnostic six-column RGB/illumination-plane model is different and
is not a selectable production fallback. No consumer performs a final per-frame
fit, grayscale fallback, prediction clipping or candidate-family residual shop.
Every source-color bin used by the actual candidate must be independently
covered in both fit and validation. Check that domain before evaluating the
response: an unsupported color is missing calibration, not an observed physical
prediction failure. Unrelated colors are not prerequisites: a blue-only
candidate does not require red calibration, while neutral-only wall support
still cannot cover a colored candidate. Unresolved source alternatives or
exposure mixtures cannot be silently resolved by calibration.

`candidate_families` must contain genuine surviving per-frame structural
hypotheses, not structurally rejected negative controls or every identity in an
unsearched adjacent bracket. Negative controls remain in sequence evidence.
The consumer must not repair a polluted hypothesis set by dropping all
unresolved entries or choosing a best-residual frame. Missing per-frame
plausibility is a producer prerequisite, not permission to suppress candidates.

Camera-only components are retrieval proposals, not no-cat labels or proof that
every proposed pixel is projected. Qualification evidence must bind matched
components to the exact camera and native source identities; unmatched proposals
may be physical objects or occluders. Fit/validation masks may use only separately
grounded, uncontaminated projected support. Do not fit the entire proposal mask
by assumption or remove poorly predicted pixels afterwards to manufacture valid
calibration.

Changing native alternatives are possible colors or exposure contributors at
the same camera pixel, not simultaneously exact independent RGB observations.
Calibration eligibility and any source-uncertainty bound must come from native
source variation and independently qualified geometry, with the disputed object,
its halo and physical occluders excluded. Never choose calibration pixels,
exposure weights or a surviving family by their camera residuals.

The retained source-invariance assessment records a tolerance-free sufficient
condition for exact RGB pairing: every genuine native alternative gives the
same warped RGB at the calibration pixel. Such a pixel supplies **one** camera
equation, not one independently weighted equation per alternative. This is not
a requirement that all useful future calibration be exactly invariant. Varying
colors need a separately justified source/exposure uncertainty treatment; neither
four-bit-bin agreement nor an arbitrary exposure average is exact RGB evidence.
Fit and validation must also retain disjoint native-source identities, not just
different camera frame numbers. A neutral-domain response can be valid without
covering a projected colored object. Report that missing candidate domain and
keep ordinary detections; do not promote the neutral fit or all-reject consumer
result to independent projected-object acceptance.

`DetectorContext.source_correspondence` and `.photometric_evidence` carry this
typed input to `LegacyContrastDetector.detect`. Both C and tasks-loop legacy
frame-detector subtraction implementations have been removed; the frame
detectors now propose ordinary contrast candidates. Unbound reference lists,
chunk-index arithmetic, scalar offsets and modulo frames are not this API.

Room-background helpers remain separate: use reviewed no-cat frames and an
`update_mask` that excludes the possible cat path. They are not photometric
evidence and cannot grant source-explanation permission.

## Why This Matters

Sher's detector has boxed projected toys while missing the visible cat. The
cutover therefore preserves ordinary detections unless the projected object is
actually explained, rather than treating subtraction availability as success.

The retained July 12 `chunk_0301` evidence has a structural 1559-second ordered
family, but structural matching does not validate camera color response. The
saved near-white outside-object diagnostic fit has RMSE 6.75 on its own wall
support and roughly 156–299 luma RMSE on colored prey, including physically
impossible predictions. It remains failed evidence, not a calibration to clip
or transfer to May. May 27 `chunk_0026` and the May 18 sent-jump regression are
different capture inputs; July calibration cannot establish their applicability.

Acceptance requires a real projected-object positive with candidate-excluded,
disjoint fit and validation under the same capture regime, plus an unexplained
physical-object control. Synthetic tests exercise API invariants only. An
all-reject result, raw-model no-human verdict, or source-unavailable receipt does
not close that evidence gap. Without retained independent colored support,
record its exact native identities, masks and missing colors; do not fit the
disputed object or start an unapproved display/acquisition run.

The retained-only `independent-explanation-assessment-20261019.json` receipt in
`receipts/projected-source-correspondence-local-20261018/` is conditional on its
recorded producer revision and family set. Its zero exact-bin-stable chromatic
support is not proof that external calibration is absent or impossible: rejected
controls and broad adjacent brackets were still carried as surviving families.
The real consumer preserved all four ordinary detections and explained none;
that is fail-closed behavior, not the required independent positive. Correct the
producer's actual hypothesis artifact before further calibration assessment,
without discarding genuine ambiguity or transferring July evidence to May.

## Eye-Safety Overlay

The live eye-safety path uses the same projector-plane geometry in the opposite
direction. `custom_components/cat_tv_play/projector_safety.py` takes person
detections in projector-camera pixels, intersects a narrow eye-slot inside the
top of the person box with the projected wall polygon, maps that overlap back
into source-video coordinates, and returns black overlay polygons plus the
camera-space eye-slot box used for debug review. The default slot is
deliberately small: it is for eyes, not for dimming the whole person.

`scripts/cat_projector_safety_overlay_server.py` is the runtime wrapper. In
production it should run with `--status-only`: the projector Android app plays
the original MP4 natively while the server samples the camera and serves
`/status.json`, `/camera.jpg`, and `/debug-camera.jpg`. Every full-frame person
detection reaches the overlay computation; no arbitrary source reference,
whole-video sample, or residual fit suppresses a person. Detector failure is
`safety_overlay_unavailable`, not a clear scene. The older `/stream.m3u8` mode
remains available for isolated tests, not normal native playback.

Production status-only units should also pass the Home Assistant active gate
(`--ha-url` plus `--ha-config-path`). When both the cat-projector session latch
and display-active sensor are explicitly `off`, the server continues serving
status but idles the camera/detector worker so a lamp-off projector does not
burn CPU. HA errors and unknown states fail open: the worker stays active rather
than risking a blind safety overlay during startup.

The player alone is not a physical safety boundary. Home Assistant production
wiring must run an independent watchdog that treats `active`, stale, or
unavailable safety status as a lamp-off command while preserving the
cat-projector session latch.
