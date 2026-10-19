# Live Notification Policy

Cat TV Play owns the behavior policy for jump-highlight delivery, even when a
house-specific bridge owns the actual Telegram transport.

A live skip because a person is visible in the review camera is temporary. The
bridge should write a scan-cooldown marker, then rescan after that cooldown so a
later jump window without the person can still be delivered.

Marker handling:

- `sent`: keep the normal delivery cooldown.
- `no_jump_highlight`: use scan cooldown, then rescan while the session is live.
- `camera_human_detected` / `human_check_unavailable`: use scan cooldown, then
  rescan; do not treat these as permanent session verdicts.

Recorder and delivery requirements:

- Recorder liveness is determined by the owned recorder process and its HA session;
  `verified.json` is a media-quality artifact, not a liveness signal.
- Delivery MUST require `verified.json` produced by the original-media quality
  gate (at least 20 effective fps). A pending verifier MUST remain able to advance
  when a later original chunk passes that gate; rendered/upscaled output is not proof.
- Unknown transport return values MUST be treated as delivery failure. A failed
  delivery MUST NOT be marked sent or dispense a food reward.
  A saved final-notification marker counts as sent only when its delivery receipt
  explicitly confirms `delivered: true`; marker existence alone is not success.
- Human-check rejection and unavailable checks remain fail-closed; no legacy
  candidate may bypass the camera-human gate.
- Reward eligibility remains at least 150 cm, once per HA session, with a global
  one-hour cooldown. Only successfully delivered live-session notifications may
  reward, using the existing two-hour session freshness bound; missing or future
  timestamps are ineligible. Final-session delivery and historic replay MUST NOT
  invoke the food actuator. A delayed live retry MUST recheck freshness.
- Local assessment remains opt-in and side-effect-free. Pending event-linked
  recordings rotate by least-recent attempt, preferring fresh recordings among
  never-attempted candidates. Failed attempts remain queued rather than blocking
  newer coverage or being discarded. Scheduling attempts MUST NOT claim chunk
  coverage; the canonical assessor alone records that coverage. A stored
  `assessment_error:*` result MUST NOT satisfy the assessor's identity cache:
  with unchanged inputs the scorer MUST be invoked again, and only healthy
  non-error results may be returned as cached verdicts. Error results MUST
  continue to claim no chunk coverage, so persistent failures stay visible and
  pending for retry rather than frozen as a permanent record.

- Notification activation MUST remain disabled until the known-human original
  clip windows are rejected by the camera-human detector; a detector result of
  no-human on a known-human window is a release blocker, not an opt-out case.

- Camera-human detection MUST use whole-frame semantic context to resolve
  crop-only results. A person-class box from any crop that overlaps a full-frame
  cat-class box at IoU >= 0.5, with both scores meeting the unchanged confidence
  floor, MUST NOT count as human. Whole-frame person detections remain positive;
  this veto applies only to crop results. This prevents an aspect-distorted cat
  silhouette in a tight crop from overriding the broader cat classification.
  Apply the rule
  to all footage, never by clip path or hash; retain the 0.35 confidence floor,
  4 fps sampling, and 1280-pixel analysis width.
- The original-footage oracle includes both cat-only windows: May 27
  `20260527T220441_20260527T220434-cats-tv-october-compilation-fMiAfGtt9rc-seed30/chunk_0026.mp4`
  and June 6
  `20260606T010937_20260606T010921-cats-tv-october-compilation-fMiAfGtt9rc-seed30/chunk_0124.mp4`.
  Both MUST remain `no_camera_human_detected`, not skipped or unavailable, while
  June 1 `20260601T134003_20260601T133950-cats-tv-october-compilation-fMiAfGtt9rc-seed30/chunk_1176.mp4`
  and October 18 `20261018T154344_20261018T153653-cats-tv-october-skipintro-30fps/bad_chunks/chunk_0002.bad.mp4`
  remain human-positive. The June 5 candidate-5/6 representative frames show
  furniture; their entire windows remain unadjudicated, not cleared.
- The recorder MUST automatically produce guarded per-chunk native
  correspondence before source-aware scoring/rendering/privacy explanation.
  Native sample identity comes from same-decode RGB/PTS, never chunk arithmetic,
  nominal seeks or modulo clocks. Reuse the retained index without a rebuild.
  The recording manifest's `source_correspondence.qualification_controls` value
  (or the canonical `<index-stem>.qualification-controls.json` sidecar when
  unset) MUST be forwarded unchanged into `produce_chunk_correspondence`. This
  control binding is generic and independent of missing valid photometry.
  Missing/invalid/unqualified controls MUST keep ordinary detection and the
  unconditional human veto; a missing sidecar surfaces the producer's
  `unavailable`/`inapplicable` reason and an invalid receipt fails closed. No
  July structural fixture transfers across captures; current-capture ordering
  controls come from the producer, never a competing parser or residual winner.
- Only crop-only person candidates may reach
  `source_subtraction.explain_projected_candidate`. Supply the typed
  `DetectorContext.source_correspondence`/`photometric_evidence`, native RGB and
  the producer's native candidate mask (bbox only if absent). Suppression MUST
  require independent, candidate-excluding frozen photometry valid across every
  surviving source alternative. Missing/invalid evidence preserves ordinary
  detection; structural matching alone is never privacy permission. Full-frame
  persons retain unconditional precedence and errors remain fail closed.
- Acceptance MUST use the production recorder entry on the complete real
  windows at one frozen revision; no skips, fabricated evidence or API-only
  success. Preserve the accepted May 18 native render's 71 PTS/contours and
  existing delivered transport evidence without rerendering or resending.

Lessons learned: sampler ordinals are not native timestamps, structural support
is not validated color response, and a chair observation is not a human label
or whole-window clearance. Keep these evidence boundaries executable.
