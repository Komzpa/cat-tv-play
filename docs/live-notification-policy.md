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
- The original-footage oracle includes the cat-only June 6 source chunk
  `20260606T010937_20260606T010921-cats-tv-october-compilation-fMiAfGtt9rc-seed30/chunk_0124.mp4`;
  it MUST remain negative while all three known-human windows in
  `test_cat_projector_clip_policy.py` remain positive, the May 27 cat-only
  control remains negative, and all six fail-closed error cases remain blocked.
