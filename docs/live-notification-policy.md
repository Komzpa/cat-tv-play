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
