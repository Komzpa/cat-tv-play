# Frozen Optical Source Publication — 2026-10-18

Agent: PublishFrozenOpticalSources. Sole git owner. Prior dispatch `PublishFrozenOpticalSources` was REFUSED and never started; this run is the first execution.

## Freeze basis

- TraceOpticalResidual explicitly froze both paths for narrow publication in its transcript: "Freezing `ProjectorDisplayActivity.java` and the harness — untouched so far and staying untouched."
- Trace edits only `CameraServerService.java` (callback buffers) — excluded from staging.

## Renderer — cat-projector-camera

- Path: `src/camera/catprojector/publisher/ProjectorDisplayActivity.java`
- Frozen SHA256 (confirmed on disk and in committed blob): `63eb95faf8ad5c773cec60b6e6abc3f517dfef3270d709f142fbe3d9a3533764`
- Branch: `main` (existing GitButler workspace branch `ma`); selective file commit ID `xp`, commit change-ID `uvr`
- Commit: `56c002e81e1de6025641748c65cf28c871e1dc3a` — `fix: recover projector playback after failed activation`
- Files changed in commit: only `src/camera/catprojector/publisher/ProjectorDisplayActivity.java` (verified via `but show uvr`)
- Push: `git push origin 56c002e81e1de6025641748c65cf28c871e1dc3a:refs/heads/main` → `12d1889..56c002e`
- Readback after fetch: `origin/main` = `56c002e81e1de6025641748c65cf28c871e1dc3a`, log: 56c002e over 12d1889. Verified.

## Optical harness — cat-tv-play

- Path: `scripts/cat_projector_realtime_rect_proof.py`
- Frozen SHA256 (confirmed on disk and in committed blob): `6c09e0c870fdc8021069553d3bac67a9aa6bcd23db71b678a9c76173f5f75238`
- Branch: existing GitButler branch `fix/dynamic-cat-tv-source` (previously published `e2a9162df81c58f1c091c4dafb06991e13743a7e`); selective file commit ID `op`, commit change-ID `zqy`
- Commit: `ade13cf98a3ee74167e719cb77fef2ef6706fb9b` — `test: measure causal projector optical round trip`
- Files changed in commit: only `scripts/cat_projector_realtime_rect_proof.py` (verified via `but show zqy`)
- Push: `git push origin ade13cf98a3ee74167e719cb77fef2ef6706fb9b:refs/heads/fix/dynamic-cat-tv-source` → `e2a9162..ade13cf`
- Readback after fetch: `origin/fix/dynamic-cat-tv-source` = `ade13cf98a3ee74167e719cb77fef2ef6706fb9b`, log: ade13cf over e2a9162. Verified.

## Preserved state

- Not staged: `CameraServerService.java` (Trace's live edits, remains uncommitted in cat-projector-camera `zz`).
- Excluded per allowlist: all other publisher/overlay files, READMEs, APKs, marker assets, mutable old receipts — all remain uncommitted in `zz` for the final-publication row.
- No force push, no wrapper-HEAD push, no reset/stash/rebase/checkout, no source edit/build/test/deploy.
- Unrelated PID 1273442 (WeeChat) untouched.

## Freeze / release

- Freezes on both published paths released after verified remote readback (release notices sent to TraceOpticalResidual and Main).
- Renderer ack note: Trace's exact-path freeze acknowledgment predates staging renderer, satisfying the pre-staging gate; harness was independently publishable regardless.
