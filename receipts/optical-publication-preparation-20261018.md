# Optical Publication Preparation — 2026-10-18 (rev2, QA-corrected)

Agent: OpticalPublicationPreparation. READ-ONLY git inspection across three
repositories; this file is the only write. No commit, push, source edit, build,
test, or live command was run; no recipe here has been executed. Final git
owner publishes from this manifest.

Per skill `but`: local stack mutations belong to `but commit`; remote
publication is an explicit append-only `git push origin <sha>:refs/heads/<fullref>`
(no force, no force-with-lease — none is authorized), read back from the forge
afterwards. Camera repo is plain-git; cat-tv-play is a GitButler workspace;
tasks-loop uses its canonical sync helper / `but` selective commit — never a
raw `git add`/`git commit` recipe.

## Already published (verified by fresh `git ls-remote` this session — do not republish)

- Renderer: `src/camera/catprojector/publisher/ProjectorDisplayActivity.java`
  `63eb95faf8ad5c773cec60b6e6abc3f517dfef3270d709f142fbe3d9a3533764`,
  commit `56c002e81e1de6025641748c65cf28c871e1dc3a` = cat-projector-camera
  `origin/main` (ls-remote confirmed).
- Harness: `scripts/cat_projector_realtime_rect_proof.py`
  `6c09e0c870fdc8021069553d3bac67a9aa6bcd23db71b678a9c76173f5f75238`,
  commit `ade13cf98a3ee74167e719cb77fef2ef6706fb9b` =
  cat-tv-play `origin/fix/dynamic-cat-tv-source` (ls-remote confirmed).
- Descriptive receipt `receipts/frozen-optical-source-publication-20261018.md`
  `f7ead92a688843805805758597d6b6bdc39886fe2ff048f393529a58620af9f7`
  (still uncommitted — in allowlist below).
- tasks-loop prior `f4ef4a0109b8fccd84d4af031503f081c64b8ca3`
  ("ops: pass latest projector camera stream to overlay") is an ancestor of
  `origin/main`; auto-sync absorbed it. Nothing to republish from it.

## Repo 1 — /home/kom/proj/cat-projector-camera (plain git)

- Branch `main`, remote `git@github.com:Komzpa/cat-projector-camera.git`,
  HEAD `56c002e81e1de6025641748c65cf28c871e1dc3a` = `origin/main`
  (ls-remote confirmed; both divergences empty). Working tree CLEAN.
- `6872031c` is the file SHA-256 prefix of the RESTORED source file (the
  rollback restored the known-initial optimized publisher state);
  `82089b37` is the file SHA-256 prefix of the installed APK on the device.
  Neither is a git commit and neither is traced in history — correctly so:
  the restored source equals the published tree, all core initial source
  fixes are already in published commits on `origin/main`.
- Unpublished allowlist: EMPTY. Nothing to stage, nothing to push.
- Preserved locally (gitignored `.gitignore:9 build/`, never stage):
  `build/cat-projector-camera-unsigned.apk`,
  `build/cat-projector-display-unsigned.apk`,
  `build/cat-projector-display.apk`, `build/cat-projector-camera.apk`.
- No action required or pending. Optional readback:
  `git -C /home/kom/proj/cat-projector-camera ls-remote origin main` →
  expect `56c002e81e1de6025641748c65cf28c871e1dc3a`.

## Repo 2 — /home/kom/proj/cat-tv-play (GitButler workspace)

- Checked-out branch `gitbutler/workspace`, workspace HEAD
  `e5f1e8d2c79dd8793334b890f9de2f516d047105`; remote
  `git@github.com:Komzpa/cat-tv-play.git`.
- Publication target: existing branch `fix/dynamic-cat-tv-source`, local =
  remote = `ade13cf98a3ee74167e719cb77fef2ef6706fb9b`. Publication is a plain
  append on top of that head — no rebase, no `but pull`, no force. Remote
  `origin/main` being ahead (`8f26a575d1a2e4aec23c32f316fff271b6c46f2a` vs
  local `main` `cf47f7eb80ad66f5ccc3bfedbda383fefc5f8899`) is irrelevant to
  this append and is NOT a reason to rebase. There is no option to
  retarget publication to `main`.
- Tracked-modified: `README.md` only (`9d18593a389cd8cee667d1c461e97178962d3006eac4e3f5f30ccdfc528a5991`,
  +62/−8): documents direct-MJPEG `--camera-stream-url`, the
  `X-Frame-Sequence` / `X-Frame-Age-Ms` header contract,
  `--camera-max-frame-age` 0.9 s default, and states publisher age + receive
  duration is NOT an optical/sensor-age bound and FPS does not prove low
  end-to-end latency. This is the 2026-10-18 latency requirement text; no
  separate requirements doc update is missing and none was invented
  (`docs/` has no latency requirement file; grep → no hits).

### Final bounded named allowlist — cat-tv-play (SHA-256 measured 2026-10-18; integration pair now FROZEN)

Tracked-modified:
1. `README.md` — `9d18593a389cd8cee667d1c461e97178962d3006eac4e3f5f30ccdfc528a5991`

Frozen integration pair (owner-frozen this session; previously listed
mutable — now included):
2. `receipts/deployed-latency-20261018.md` — `858d770c25482da31ddb53cc16daf45ab5f9dfbafac06ada8887f1d2ec517f95`
   (fresh hash after the line-32 correction; the earlier `8a0041cd…` prefix
   seen in older notes is STALE — use the value above for R1/R2 QA)
3. `receipts/deploy-latency-20261018.md` — `33d9d6c49ee53fc1b6672798e4a3272ad9cae686be6efbadd7fbd33685dc9a58`
   (companion, matches owner-pinned SHA)

Durable evidence receipts:
4. `receipts/frozen-optical-source-publication-20261018.md` — `f7ead92a688843805805758597d6b6bdc39886fe2ff048f393529a58620af9f7`
5. `receipts/frigate-browser-oracle-20261018.md` — `3d7da4ce01f2d33908a1bf0ed5ce8b09043d711d7eb6459cea0bf25a3d34479b` (accepted, frozen)
6. `receipts/frigate-browser-marker-pairs-paced-r1-20261018.json` — `064c1f4ba3cb022affd62b990695b9844b69533f5ebfce6d1d7fa3aa17108f11` (raw pairs)
7. `receipts/frigate-browser-marker-pairs-paced-r1-reanalysis-20261018.json` — `49eceb6ef6c5dbf03d99090a63a5ef61e7b18e1dfdfae198e1cafc6868a89711` (reanalysis)
8. `receipts/frigate-paced-r1-reanalysis-20261018.py` — `458172fbf60c7cc1d7d777d626085ee6edc1d934593d252de801b268c8df77e2` (analysis script)
9. `receipts/frigate-browser-marker-pairs-baseline-jsmpeg-20261018.json` — `064ecf80888cad3473bda6a3d70f23403755222cfd4b8ee9014a7b18c41b3275` (baseline pairs)
10. `receipts/frigate-browser-latency-20261018.md` — `bc603cee071a703e2a9572bca23211d26006ed95953061a3b05151f0f669596c`
11. `receipts/frigate-browser-latency-20261018.json` — `91cb499af22d072e1fbdecfc29d566c93667b69b467736898107731a69b21be3`
12. `receipts/frigate-browser-latency-20261018.png` — `8e639b38e00fb5047d21e130ac39e0fbcf26aeee71b29fa572ca800d26eae161`
13. `receipts/capture-latency-20261018.md` — `d6cb89f603d676f182c78624e2ed3317f9b996a837aa781681dc29617a548642`
14. `receipts/residual-latency-20261018.md` — `27faf457cd1722a52cc54e2a70979d7b7b14230007c29904dcba3d1ee919a253` (the #FC8F corrected + frozen residual receipt)
15. `receipts/overlay-latency-20261018.md` — `75b1b1175768b0ba9f25d5835473c0f36587f60668f0bc576226f976a0f9f30a`
16. `receipts/frigate-pacing-20261018.md` — `0c2c107ab763fc27a4d3a15560ccd2336b39aa7246704e97d9b81ea6c7db28bc`
17. `receipts/frigate-h264-20261018.md` — `531a1a92dbcccbe379c11f3aaf77be37e81cd3f253f75d5b5a10071141c4c718`

Small exact supporting files (make the accepted claims auditable — each is
the core summary/driver/raw/schedule/screenshot proof, NOT the 21 MB raw set):
18. `receipts/paced-r1-driver-20261018.json` — `b26eea9ee05f961c9c778fa9bccdf374e4ad025d74e9757c70ea60957e4c42c3` (paced-r1 driver)
19. `receipts/frigate-marker-pairs-analysis-20261018.json` — `67d7bcbde2ceb4acc37e9eabdfee0f15be0bf289899f433b285ebe0af7735afb` (marker-pairs pacing raw)
20. `receipts/frigate-click-select-pacing-r1.json` — `d95c805deaef723894e099c44b527a27019d4545e30e63d2b7c515df9421b533` (pacing-r1 click-select raw)
21. `receipts/frigate-click-select-v2.json` — `11c04020663fcb79e5a0432d8f5a6482bc151e5118528d5b116787b9dbd246cf` (v2 click-select raw, contains the #FC8F marker)
22. `receipts/renderer-activation-20261018.md` — `b4eb84089fd3dc59f6dc352d37472d2d7ac63466b18d5daa18ed4005cbd036e9` (renderer activation receipt)
23. `receipts/optical-loop-20261018/proof-summary.json` — `fea3e7e08969236f4d24db5f203267d4f4eb9a5ff19e3e0486f04631008a3f07` (core loop proof summary)
24. `receipts/optical-loop-repaired-20261018/proof-summary.json` — `accfa31d34789aac7bfd6aa378149a2f1de694d22b6c68931e4d41cc1039f595` (repaired-loop proof summary)
25. `receipts/optical-loop-optimized-20261018/proof-summary.json` — `3eb156af356bd85544db0e45cb8434df7bd2dff62691afc2216aca0bf5e6ffb1` (optimized-run proof summary, 0 valid trials)
26. `receipts/optical-residual-diag-20261018/diag-summary.json` — `1c913550af6f2f6dff891d7a53383811f696fe07cf20293a59f42e7ccf0b45f6` (residual diagnostic summary)
27. `receipts/browser-identity-marker-20261018/transition-schedule.json` — `7d49ef5976379337aa5a13cea691ec9813e57d440fd4ee56b11062af4a428bb6` (browser identity transition schedule)
28. `receipts/renderer-activation-20261018/screen-play-start.png` — `7ab6acc3383b5d5df25db2bb8ec400eca37add2c98c5e12ac682b6c674b42152` (screenshot proof, play start)
29. `receipts/renderer-activation-20261018/screen-play-sustain.png` — `7ab6acc3383b5d5df25db2bb8ec400eca37add2c98c5e12ac682b6c674b42152` (screenshot proof, sustained playback; byte-identical to start frame)
30. `receipts/optical-loop-20261018/renderer-screen.png` — `f96b1f6439cd444b0ab4b197c949b3f43c27fb02e8d8c599fab3303e7205b5cb` (renderer screen proof)
31. `receipts/optical-publication-preparation-20261018.md` — this file (self-hash not embedded; owner pins it post-write)

### Excluded, preserved locally (do NOT stage; not part of the frozen set)

- The remaining ~195 raw entries under `receipts/` (~21 MB):
  full-frame/ROI JPEG-PNG surfaces (`frigate-marker-roi-*.png` ×39,
  `frigate-target-crop-*.png` ×16, `frigate-selected-*`,
  `frigate-dashboard-*`), the raw directory bodies of
  `latency-proof-20261018/`, `optical-loop-*/`, `optical-residual-diag-20261018/`,
  `frigate-h264-20261018-backup/`, `browser-identity-marker-20261018/`
  (except the schedule in #27), `renderer-activation-20261018/` (except the
  two screenshots in #28/#29), and all APKs
  (`renderer-activation-20261018/display-apk-before.apk`, camera-repo
  `build/*.apk`). These remain on disk as the evidence base the allowlisted
  files reference; the allowlist is the durable package.
- Measured claims boundary (restated so no downstream text overclaims):
  whole loop 740.7 / 965.6 ms over 20 trials; browser-relative 271.2 /
  391.5 ms over 19 pairs; the fixed-marker run bypasses the production
  detector. NO sub-100 ms claim exists or may be derived from this evidence.

### Ready recipe — cat-tv-play (owner runs; NOT executed by this lane)

```
cd /home/kom/proj/cat-tv-play
but diff                       # expect exactly the 31 allowlist paths dirty; commit ONLY those IDs
but commit -b fix/dynamic-cat-tv-source -m "docs: freeze 2026-10-18 optical latency evidence" <file-ids…>
git push origin <new-sha>:refs/heads/fix/dynamic-cat-tv-source    # append-only, no force flags
git ls-remote origin refs/heads/fix/dynamic-cat-tv-source          # readback: expect <new-sha>
```

Append-only on top of `ade13cf…`; no `but pull`, no rebase, no force, no
retarget to `main`.

## Repo 3 — /home/kom/Dropbox/Reemxy/tasks-loop (canonical sync lane)

- Branch `main`, HEAD `08a0c81f10aecf536ef6483e067a6ea46aaa93e9` =
  `origin/main` (ls-remote confirmed). Remote
  `git@github.com:Komzpa/tasks-loop.git` (`smallcat`/`upstream`/`canonical`/
  `fork` present, not used).
- Working tree heavily dirty with unrelated ops state (hundreds of
  `Intermediate/`, `Config/`, telegram export modifications). Do NOT stage any
  of it. Untracked konsole packaging files and `local/` logs are excluded.
- Bounded named allowlist — tasks-loop (single file):
  1. `ops/frigate-nucat/README.md` — `74d496ee9bab2eaffd9b4abee636534ed3cafa494708016efd8fa42504a315aa`;
     tracked-modified, +83 lines, "Frame-pacing repair (deployed 2026-10-18)":
     go2rtc `#raw=-fps_mode:v passthrough` fix, live config SHA
     `649c5e97e08e639532d1dbee4ccfd7b2f542969ffb62edd3705ce3396f6be446`,
     pre-change backup `767cb993…`, FrigateConfig VALID + 6/6 semantic PASS,
     one `docker restart frigate-nucat`, `Up (healthy)`.
- `ops/systemd/` clean; the overlay unit change was already published inside
  `f4ef4a01` (on `origin/main`). No untracked files under `ops/`.
- Ready recipe — tasks-loop (canonical helper / `but` selective; raw
  `git add`/`git commit` is NOT the route):

```
cd /home/kom/Dropbox/Reemxy/tasks-loop
# canonical tasks-loop sync helper (skill://tasks-loop-repo-sync) with exactly
# this one path staged, or `but diff` + `but commit -b main` selecting ONLY
# the ops/frigate-nucat/README.md file id; nothing else enters the commit
git push origin main          # forward push only; readback via ls-remote
```

## What remains after this preparation

1. cat-tv-play owner: selective `but commit` of the 31 allowlist paths onto
   `fix/dynamic-cat-tv-source` + append-only push + ls-remote readback.
2. tasks-loop owner: canonical-helper selective commit of the single README
   path + forward push.
3. No camera-repo action pending.

Unrelated state preserved: PID 1273442 (WeeChat) untouched; all three repos
left exactly as found apart from this receipt file.
