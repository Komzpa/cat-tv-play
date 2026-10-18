# Optical Publication — 2026-10-18

Publication receipt for the frozen 2026-10-18 optical evidence set. Executed from the
checked preparation manifest `receipts/optical-publication-preparation-20261018.md`
(rev2, sha256
`06459d8d477e290bb95fb87028fd79b4b31e437dc044ece17d9a597f435d6315`, verified before
mutation).

## Publication target and shape

- Repo `/home/kom/proj/cat-tv-play`, existing branch `fix/dynamic-cat-tv-source`,
  append-only on top of `ade13cf98a3ee74167e719cb77fef2ef6706fb9b` (local = remote at
  start, ls-remote confirmed this session; no rebase, no `but pull`, no force flags, no
  retarget to `main`, no branch rewrite).
- Commit 1 message: `docs: record measured projector latency and remaining safety gap`.
- Allowlist: the 31 named paths from the manifest — each sha256-verified against its
  pinned value before staging — plus
  `receipts/production-optical-stimulus-20261018.md` (relocated wrong-cwd gate receipt,
  corrected; original sha256
  `b5b28842e74f2bc276a2622dc119691f8893dc71ccd12646932f430b7079afce`) and this
  receipt. 33 files total.
- Raw evidence (~195 untracked entries under `receipts/`, ~21 MB: ROI/full-frame PNG
  surfaces, raw run directories, APKs) stays untracked on local disk as the evidence
  base the allowlisted summaries reference. Candidate early-buffer change
  `207210774e3a` is unproven and rolled back; no candidate commit or APK is published.

## Measured-claims boundary — QA acceptance FAILED (partial publication)

- QA `AuditLatencyAcceptance` FAILED R1/R2 against
  `receipts/deployed-latency-20261018.md` (sha256
  `858d770c25482da31ddb53cc16daf45ab5f9dfbafac06ada8887f1d2ec517f95`): the fixed-marker
  run (20 trials, p50 740.7 ms / p95 965.6 ms) bypasses the production detector; the
  <100 ms target is NOT met.
- Browser-relative measurement: 19 unique pairs, p50 271.2 ms / p95 391.5 ms.
- This publication records the measured fixes and the honest partial state. It is NOT a
  full acceptance claim. The main verification row stays OPEN, blocked on an
  independently visible physical person/stimulus (see the stimulus-gate receipt).
- No sub-100 ms claim exists or may be derived from this evidence.

## Sister repositories (verified, no action needed)

- Camera repo `/home/kom/proj/cat-projector-camera`: working tree clean,
  `origin/main` = `56c002e81e1de6025641748c65cf28c871e1dc3a` already contains the core
  source fixes (ls-remote readback this session). Nothing staged or pushed.
- tasks-loop: the single-file allowlist `ops/frigate-nucat/README.md` (sha256
  `74d496ee9bab2eaffd9b4abee636534ed3cafa494708016efd8fa42504a315aa`) is already
  identical on `origin/main` = `08d8e07c6d5810b088871ad8a7b9626ee862bcfb` (exact-path
  diff empty; auto-sync published it). No commit made; unrelated dirty ops state
  untouched. The wrong-cwd stray receipt was moved out of tasks-loop into this repo.

## Safe-OFF evidence

Unchanged and already existing on the published branches: the optical harness stops the
display with `--restore-off` recording the baseline as having no display task and never
controls the physical lamp; lamp-state changes remain the proof owner's documented Home
Assistant responsibility. All hardware was OFF during this publication. No physical
controls, builds, tests, or live ops were run by this lane.

## Remote proof

- Publication commit: `49cdd5aecdb49593d524d1b986b4607e19a4bbd4` on
  `fix/dynamic-cat-tv-source`, parent exactly `ade13cf98a3ee74167e719cb77fef2ef6706fb9b`
  (33 files: the 31 manifest paths + the relocated stimulus receipt + this receipt).
- Push: `git push origin 49cdd5aecdb49593d524d1b986b4607e19a4bbd4:refs/heads/fix/dynamic-cat-tv-source`
  — clean fast-forward `ade13cf..49cdd5a`, no force flags, exit 0.
- Forge readback: `git ls-remote origin refs/heads/fix/dynamic-cat-tv-source` ->
  `49cdd5aecdb49593d524d1b986b4607e19a4bbd4`.
- This receipt does not embed its own content hash (self-hash paradox); its sha256 is
  pinned by the publisher's result. The follow-up commit carrying this section is
  recorded in the publisher's result rather than self-referentially here.
