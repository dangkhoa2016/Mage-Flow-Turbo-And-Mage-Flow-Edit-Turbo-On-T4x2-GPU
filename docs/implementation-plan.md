# Implementation plan and current state

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](implementation-plan.vi.md)

## Phase A — clean project foundation

Status: **complete**.

- Local Git project on branch `main`.
- The repository contains only public source, tests, documentation, and release-facing assets.
- Packaging, tests, lifecycle scripts, and documentation live in one reviewable repository.
- License selected: MIT (see `LICENSE`).

## Phase B1 — T2I runtime integration

Status: **complete; CPU tests PASS; GPU acceptance PASS**.

- Strict routing: `cuda:0`, no CPU fallback.
- Localhost worker on `127.0.0.1:8101`.
- Restored validated Mage `.venv` separated from the coordinator environment.
- Local Kaggle Mage-Flow-Turbo model mount passed directly to the runtime.
- T2I transformer/VAE on `cuda:0`; accepted text-encoder placement on CPU.
- Real REST generation acceptance completed at `1024x1024`.

## Phase B2 — Edit runtime integration

Status: **complete; CPU tests PASS; GPU acceptance PASS**.

- Strict routing: `cuda:1`, no CPU fallback.
- Localhost worker on `127.0.0.1:8102`.
- Source-image transport through the authenticated coordinator.
- Frozen Edit parameters: `steps=4`, `cfg=1.0`, `prompt_template="mage-flow-edit"`, `vl_cond_long_edge=384`, blank negative prompt, maximum size `1024`.
- Edit transformer/VAE on `cuda:1`; accepted text-encoder placement on CPU.
- Real REST edit acceptance completed.

## Phase C — dual-worker orchestration and acceptance

Status: **complete**.

- `scripts/run_public_acceptance.sh` validates runtime, model mounts, source commit, token handling, and T4 x2 topology.
- Healthy resident workers are reused instead of reloaded.
- Coordinator readiness is fail-closed until both workers are ready.
- Control-plane acceptance verifies health, readiness, security, runtime info, and device routing; later notebook stages perform one real T2I inference and one real Edit inference and verify image validity, latency metadata, and output checksums.
- Accepted acceptance records resident reuse, no runtime re-extraction, and no vendor source modification.

## Phase D — bilingual Kaggle notebook

Status: **structural validation PASS; release execution evidence is external**.

The notebook uses an unnumbered bilingual introduction followed by stages `01`–`19`. Every code cell is preceded by bilingual English/Vietnamese guidance. Offline validation checks stage structure, public-safe labels, no CPU fallback, and no heavy model imports at notebook cell scope. The final Kaggle Saved Version is release evidence and is intentionally recorded outside the source tree.

## Phase E — publication

Status: **source complete; external release evidence required before tagging**.

The public source layout is complete. T4 x2 acceptance has been exercised on the supported runtime. Final execution evidence is produced as a Kaggle Saved Version against the exact revision selected for release and is linked from the GitHub Release rather than written back into source metadata.

Publication sequence:

1. Freeze publication metadata and repository history.
2. Run the public Kaggle notebook end-to-end from a clean saved-version context.
3. Verify that the Saved Version is bound to the exact frozen release revision.
4. Tag/release `v1.0.0` without changing the verified source tree.
5. Link the Saved Version and its log/output evidence from the GitHub Release.

This avoids a circular workflow where recording a successful run in source would itself change the revision that was verified.
