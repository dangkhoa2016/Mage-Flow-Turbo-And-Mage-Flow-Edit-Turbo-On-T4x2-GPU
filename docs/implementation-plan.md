# Implementation plan and current state

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](implementation-plan.vi.md)

## Phase A — clean project foundation

Status: **complete**.

- Local Git project on branch `main`.
- Public source remains separate from internal qualification/evidence artifacts.
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
- Live acceptance verifies health, readiness, runtime info, T2I generation, Edit generation, image validity, latency, and output checksums.
- Accepted acceptance records resident reuse, no runtime re-extraction, and no vendor source modification.

## Phase D — bilingual Kaggle notebook

Status: **structural validation PASS; public live Run All pending**.

The notebook uses stages `00`–`19`. Every code cell is preceded by bilingual English/Vietnamese guidance. Offline validation checks stage structure, public-safe labels, no CPU fallback, and no heavy model imports at notebook cell scope.

## Phase E — publication

Status: **pending**.

Source-level canonicalization closeout is complete. Historical/local T4 x2 acceptance evidence remains valid for the frozen public acceptance surface, but the formal GPU qualification gate in the current canonical chain has not been executed.

Remaining publication-only steps:

1. Freeze publication metadata and repository description.
2. Create the official GitHub repository and push the clean rewritten history.
3. Run the public Kaggle notebook end-to-end from a clean saved-version context.
4. Verify the saved public notebook version.
5. Tag/release `v1.0.0` only after publication checks pass.

These metadata/publication corrections do not authorize a GPU qualification rerun. Any formal GPU qualification must be entered only through its separately authorized gate after the current P14/P15 chain.
