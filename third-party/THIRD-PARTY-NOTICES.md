# THIRD-PARTY NOTICES — MAGE-FLOW T4x2 Public Canonicalization

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](THIRD-PARTY-NOTICES.vi.md)

This tree redistributes or references third-party components. Each is listed with its
upstream origin, applicable license authority, and the exact redistribution condition.

## 1. llama.cpp (runtime llama-server binary)
- Upstream: ggerganov/llama.cpp (MIT — https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)
- Runtime-surface: `runtime/bin/llama-server` in the support Dataset (byte-frozen copy of the
  qualified DFlash2 llama.cpp runtime build; SHAs in manifest/SHA256SUMS).
- Redistribution condition: this is a byte-copy of a build distributed via Kaggle public Dataset
  `mage-flow-glimmer-30b-dflash2-llama-runtime`; the served bytes are unchanged. MIT license text
  is provided verbatim in `third-party/licenses/MIT-llama.cpp.md`. We do not claim authorship of
  llama.cpp; the binary provenance and build hash are documented in runtime-provenance.txt.

## 2. stable-diffusion.cpp (CPU-safe gguf runtimes; not in consolidated Dataset)
- Upstream: leejet/stable-diffusion.cpp (MIT — https://github.com/leejet/stable-diffusion.cpp)
- Not folded into the consolidated runtime Dataset; retained on separate Kaggle support resources.

## 3. Mage-Flow Community model weights (open redistribution)
- Upstream model owners (Mage-Flow community Turbo / Edit Turbo) license their weights for
  community redistribution via Kaggle Models. The canonical Dataset does not bundle the weights;
  it references the public Kaggle Models surfaces.

## 4. Python/Qwen/opensource dependencies used by the source tree
- Only materially redistributed binaries are cited above. Source-tree dependencies are installed
  from public wheel mirrors at runtime and listed in requirements; no vendored third-party
  executable is bundled with the canonical source archive.

Contact: see LICENSE (project) — MIT, Mage-Flow authors.
