---
name: pynq-fitguard-release-audit
description: Audit, validate, and package a PYNQ FPGA project for reproducible release without modifying the live board. Use when preparing a PYNQ engineering submission, checking HLS/Vivado/PS artifacts, proving that a PL backend actually ran, separating private datasets from public evidence, or building a competition-ready release checklist.
---

# PYNQ FitGuard Release Audit

Use a read-only snapshot of the board as evidence. Never edit the live board during a release audit unless the user separately authorizes deployment.

## Workflow

1. Read `references/workflow.md` and map the requested submission rules to concrete artifacts.
2. Inventory HLS/RTL, Vivado sources, bit/hwh/xsa, PS code, firmware, models, verification, reports and documentation.
3. Compare hashes of board bit/hwh and local golden deliverables.
4. Prove backend activity with both a deterministic board self-test and a real runtime record. Do not infer PL use from filenames alone.
5. Separate public material from raw video, identifiable sensor trials, event captures, credentials, backups and caches.
6. Run `scripts/verify_artifacts.py <release-root>` and resolve every error.
7. Use `references/evidence-schema.md` to write the performance table. Keep accelerator latency separate from end-to-end latency.
8. Finish with the checklist in `references/troubleshooting.md`, then create a ZIP and SHA256 only after validation passes.

## Required claims discipline

- State exactly which operators run in PL and which remain in PS.
- Treat a fallback run as software robustness evidence, not FPGA acceleration evidence.
- Treat class agreement on the project dataset as engineering validation, not clinical accuracy.
- Record tool version, part, clock, timing, resources, numerical error and board latency.
- If the training dataset is private, explain the limitation and provide anonymous golden samples plus collection code.

## Failure conditions

Stop the release and report the problem if any of these occurs:

- bit and hwh do not match or hashes differ from the tested board artifacts;
- board self-test fails or exceeds its numerical threshold;
- runtime evidence shows a NumPy fallback instead of the claimed PL backend;
- build paths require an unavailable private project without disclosure;
- the package includes credentials, raw identifiable media, cache files or incomplete temporary downloads;
- a performance claim has no corresponding report or JSON evidence.
