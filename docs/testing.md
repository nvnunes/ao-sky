# Testing

This document is the source of truth for verification commands and completion expectations in `ao-sky`.

`ao-sky` is still in its bootstrap stage. The repo does not yet have a Python
package, test suite, or docs build command. This document therefore records:

- the current verification expectations for the repo as it exists now
- the command categories that Phase 0 must establish explicitly
- the completion expectations that should remain stable as the repo grows

## Shared Validation

Use the shared base testing guidance in `astro-agents/validation/base-testing.md`.

## Current Repo-Local Verification

At the current stage, repo-local verification is documentation-surface
verification.

For docs-only changes in the current repo state:

- verify that changed docs remain internally consistent
- verify that source-of-truth ownership remains clear across `README.md`,
  `AGENTS.md`, `docs/architecture.md`, `docs/development.md`, `docs/plan.md`,
  `docs/benchmarking.md`, and `docs/testing.md`
- verify that cross-links and doc discovery paths remain current
- verify that commands are not claimed to exist if they do not yet exist
- verify that benchmark claims distinguish historical baselines from current
  operating assumptions when that distinction matters

## Completion Expectations By Change Type

### Docs-Only Changes

Docs-only work is complete when:

- the changed documents are internally consistent
- any affected source-of-truth cross-links are updated
- no stale command, package, or lifecycle claims remain

### Planning Or Architecture Changes

Planning or architecture work is complete when:

- the changed design or planning decision is reflected in the owning document
- adjacent documents are updated when doc discovery or ownership changes
- any benchmark-sensitive design claim is consistent with `docs/benchmarking.md`

### Development Bootstrap Changes

Environment or toolchain bootstrap work is complete when:

- `docs/development.md` is updated with the exact canonical commands
- `docs/testing.md` is updated with the exact canonical verification commands
- the commands recorded here are the ones actually used for the work

### Code Changes

Once code lands in the repo, code changes are not complete until:

- the relevant canonical verification commands from this document have been run
- docs are updated when supported usage, public imports, or workflow changes
  changed

## Canonical Verification Commands To Add In Phase 0

Phase 0 should update this document with the exact commands for:

- package installation or packaging smoke verification
- the canonical test command
- the canonical docs build command, if docs build tooling exists
- any stricter release or full-repo verification path

## Verification Scope Boundaries

- Keep environment creation, activation, and daily workflow commands in
  `docs/development.md`.
- Keep verification commands and completion expectations here.
- Keep benchmark methodology and benchmark results in `docs/benchmarking.md`.
