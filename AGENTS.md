# AGENTS.md

## Scope
- Documentation surface profile: public-python.

## Source Of Truth Docs
- Follow `README.md` for the repo's public summary and starting docs.
- Follow `docs/architecture.md` for package shape, artifact boundaries, and stable design decisions.
- Follow `docs/api.md` for the supported public Python API and CLI-facing helper contracts.
- Follow `docs/algorithms.md` for traversal, dense-field NGS selection, winner
  selection, and AO metric aggregation algorithms.
- Follow `docs/development.md` for local bootstrap, environment, and daily workflow.
- Follow `docs/benchmarking.md` for benchmark evidence that informs storage, scheduling, and cache decisions.
- Follow `docs/testing.md` for verification expectations.

## Research Logs
- `docs/benchmarking.md` is the benchmarking research log. Its companion folder is `docs/benchmarking/`. Common aliases: `benchmarking`, `benchmarks`, `benchmarking.md`.
- `docs/validation.md` is the validation research log. Its companion folder is `docs/validation/`. Common aliases: `validation`, `validation.md`.
- Use `$research-logging` for capture, entry updates, summary checks, concept maintenance, or source-document conversion in these logs.

## Shared Validation
- Use `$agent-surface-review` for shared agent-surface review.
- Use `$documentation-surface-review` for documentation-surface review with the `public-python` profile.
- Use `$code-quality-review` for source-code quality review.

## Skill Requirements
- For Python code, use `$python-code-writing`.
- For project docs such as `docs/architecture.md`, `docs/testing.md`, `docs/development.md`, and similar long-lived project documents, use `$project-docs-writing`.
- For `README.md`, use `$readme-writing`.
- For plan documents or execution-roadmap docs when they are created or revised, use `$plan-writing`.

## Working Rules
- For package structure, public API boundaries, persisted contracts, and lifecycle-sensitive changes, consult `docs/architecture.md` before editing.
- For local Python bootstrap, environment choice, and daily workflow, consult `docs/development.md`.
- Before concluding substantial work, satisfy the verification expectations in `docs/testing.md`.
- For storage, scheduling, traversal, or cache decisions that depend on measured evidence, consult `docs/benchmarking.md`.
