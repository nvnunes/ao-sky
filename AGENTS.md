# AGENTS.md

## Astro-Agents Bootstrap
- Use `astro-agents` for reusable authoring, review, and routing guidance in this repo.

## Scope
- Documentation surface profile: public-python.

## Source Of Truth Docs
- Follow `README.md` for the repo's public summary and starting docs.
- Follow `docs/architecture.md` for package shape, artifact boundaries, and stable design decisions.
- Follow `docs/api.md` for the supported public Python API and CLI-facing helper contracts.
- Follow `docs/development.md` for local bootstrap, environment, and daily workflow.
- Follow `docs/plan.md` for transitional migration context and implementation sequencing.
- Follow `docs/benchmarking.md` for benchmark evidence that informs storage, scheduling, and cache decisions.
- Follow `docs/testing.md` for verification expectations.

## Shared Guidance
- Use `astro-agents/guidance/agent-surface.md` for shared agent-surface guidance.
- Use `astro-agents/guidance/public-python-projects.md` for shared public Python repo guidance.
- Use `astro-agents/guidance/python-development.md` for shared Python development guidance.

## Authoring Requirements
- For Python code, follow `astro-agents/authoring/code/python.md`.
- For repo docs such as `docs/architecture.md`, `docs/testing.md`, `docs/development.md`, and similar long-lived repo documents, follow `astro-agents/authoring/writing/repo-docs.md`.
- For `README.md`, follow `astro-agents/authoring/writing/readme-md.md` in addition to `astro-agents/authoring/writing/repo-docs.md`.
- For plan documents or execution-roadmap docs when they are created or revised, follow `astro-agents/authoring/writing/plan.md`.

## Working Rules
- For package structure, public API boundaries, persisted contracts, and lifecycle-sensitive changes, consult `docs/architecture.md` before editing.
- For local Python bootstrap, environment choice, and daily workflow, consult `docs/development.md`.
- Before concluding substantial work, satisfy the verification expectations in `docs/testing.md`.
- For transitional migration boundaries and intended-versus-implemented structure, consult `docs/plan.md` before making structural changes.
- For storage, scheduling, traversal, or cache decisions that depend on measured evidence, consult `docs/benchmarking.md`.
