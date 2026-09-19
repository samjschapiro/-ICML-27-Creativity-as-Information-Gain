# CLAUDE.md

Read and follow `docs/repo_usage.md` — it is the single source of truth for all development conventions, project structure, and workflow guidelines.

## Standing rule: no --overwrite
Claude is never allowed to use an `--overwrite` flag in this repo (or anywhere else). If an
`output_dir` already exists, choose a new, descriptively named `output_dir` in the config or ask
the user to run the command. See `docs/memos/2026-09-18_overwrite_hazard.md`.
