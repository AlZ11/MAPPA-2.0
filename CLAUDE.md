# MAPPA 2.0: working rules

## Build philosophy (the owner's standing rule)
Working concept first: the actual logic, end to end, before anything production-grade.
- No CI, Docker, lint/format configs, LICENSE, CONTRIBUTING or extra docs until asked.
- Flat package, one module per pipeline part. Neat, readable, professional.
- Tests only for deterministic logic (parsers, matching, metrics). No tests that hit the network.

## Non-negotiables (these make the output research evidence, not a demo)
- Every fetched item is frozen: SHA-256 of the bytes, UTC fetch time, final URL after redirects, HTTP status.
- Crawls are resumable and safe to re-run.
- Every policy-derived row carries a verbatim quote that code finds in the policy text; rows whose quote
  is not found are dropped **and counted**.
- Unknown labels/values are kept and logged, never silently dropped.
- A parser failure or a missing page never becomes a "no data collected/shared" claim.
- The extractor that gets evaluated is the one that gets deployed. Prompt work happens on dev splits only;
  freeze `PROMPT_VERSION` before touching a test split.
- Never commit third-party datasets or anything derived from them (APP-350, OPP-115, unibo): `data/` is git-ignored.

## Layout
`mappa/`: vocab (shared terms + Claim row), store (SQLite + gzip blobs), play (Google Play client), collect (Part 1),
label_reader (Part 3), text (html → text, segments, quote matching), policy_reader (Part 2), llm (Gemini client),
datasets (gold corpora), evaluate (metrics). Run `python -m pytest -q` from the repo root.

## Explaining things to the owner
Plain English, no jargon walls: explain the mechanics of what the code does and why.
