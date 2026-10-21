# Policy reader, step 2: denials, claims, raw-output log, demo

## What was added, in plain terms

**Denials.** Rows now have `negated: true/false`. "We do not sell your personal information" becomes
`NOT sell Personal info`. This is the most useful thing a policy can say for MAPPA: a denial plus a
tracker in the APK that does the thing is a direct contradiction. Scored against APP-350's
NOT_PERFORMED labels (36 policy/category pairs in the 100 TEST policies; too few to measure on 10).

**Claims pass.** A second prompt over the same chunks extracts `topic / value / detail / quote` for
security, retention, user rights, breach and contact, children, automated decisions, and AI.
Each topic has a fixed value list (`schema.CLAIM_VALUES`), checked in code like the row fields.
`detail` (e.g. "24 months", "16", "OpenAI") must be copied from its own quote, or the claim is dropped.
Trade-off: two passes cost about twice as much as one, but each prompt stays short, and the data-row
numbers stay comparable with older runs. APP-350 has no labels for any of this, so claims are
unmeasured until there's a hand-labelled set. `eval.py` skips the claims pass to save the daily allowance.

**Quote check, slightly relaxed.** Capital letters and punctuation at the very end are now ignored.
The first live demo run showed why: the model quoted "We do not sell your personal information to
anyone." (the real sentence carries on after a comma), and the strict check threw away the single
most important denial. Any changed, missing or "..." word still fails.

**Every raw reply is saved** (`llm.recording` / `llm.replay`). Each call is stored under a hash of the
exact prompt + excerpt. This replaces "temperature 0" as the reproducibility mechanism. `eval.py`
uses it as a resume cache: calls already answered are replayed, so a run cut off by the daily cap
carries on, and a prompt edit automatically means new calls. The checks always re-run on current code.

**Daily cap handling.** Groq's free tier allows 200k tokens a day on gpt-oss-120b (about 40 APP-350
policies). A 429 asking to wait more than 2 minutes now raises an error. Before, it was silently
counted as "this chunk says nothing", which would have quietly lowered recall.

**Prompt changes.** One-line Play definitions for the 14 categories (the model had filed "device
model" under Device IDs), and a firmer "copy the whole sentence" rule.

## Demo
`python demo.py` replays a recorded gpt-oss-120b run on `demo_policy.txt`, a fictional Australian
heart-health app policy written for this repo (so it can be committed, unlike APP-350). It plants one
made-up quote to show the check dropping it. The recorded run cost $0.002.

## Still weak
- Claims and purpose/condition fields have no ground truth. The hand-labelled AU health set is the fix.
- Same policy, same prompt, temperature 0: policy 1 gave 10 rows in one run and 6 in the next.
  Report averages over repeated runs before quoting any number.
