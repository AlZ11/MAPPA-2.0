# Policy reader, step 1: what's built, what's measured, what's not

Code: `/mnt/project-files/mappa/` (602 lines total). Run from inside that folder.

| File | What it does |
|---|---|
| `schema.py` | The row form (Pydantic). Every field except `recipient` and `quote` is a fixed list. |
| `mappings.py` | APP-350 label → Play category table. Unmappable labels → `None` (left out of scoring). |
| `app350.py` | Reads one APP-350 policy in place: plain text + expert labels. Never copies the data. |
| `policy_reader.py` | chunk → LLM → validate → quote check → recipient check → dedupe → tokens/cost. |
| `eval.py` | Per-category P/R/F1, fabricated-quote rate, cost per policy, on APP-350 TEST policies. |

Commands:
- `python policy_reader.py 1`: one APP-350 policy end to end, prints rows + stats.
- `python eval.py 10`: real run on the first 10 TEST policies.
- `python eval.py oracle 100`: no API calls. Checks the plumbing (see below).

## How each step works, in plain terms

**Chunking.** The policy is split at blank lines (paragraphs), then whole paragraphs are packed into chunks of up to 8,000 characters (about 2,000 tokens). A single paragraph longer than that gets split at sentence ends. Trade-off: bigger chunks mean fewer calls, and the instructions (about 600 tokens) are re-sent with each one, so bigger is cheaper. But the model skims more text per call and misses things. 8,000 is a starting guess; tune it with eval numbers, not intuition.

**One LLM function.** `call_llm(system, user)` is the only code that knows about a provider. It returns text plus token counts. Swapping model = edit that one function (or pass a different function into `read_policy`). The model is asked for plain JSON, not a provider-specific "structured output" feature. Trade-off: the model can occasionally return broken JSON (handled: one retry, then the chunk is counted as failed). The upside is that the validation is *your* code, so it works identically for any provider and you can explain every line of it.

**Validation.** Each row goes through the Pydantic model. Any field outside its allowed list → the row is dropped and counted, with the reason saved. Nothing is coerced. "Emails" doesn't become "Personal info"; it's dropped.

**Quote check.** Both the quote and the policy are normalised (curly quotes → straight, dashes unified, all whitespace runs → one space). Then it's a plain substring search. Not found → dropped, counted as fabricated. It's deliberately strict: a quote with "..." or a trimmed clause counts as fabricated. If that rate comes back high on a real run, read the dropped rows before loosening anything. Most will be the model tidying up sentences, which is a prompt fix, not a check fix.

**Recipient check (added).** If the model names a company ("Google"), that name must appear somewhere in the policy. Same idea as the quote check, applied to the one free-text field that matters downstream.

**Dedupe.** Rows with the same (data_type, action, recipient, purpose, condition) are merged; the first quote is kept. Trade-off: you lose the second and third supporting quotes. That's fine for scoring. For the evidence store later you'll probably want all of them, so change the merge to collect a list of quotes then.

**Cost.** Every call's input/output tokens are summed per policy and priced from a table in `policy_reader.py`. Thinking tokens are billed as output tokens and are included in the count.

## What was measured (oracle run, no model)

The oracle is a fake LLM that answers with the expert labels, plus one deliberately made-up quote per chunk. It tests the plumbing, not a model.

On all 100 TEST policies: 295 chunks, P = R = F1 = 1.00 on every category, and exactly 295 fabricated quotes caught (one per chunk). So chunking doesn't lose labelled sentences, the quote check catches fabrication, and the scoring math is right. Biggest policy (#242, 200k chars) → 28 chunks, none over 8k.

Failure paths checked by hand: non-JSON reply → retried once, then counted as a failed chunk; off-list data type → dropped; quote missing punctuation that's in the policy → counted as fabricated (strict on purpose).

## Where the spec is weak (pushback)

1. **"Temperature 0" isn't possible on current Claude models.** They reject sampling parameters outright (the API returns an error). Even where temperature 0 exists, it doesn't make outputs identical across runs. Reproducibility has to come from saving every raw output to disk (eval already does: `results/*.json`), not from a sampling knob. Rewrite the rule as "every run's output is saved and re-scorable".
2. **APP-350 can only test 4 of your 14 categories**: Location, Personal info, Contacts, Device or other IDs. It says nothing about Health and fitness, which is the category that matters most for mHealth apps. Precision/recall on APP-350 checks the extraction mechanics. It does not validate the tool for your actual use case. The hand-labelled Australian health-app set isn't "later"; it's the real eval. Start labelling 15–20 policies now, while the model runs on APP-350.
3. **APP-350 doesn't label purpose, condition, sell, or company names.** Those fields go completely unmeasured. Same fix: your own labelled set.
4. **APP-350's "3rdParty" ≠ your "share".** For them, it means a third party collects the data (often via an SDK). For you, it means the developer passes data on. The mapping treats any non-first-party row as 3rdParty, which is the closest honest match. Expect the category+party table to score lower than category-only, partly because of this mismatch and not because of model error.
5. **"Not performed" statements are thrown away.** APP-350 has many ("we do not collect your phone number": 184 labels for phone alone). For a contradiction engine, "we don't collect X" plus a tracker that collects X is the most valuable finding you can produce. Your row form has no way to say it. Add `action: not_collect` (or a `negated` flag) before you build the consistency engine.
6. **I added `purpose: unspecified`.** Most policy sentences give no purpose. Forcing those into "other" would make "other" mean two different things.

## What's missing

- **No real model run yet**: there's no API key in this environment. Numbers come after that.
- Cost estimate (not measured): the 100 TEST policies are roughly 400k tokens of text. Under $0.10 total on Haiku 5.5, roughly $2–4 on Opus 5.5 including output. Cost isn't your bottleneck on APP-350; label quality is.
- `results/` files contain APP-350 text. Keep them local; don't commit them.
