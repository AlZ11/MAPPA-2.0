# Groq backend (offline-verified, not yet run for real)

- New file `mappa/llm.py` now holds the only provider code: `call_groq`, `call_anthropic`, `call_llm`, `PRICES`, `cost_usd`. `policy_reader.py` and `eval.py` import from it. Nothing else changed.
- Default provider: Groq (`MAPPA_PROVIDER=anthropic` switches back). Default model `openai/gpt-oss-120b` ($0.15 in / $0.60 out per 1M tokens, per console.groq.com/docs, Oct 2026). Override with `MAPPA_GROQ_MODEL`.
- Plain HTTP via `requests`, no SDK. `temperature=0` IS set (Groq accepts it). It reduces variation; it doesn't guarantee identical runs, so raw outputs are still saved.
- `response_format: json_object` makes Groq reject non-JSON replies (a 400 counts as a failed attempt, so the existing one-retry rule applies). `reasoning_effort=low`, `include_reasoning=false`.
- 429 (rate limit) and 5xx: waits `retry-after` (or 1, 2, 4... s) and retries up to 6 times. The free tier's per-minute limits are low, so expect slow runs rather than errors.
- Checked offline: oracle on 100 TEST policies is unchanged (P=R=1.0, 295/295 planted quotes caught); a fake HTTP layer confirmed the 429 retry, the token parsing and the cost maths.
- Rough cost: one 8k-char chunk is about 2.6k tokens in and a few hundred out, about $0.0006. 10 policies should cost cents.
- Not done (deliberately, handed to the eval-run thread): the real run, the later forms, the negative-claim fix. Ideas for the negation fix: a `negated: bool` on Row (it covers "don't collect / share / SELL", not just collect), scored against APP-350's NOT_PERFORMED labels, which makes it measurable.
