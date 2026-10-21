# Policy reader: what you can demo (21 Oct 2026)

Run everything from `mappa/`.

## The demo: one command, no API key

```
python demo.py
```

Replays a recorded gpt-oss-120b run on `demo_policy.txt`, a fictional Australian heart-health app
policy written for this repo. It needs no key and no network, and gives the same output every time. It shows:

1. **Data practices** with the exact supporting sentence, e.g. `share Health and fitness -> OpenAI`.
2. **Denials** ("we do NOT collect X"): `NOT collect Location` (precise GPS), `NOT sell Personal info`,
   `NOT share Health and fitness` with advertisers. These are what the consistency engine will later
   check against the Play label, the APK and the traffic.
3. **Claims**: encryption in transit/at rest, retention "24 months after your last login" and
   "30 days", access/correction/deletion rights, OAIC complaint, breach notification, privacy
   contact, minimum age 16, an automated risk score with clinician review (APP 1.7-1.9 territory),
   AI summaries, OpenAI named as AI provider, "not used to train".
4. **The quote check catching a fabrication**: one made-up row ("We sell your heart rate history to
   insurance companies.") is planted by the demo and dropped as `fabricated_quote`.
5. **Cost**: 2.6k tokens in, 2.8k out, $0.002 for the whole policy (both passes).

`python demo.py live` does the same with a real Groq call and re-records it.

## Other commands

```
python eval.py oracle 100       # no key: proves chunking, quote check, denial scoring are correct
python eval.py 10               # real scores on 10 APP-350 TEST policies (needs key + daily allowance)
python policy_reader.py 42      # live read of APP-350 policy 42; or pass a .txt file path
```

## Real-model numbers (old prompt, 10 APP-350 policies)

Category F1 0.85, category + who-gets-it F1 0.80, made-up quotes 3/89 (3.4%, all from shortening a
quote with "...", none invented). $0.0013 per policy. The current prompt (category definitions,
denials) has not been scored on APP-350 yet: Groq's free tier daily cap (200k tokens) ran out.
`eval.py` resumes from saved replies, so rerunning `python eval.py 10` once tokens free up continues.

## Not proven yet

- Claims have no ground truth anywhere; APP-350 doesn't label them. One fictional policy is a demo,
  not a measurement.
- Health and fitness, the category that matters for this study, has no evaluation. That needs the
  hand-labelled AU health set, not code.
- 10 policies is a smoke test; at temperature 0 the same policy still varied between runs.
