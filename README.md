# MAPPA 2.0

Checks whether an Android app's privacy claims hold up: what its privacy policy says, versus its
Google Play Data Safety label, its APK and its network traffic. Findings are scored against ACSIMA 2.0.

This repo currently holds step 1, the **policy reader**: an LLM turns a privacy policy into fixed-form
rows (data type, action, recipient, purpose, condition, verbatim quote), and code checks every row
(allowed values only, quote must appear word for word in the policy).

## Run

```bash
cd mappa
pip install -r requirements.txt
export GROQ_API_KEY=...          # Groq, openai/gpt-oss-120b at temperature 0
export MAPPA_APP350=/path/to/APP-350_v1.1

python eval.py oracle 100        # no API calls: checks chunking, quote check and scoring
python eval.py 10                # real run on the first 10 APP-350 TEST policies
python policy_reader.py 1        # one policy end to end
```

`eval.py` caches each policy's result, so a long run resumes where the daily rate limit stopped it.

APP-350 is research-only licensed and is not included. Get it from its authors and point
`MAPPA_APP350` at it. Run outputs in `mappa/results/` quote policy text and are git-ignored.

Design notes: [`notes/`](notes/).
