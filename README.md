# MAPPA 2.0

Checks whether an Android app's privacy claims hold up: what its privacy policy says, versus its
Google Play Data Safety label, its APK and its network traffic. Findings are scored against ACSIMA 2.0.

This repo currently holds the **policy reader**: an LLM turns a privacy policy into
- fixed-form **rows**: data type, action, recipient, purpose, condition, and whether it's a denial
  ("we do NOT collect X"), each backed by a verbatim quote;
- **claims**: security, retention, user rights, breach handling and contact, children, automated
  decisions, and AI providers.

Code checks everything the model returns: allowed values only, quotes must appear word for word in the
policy, named companies and details (periods, ages, emails) must appear in the text.

## Demo (no API key needed)

```bash
cd mappa
pip install -r requirements.txt
python demo.py            # replays a recorded run on a fictional health-app policy
python demo.py live       # same, calling Groq for real (needs GROQ_API_KEY)
```

## Run

```bash
export GROQ_API_KEY=...          # Groq, openai/gpt-oss-120b at temperature 0
export MAPPA_APP350=/path/to/APP-350_v1.1

python eval.py oracle 100        # no API calls: checks chunking, quote check and scoring
python eval.py 10                # real run on the first 10 APP-350 TEST policies
python policy_reader.py 1        # one policy end to end
```

`eval.py` saves every raw model reply and replays it on reruns, so a long run resumes where the
daily rate limit stopped it, and old runs can be re-scored for free.

APP-350 is research-only licensed and is not included. Get it from its authors and point
`MAPPA_APP350` at it. Run outputs in `mappa/results/` quote policy text and are git-ignored.

Design notes: [`notes/`](notes/).
