# MAPPA 2.0

Checks what Australian health apps **say** about your data (privacy policy, Google Play Data Safety label)
against what they **do** (code, network traffic), and scores them against ACSIMA. Extends the CSIRO Data61
ACSIMA / MAPPA study.

Build 1 (this code) is the claims side: collect and freeze the policy and the label, and turn both into the
same evidence rows.

```bash
pip install -r requirements.txt
python -m mappa discover --run d1                 # search Play (AU) for health apps
python -m mappa snapshot --run s1a --from d1      # freeze details, label and policy per app (resumable)
python -m mappa status --run s1a --manifest s1a.csv
python -m pytest -q
```

Later waves re-fetch the same apps: `snapshot --run s1b --from s1a`.
All data goes to `./data` (git-ignored). Set `MAPPA_CHROMIUM` to a Chromium binary if Playwright's own
browser download is not available.
