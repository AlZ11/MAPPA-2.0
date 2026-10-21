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
python -m mappa labels --run s1a                  # Data Safety pages -> evidence rows
python -m mappa show --run s1a --app com.calm.android

python -m mappa train                             # baseline policy reader (downloads APP-350, OPP-115)
python -m mappa read --run s1a                    # policies -> evidence rows (--extractor llm for the local model)
python -m mappa evaluate                          # precision/recall/F1 on APP-350, OPP-115, unibo test sets
python -m mappa annotate --run s1a --apps com.calm.android,... --out sheet.csv   # hand-labelling sheet + codebook
python -m mappa kappa sheet_a.csv sheet_b.csv     # agreement between two annotators
python -m mappa evaluate --corpus indomain --gold adjudicated.csv
python -m pytest -q
```

Later waves re-fetch the same apps: `snapshot --run s1b --from s1a`.
All data goes to `./data` (git-ignored). Set `MAPPA_CHROMIUM` to a Chromium binary if Playwright's own
browser download is not available.
