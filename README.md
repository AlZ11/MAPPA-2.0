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
python -m mappa demo                              # one APP-350 policy, explained: expert labels vs the reader
python -m mappa read --run s1a                    # policies -> evidence rows (--extractor llm for Gemini)
python -m mappa evaluate                          # precision/recall/F1 on APP-350, OPP-115, unibo test sets
python -m mappa annotate --run s1a --apps com.calm.android,... --out sheet.csv   # hand-labelling sheet + codebook
python -m mappa kappa sheet_a.csv sheet_b.csv     # agreement between two annotators
python -m mappa evaluate --corpus indomain --gold adjudicated.csv
python -m pytest -q
```

**LLM reader (Gemini).** Create a key at https://aistudio.google.com/apikey and `export GEMINI_API_KEY=...`, then:

```bash
python -m mappa demo --extractor llm                                              # same policy, read by Gemini
python -m mappa evaluate --extractor llm --corpus app350 --split dev --subset 10   # prompt work: dev split only
python -m mappa evaluate --extractor llm --reps 3 --subset 30                      # test, after freezing PROMPT_VERSION
python -m mappa read --run s1a --extractor llm --model gemini-3.1-pro-preview
```

The default model is `gemini-3.8-flash` (`--model` or `MAPPA_LLM_MODEL` to change). Every reply is cached under
`data/cache/llm`, so interrupted runs resume, nothing is paid for twice, and results can be re-scored. A used-up
daily quota stops the run cleanly; re-run later and it carries on. `--base-url` points the reader at any other
OpenAI-compatible server instead.

Later waves re-fetch the same apps: `snapshot --run s1b --from s1a`.
All data goes to `./data` (git-ignored). Set `MAPPA_CHROMIUM` to a Chromium binary if Playwright's own
browser download is not available.
