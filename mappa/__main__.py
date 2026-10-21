"""Command line: python -m mappa <command> --help"""

import argparse

from pathlib import Path

from . import collect, evaluate, label_reader, policy_reader
from .store import Store


def main() -> None:
    p = argparse.ArgumentParser(prog="mappa", description="MAPPA 2.0: what health apps say vs what they do.")
    p.add_argument("--data", default="data", help="data folder (default: ./data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("discover", help="search Google Play (AU) for health apps")
    s.add_argument("--run", required=True)

    s = sub.add_parser("snapshot", help="freeze details, Data Safety label and policy for each app")
    s.add_argument("--run", required=True)
    s.add_argument("--from", dest="source", required=True, help="a discover run, or an earlier snapshot (panel)")
    s.add_argument("--limit", type=int)
    s.add_argument("--apps", help="comma-separated app ids: snapshot only these")
    s.add_argument("--no-render", action="store_true", help="never use the headless browser")

    s = sub.add_parser("status", help="what a run holds")
    s.add_argument("--run", required=True)
    s.add_argument("--manifest", help="also write a CSV of every fetch and print its SHA-256")

    s = sub.add_parser("labels", help="parse every frozen Data Safety page of a run into evidence rows")
    s.add_argument("--run", required=True)

    s = sub.add_parser("show", help="print one app's evidence rows, label and policy side by side")
    s.add_argument("--run", required=True)
    s.add_argument("--app", required=True)

    sub.add_parser("train", help="train the baseline policy reader on APP-350 + OPP-115 (train splits)")

    s = sub.add_parser("read", help="extract evidence rows from every frozen policy of a run")
    s.add_argument("--run", required=True)
    add_extractor_args(s)

    s = sub.add_parser("evaluate", help="score an extractor against gold corpora")
    s.add_argument("--corpus", default="app350,opp115,unibo", help="any of app350, opp115, unibo, indomain")
    s.add_argument("--split", default="test", choices=["test", "dev"], help="dev for prompt work; test for reporting")
    s.add_argument("--gold", type=Path, help="adjudicated annotation sheet (for --corpus indomain)")
    s.add_argument("--reps", type=int, default=1, help="repeat the LLM runs to measure stability")
    s.add_argument("--subset", type=int, help="only the first N documents of each corpus")
    add_extractor_args(s)

    s = sub.add_parser("annotate", help="write a blank annotation sheet for hand-labelling policies")
    s.add_argument("--run", required=True)
    s.add_argument("--apps", required=True, help="comma-separated app ids")
    s.add_argument("--out", required=True, type=Path)

    s = sub.add_parser("kappa", help="agreement between two filled-in annotation sheets")
    s.add_argument("sheet_a", type=Path)
    s.add_argument("sheet_b", type=Path)

    a = p.parse_args()
    store = Store(a.data)
    if a.cmd == "discover":
        n = collect.discover(store, a.run)
        print(f"{n} distinct apps found")
    elif a.cmd == "snapshot":
        only = a.apps.split(",") if a.apps else None
        collect.snapshot(store, a.run, a.source, limit=a.limit, render=not a.no_render, only=only)
    elif a.cmd == "status":
        collect.status(store, a.run, a.manifest)
    elif a.cmd == "labels":
        label_reader.read_labels(store, a.run)
    elif a.cmd == "show":
        show(store, a.run, a.app)
    elif a.cmd == "train":
        from .datasets import load
        baseline = policy_reader.Baseline.train({c: load(c, store.root, "train") for c in ("app350", "opp115")})
        baseline.save(store.root / "models" / "baseline.pkl")
        print(f"trained {len(baseline.info['labels'])} labels; too few examples for {baseline.info['skipped']}")
    elif a.cmd == "read":
        policy_reader.read_policies(store, a.run, make_extractor(a, store))
    elif a.cmd == "evaluate":
        report = evaluate.run_eval(make_extractor(a, store), a.corpus.split(","), store.root, split=a.split,
                                   reps=a.reps, subset=a.subset, gold_path=a.gold)
        evaluate.print_report(report)
        print(f"\nsaved {evaluate.save_report(report, store.root)}")
    elif a.cmd == "annotate":
        evaluate.annotate(store, a.run, a.apps.split(","), a.out)
    elif a.cmd == "kappa":
        evaluate.kappa_report(a.sheet_a, a.sheet_b)


def add_extractor_args(s) -> None:
    s.add_argument("--extractor", default="baseline", choices=["baseline", "llm"])
    s.add_argument("--model", help="LLM model name (default: $MAPPA_LLM_MODEL or qwen2.5:7b-instruct)")
    s.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $MAPPA_LLM_URL or Ollama on localhost)")


def make_extractor(a, store: Store):
    if a.extractor == "baseline":
        return policy_reader.Baseline.load(store.root / "models" / "baseline.pkl")
    return policy_reader.LLM(model=a.model, base_url=a.base_url, cache_dir=store.root / "cache" / "llm")


def show(store: Store, run: str, app_id: str) -> None:
    app = store.db.execute("SELECT * FROM apps WHERE run = ? AND app_id = ?", (run, app_id)).fetchone()
    if app is None:
        raise SystemExit(f"{app_id} is not in run {run}")
    print(f"{app['title']} ({app_id}) by {app['developer']}, {app['genre']}, updated {app['updated']}")
    print(f"policy {app['policy_url']} [{app['policy_flags'] or 'ok'}], dated {app['policy_date']}; label {app['label_status']}")
    for row in store.claims(run, app_id):
        what = row["practice"] or f"{row['category']} > {row['subtype'] or '*'}"
        verb = "practice" if row["kind"] == "practice" else row["action"]
        neg = "NOT " if row["polarity"] == "deny" else ""
        src = f"{row['source']}/{row['extractor']}{'/' + row['model'] if row['model'] else ''}"
        print(f"  {src:28} {neg + verb:12} {what:45} | {(row['quote'] or '')[:90]}")


if __name__ == "__main__":
    main()
