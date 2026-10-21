"""Command line: python -m mappa <command> --help"""

import argparse

from . import collect, label_reader
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
        print(f"  {src:24} {neg}{verb:8} {what:45} | {(row['quote'] or '')[:90]}")


if __name__ == "__main__":
    main()
