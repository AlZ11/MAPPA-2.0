"""Command line: python -m mappa <command> --help"""

import argparse

from . import collect
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


if __name__ == "__main__":
    main()
