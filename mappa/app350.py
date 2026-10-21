"""Load APP-350 policies: plain text for the reader, expert labels for scoring.

APP-350 is research-only licensed. It is read in place and never copied into this repo.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mappings import APP350_TO_PLAY, split_practice

APP350_DIR = Path(os.environ.get("MAPPA_APP350", "/mnt/project-files/APP-350_v1.1"))


@dataclass
class Policy:
    policy_id: int
    name: str
    split: str  # TRAINING / VALIDATION / TEST
    text: str
    # Play categories the experts marked as PERFORMED anywhere in the policy.
    gold_types: set[str] = field(default_factory=set)
    # Same, but keeping who does it: {("Location", "1stParty"), ...}
    gold_pairs: set[tuple[str, str]] = field(default_factory=set)
    # (sentence, play_category, party) for every PERFORMED label; used by the oracle check.
    gold_sentences: list[tuple[str, str, str]] = field(default_factory=list)
    # Categories the experts marked NOT_PERFORMED ("we do not collect X"), and their sentences.
    gold_neg_types: set[str] = field(default_factory=set)
    gold_neg_sentences: list[tuple[str, str]] = field(default_factory=list)


def load_policy(policy_id: int) -> Policy:
    raw = yaml.safe_load((APP350_DIR / "annotations" / f"policy_{policy_id}.yml").read_text())
    # Segments are roughly paragraphs; keep them as paragraphs so the chunker can split on them.
    text = "\n\n".join(seg["segment_text"] for seg in raw["segments"])
    policy = Policy(raw["policy_id"], raw["policy_name"], raw["policy_type"], text)

    for seg in raw["segments"]:
        for sent in seg["sentences"]:
            for ann in sent["annotations"]:
                label, party = split_practice(ann["practice"])
                play = APP350_TO_PLAY[label]
                if play is None:
                    continue
                # NOT_PERFORMED ("we do not collect X") is scored separately, never as a positive.
                if ann["modality"] == "NOT_PERFORMED":
                    policy.gold_neg_types.add(play)
                    policy.gold_neg_sentences.append((sent["sentence_text"], play))
                    continue
                policy.gold_types.add(play)
                policy.gold_pairs.add((play, party))
                policy.gold_sentences.append((sent["sentence_text"], play, party))
    return policy


def policy_ids(split: str = "TEST") -> list[int]:
    """IDs of all policies in one split, in numeric order."""
    ids = []
    for path in (APP350_DIR / "annotations").glob("policy_*.yml"):
        # The split is on line 3; avoid parsing 350 YAML files just to read it.
        with path.open() as f:
            head = "".join(next(f, "") for _ in range(4))
        if f"policy_type: {split}" in head:
            ids.append(int(path.stem.split("_")[1]))
    return sorted(ids)
