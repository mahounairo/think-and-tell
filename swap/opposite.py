#!/usr/bin/env python3
# Does "don't think of X" read as minus X? For every word, the reader's answer to the writer lowering X is compared with
# its answers to X's own state vector negated, to the writer raising X, and so on; each pair is scored as the rank of
# the same word's answer among all words' answers (gte). Answers the reader gives as a "no" are left out on either side,
# since identical "I feel blank" answers would match each other whatever was pushed.
#   opposite.py   (reads results/pair-raise-full.json and results/pair-lower-full.json)
import json

import torch
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("thenlper/gte-base")
reports = json.load(open("../reader/reports-local-full.json"))
split = json.load(open("../reader/runs/self-instruct-split.json"))
NO = embedder.encode(list(reports["no"]), normalize_embeddings=True, convert_to_tensor=True)
POOL = embedder.encode([r for s in split["train"][:400] for r in reports["states"][s]["report"]],
                       normalize_embeddings=True, convert_to_tensor=True)
lower = json.load(open("results/pair-lower-full.json"))["samples"]
raise_ = json.load(open("results/pair-raise-full.json"))["samples"]
rows = {"lowered X": lower["reader / writer, at the state's size"], "X negated": lower["reader / state vector, negated"],
        "raised X": raise_["reader / writer, at the state's size"], "X": raise_["reader / state vector (ceiling)"]}
emb = {k: embedder.encode([a for _, a in v], normalize_embeddings=True, convert_to_tensor=True) for k, v in rows.items()}
is_no = {k: (v @ NO.T).max(1).values > (v @ POOL.T).max(1).values for k, v in emb.items()}

for a, b in (("lowered X", "X negated"), ("lowered X", "raised X"), ("raised X", "X"), ("X negated", "X"),
             ("lowered X", "X"), ("raised X", "X negated")):
  keep = (~is_no[a] & ~is_no[b]).nonzero().squeeze(1)
  s = emb[a][keep] @ emb[b][keep].T
  n = len(s)
  rank = (s > s.diag()[:, None]).sum(1) + 1
  print(f"{a:10s} vs {b:10s} words {n:3d}  same word in top 10 {float((rank <= 10).float().mean()):.1%} "
        f"(chance {10 / n:.1%})  mean rank {float(rank.float().mean()):.0f} of {n}")
