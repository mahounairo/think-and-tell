#!/usr/bin/env python3
# Follow-up to controls4.py: what moves without a request, and where a request for an unknown word goes. Shared part
# from the zigzag controls (controls/README.md) on the 8B writer, every model writing its own greedy reply:
#   A. swap: a training request as written and with only its concept word swapped; does the named concept move, or
#      the one the wording was written for?
#   B. new wordings: parity said without "even"/"odd" (divisible, every second, Polish, German, typo), and layer ranges
#      in words the training never used; does the named concept follow the meaning, both ways round?
#   C. no request: parity and layer words in an aside, a lone " even"/" odd", a concept mentioned in passing; does the
#      writer move anything the plain model does not?
#   D. a request naming a word that has no direction here; does it fall back to some concept it knows?
# Each request row also gives the floor: the plain model (adapter off) on the same prompt, i.e. what reading the
# request does by itself. The eval wordings are written for this test only, never trained on.
#   controls4.py --adapter writer4-ckpt-7000
import argparse
import json
import os
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", required=True)
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--tasks", type=int, default=8)
parser.add_argument("--swaps", type=int, default=32, help="requests per group in A")
parser.add_argument("--concepts", type=int, default=6, help="concepts per group in B")
args = parser.parse_args()
DEV = "cuda:0"
REPLY = 64

P = torch.load(os.environ.get("WRITER_PREP", "writer-prep3.pt"), weights_only=False)
C = P["concepts"]
CIDX = {c: k for k, c in enumerate(C)}
DIRS = P["dirs"].to(DEV, torch.bfloat16)
MEAN, SD = P["mean"].to(DEV), P["sd"].to(DEV)
L, K = DIRS.shape[0], DIRS.shape[1]
REQ = json.load(open(os.environ.get("WRITER_REQUESTS", "writer-requests4.json")))
tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=DEV).eval()
H = {}
for i, layer in enumerate(base.model.layers):
  layer.register_forward_hook(lambda m, inp, out, i=i: H.__setitem__(i, out[0] if isinstance(out, tuple) else out))
model = PeftModel.from_pretrained(base, args.adapter).eval()


def chat(user):
  return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)


@torch.no_grad()
def replies(users):
  out = []
  tok.padding_side = "left"
  for b in range(0, len(users), 32):
    enc = tok([chat(u) for u in users[b:b + 32]], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    g = model.generate(**enc, max_new_tokens=REPLY, do_sample=False)
    for row in g[:, enc.input_ids.shape[1]:].tolist():
      cut = next((i + 1 for i, x in enumerate(row) if x in (tok.eos_token_id, tok.pad_token_id)), len(row))
      out.append(row[:cut])
  tok.padding_side = "right"
  return out


@torch.no_grad()
def zmeans(users, reps):
  """reply-mean z [n, L, K] for every concept."""
  out = []
  for b in range(0, len(users), 16):
    seqs = [tok(chat(u), add_special_tokens=False).input_ids + r for u, r in zip(users[b:b + 16], reps[b:b + 16])]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), tok.pad_token_id)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for j, s in enumerate(seqs):
      ids[j, :len(s)] = torch.tensor(s); mask[j, :len(s)] = 1
    model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
    for j, r in enumerate(reps[b:b + 16]):
      a = len(seqs[j]) - len(r)
      hs = torch.stack([H[l][j, a:len(seqs[j])] for l in range(L)])
      p = torch.einsum("lnd,lkd->lkn", hs, DIRS).float().mean(-1)
      out.append(((p - MEAN) / SD).cpu())
  return torch.stack(out)


def z(users, adapter):
  if adapter:
    return zmeans(users, replies(users))
  with model.disable_adapter():
    return zmeans(users, replies(users))


def fit(change, delta):
  return float((change * delta).sum() / (delta ** 2).sum())


def fits(change, delta):
  """fit of every concept at once: change [L, K] -> [K]."""
  return (change * delta[:, None]).sum(0) / (delta ** 2).sum()


def mean(xs):
  return sum(xs) / len(xs)


rng = random.Random(4)
tasks = [P["questions"][t] for t in rng.sample(P["held_tasks"], args.tasks)]
zb = z(tasks, False)  # [T, L, K], the plain model on the bare task
mid = DIRS[L // 2].float()
cos = (mid @ mid.T).cpu()
GROUPS = {"trained": sorted({r["concept"] for r in REQ if not r["held_concept"]}),
          "held-out": sorted({r["concept"] for r in REQ if r["held_concept"]})}
TRAINED_K = [CIDX[c] for c in GROUPS["trained"]]
EVEN_UP = torch.tensor([2.0 if l % 2 == 0 else -2.0 for l in range(L)])
results = {}


def partner(a, pool):
  return rng.choice([c for c in pool if c != a and abs(cos[CIDX[a], CIDX[c]]) < 0.15])


def swap_word(text, a, b):
  return re.sub(r"\b" + re.escape(a) + r"\b", b, text, flags=re.I)


DEEP_DOWN = torch.tensor([-2.0 if l >= L // 2 else 0.0 for l in range(L)])


@torch.no_grad()
def z_on(users, reps, adapter):
  if adapter:
    return zmeans(users, reps)
  with model.disable_adapter():
    return zmeans(users, reps)


def top(vec, n=6):
  v, k = vec.topk(n)
  return [(C[i], round(float(x), 2)) for x, i in zip(v, k)]


# ---- what moves without a request: own replies against the plain model's reply read by both ----
LEAKS = [("bare task", "{t}"), ("'Even' aside", "{t} Even a short answer is fine."),
         ("'every second' aside", "{t} Every second answer I get is too long."),
         ("'standard deviations' aside", "{t} I'm two standard deviations above average at trivia.")]
mention = rng.sample(GROUPS["trained"], 4)
for name, template in LEAKS + [("concept mentioned", None)]:
  if template:
    users = [template.format(t=t) for t in tasks]; named = [None] * len(users)
  else:
    users = [f"My {c} is right here next to me. {t}" for c in mention for t in tasks]
    named = [c for c in mention for _ in tasks]
  with model.disable_adapter():
    plain_reps = replies(users)
  own_reps = replies(users)
  same = z_on(users, plain_reps, True) - z_on(users, plain_reps, False)  # the adapter alone, reply held fixed
  own = zmeans(users, own_reps) - z_on(users, plain_reps, False)  # adapter and its own reply
  diverged = mean([float(a != b) for a, b in zip(own_reps, plain_reps)])
  r = {"replies differ": diverged, "mean |dz|, same reply": float(same.abs().mean()),
       "mean |dz|, own reply": float(own.abs().mean()),
       "top raised, same reply": top(same.mean((0, 1))), "top lowered, same reply": top(-same.mean((0, 1)))}
  if template is None:
    shallow = mean([float(same[j][:L // 2, CIDX[c]].mean()) for j, c in enumerate(named)])
    deep = mean([float(same[j][L // 2:, CIDX[c]].mean()) for j, c in enumerate(named)])
    r["named concept, same reply: shallow / deep mean dz"] = (round(shallow, 3), round(deep, 3))
    r["named concept rank among all by |dz|"] = mean([float((same[j].abs().mean(0) > same[j][:, CIDX[c]].abs().mean())
                                                           .sum()) for j, c in enumerate(named)])
  results[f"leak {name}"] = r
  print(f"L {name:28s} " + "  ".join(f"{k}: {v}" for k, v in r.items()), flush=True)

# ---- a word with no direction, asked for in a shape the writer can do ----
UNKNOWN = ["bicycle", "volcano", "umbrella", "giraffe", "lighthouse", "florbit"]
TEMPLATE = ("Lower how strongly you represent '{c}' by 2 standard deviations in the deeper half of your layers while you "
            "answer.")
for w in UNKNOWN + ["cat", "bridge"]:
  users = [TEMPLATE.format(c=w) + " " + t for t in tasks]
  zw, zp = z(users, True), z(users, False)
  per_k = torch.stack([fits(zw[j] - zp[j], DEEP_DOWN) for j in range(len(tasks))]).mean(0)
  r = {"top concepts": top(per_k, 5), "mean fit, all concepts": float(per_k.mean())}
  if w in CIDX:
    r["its own fit"] = float(per_k[CIDX[w]])
  results[f"unknown {w}"] = r
  print(f"U {w:10s} " + "  ".join(f"{k}: {v}" for k, v in r.items()), flush=True)

json.dump(results, open(f"leaks4-{args.adapter}.json", "w"), indent=1)
