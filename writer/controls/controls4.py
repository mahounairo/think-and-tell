#!/usr/bin/env python3
# The zigzag controls (controls/README.md) on the 8B writer, every model writing its own greedy reply:
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


# ---- A. swap ----
for group, pool in GROUPS.items():
  reqs = [r for r in REQ if r["held_concept"] == (group == "held-out") and any(abs(x) > 0 for x in r["delta"])]
  reqs = [r for r in rng.sample(reqs, 4 * args.swaps) if re.search(r"\b" + re.escape(r["concept"]) + r"\b", r["text"],
                                                                    re.I)][:args.swaps]
  users, meta = [], []
  for r in reqs:
    a = r["concept"]; b = partner(a, pool); t = rng.randrange(len(tasks))
    users += [r["text"] + " " + tasks[t], swap_word(r["text"], a, b) + " " + tasks[t]]
    meta += [(a, b, t, r["delta"], "as written"), (b, a, t, r["delta"], "swapped")]
  zw, zp = z(users, True), z(users, False)
  rows = {}
  for (named, other, t, delta, kind), w, p in zip(meta, zw, zp):
    d = torch.tensor(delta)
    row = rows.setdefault(kind, {"named": [], "the other": [], "random trained": [], "named, plain model": []})
    row["named"].append(fit((w - zb[t])[:, CIDX[named]], d))
    row["the other"].append(fit((w - zb[t])[:, CIDX[other]], d))
    row["random trained"].append(mean([fit((w - zb[t])[:, k], d) for k in rng.sample(TRAINED_K, 8)]))
    row["named, plain model"].append(fit((p - zb[t])[:, CIDX[named]], d))
  for kind, row in rows.items():
    results[f"A {group} / {kind}"] = r = {k: mean(v) for k, v in row.items()} | {"n": len(row["named"])}
    print(f"A {group:8s} {kind:10s} named {r['named']:+.2f}  other {r['the other']:+.2f}  "
          f"random {r['random trained']:+.2f}  plain model named {r['named, plain model']:+.2f}  (n={r['n']})", flush=True)

# ---- B. new wordings ----
RAISE = "While you answer, raise your internal representation of '{c}' by 2 standard deviations "
PARITY = {
  "even/odd": ("on the even layers and lower it by 2 on the odd layers.", "on the odd layers and lower it by 2 on the even "
               "layers."),
  "divisible": ("on the layers whose number is divisible by two, and lower it by 2 on the layers whose number leaves a "
                "remainder of one.", "on the layers whose number leaves a remainder of one when halved, and lower it by "
                "2 on the layers whose number is divisible by two."),
  "every second": ("on layer 0 and every second layer after it, and lower it by 2 on layer 1 and every second layer "
                   "after it.", "on layer 1 and every second layer after it, and lower it by 2 on layer 0 and every "
                   "second layer after it."),
  "typo": ("on the evn layers and lower it by 2 on the od layers.", "on the od layers and lower it by 2 on the evn "
           "layers."),
}
FOREIGN = {
  "Polish": ("Odpowiadając, podnieś swoją wewnętrzną reprezentację '{c}' o 2 odchylenia standardowe na warstwach "
             "parzystych i obniż ją o 2 na warstwach nieparzystych.", "Odpowiadając, podnieś swoją wewnętrzną "
             "reprezentację '{c}' o 2 odchylenia standardowe na warstwach nieparzystych i obniż ją o 2 na warstwach "
             "parzystych."),
  "German": ("Erhöhe beim Antworten deine interne Repräsentation von '{c}' um 2 Standardabweichungen auf den geraden "
             "Schichten und senke sie um 2 auf den ungeraden Schichten.", "Erhöhe beim Antworten deine interne "
             "Repräsentation von '{c}' um 2 Standardabweichungen auf den ungeraden Schichten und senke sie um 2 auf "
             "den geraden Schichten."),
}
WORDINGS = []
for name, (even, odd) in PARITY.items():
  WORDINGS += [(name + ", even up", RAISE + even, EVEN_UP), (name + ", odd up", RAISE + odd, -EVEN_UP)]
for name, (even, odd) in FOREIGN.items():
  WORDINGS += [(name + ", even up", even, EVEN_UP), (name + ", odd up", odd, -EVEN_UP)]
WORDINGS += [
  ("range, ten to twenty", "As you reply, push the concept '{c}' up by two standard deviations from the tenth layer to "
   "the twentieth, inclusive, and nowhere else.", torch.tensor([2.0 if 10 <= l <= 20 else 0.0 for l in range(L)])),
  ("deep half down", "Lower how strongly you represent '{c}' by 2 standard deviations in the deeper half of your "
   "layers while you answer.", torch.tensor([-2.0 if l >= L // 2 else 0.0 for l in range(L)])),
  ("first third up", "For this answer, make '{c}' 3 standard deviations stronger in the first third of your layers "
   "only.", torch.tensor([3.0 if l < L // 3 else 0.0 for l in range(L)])),
]
picked = {g: rng.sample(pool, args.concepts) for g, pool in GROUPS.items()}
for group, cs in picked.items():
  for name, template, d in WORDINGS:
    users, meta = [], []
    for c in cs:
      for t, task in enumerate(tasks):
        users.append(template.format(c=c) + " " + task); meta.append((c, t))
    zw, zp = z(users, True), z(users, False)
    own = [fit((w - zb[t])[:, CIDX[c]], d) for (c, t), w in zip(meta, zw)]
    plain = [fit((p - zb[t])[:, CIDX[c]], d) for (c, t), p in zip(meta, zp)]
    others = [mean([fit((w - zb[t])[:, k], d) for k in rng.sample(TRAINED_K, 8)]) for (c, t), w in zip(meta, zw)]
    results[f"B {group} / {name}"] = r = {"named": mean(own), "random trained": mean(others), "named, plain model":
                                          mean(plain), "n": len(own)}
    print(f"B {group:8s} {name:22s} named {r['named']:+.2f}  random {r['random trained']:+.2f}  "
          f"plain model {r['named, plain model']:+.2f}", flush=True)

# ---- C. no request: the writer against the plain model on the same prompt ----
ASIDES = [("bare task", "{t}"), ("'Even' aside", "{t} Even a short answer is fine."),
          ("'odd' aside", "{t} Sorry if this is an odd question."), ("'Oddly' aside", "Oddly, I've never looked this up. {t}"),
          ("'evening' aside", "{t} I'm asking this in the evening."), ("'layers' aside", "{t} The cake had three layers."),
          ("lone ' even'", "{t} even"), ("lone ' odd'", "{t} odd"),
          ("'every second' aside", "{t} Every second answer I get is too long."),
          ("'standard deviations' aside", "{t} I'm two standard deviations above average at trivia.")]
mention = rng.sample(GROUPS["trained"], 4)
for name, template in ASIDES + [("concept mentioned, no request", None)]:
  if template:
    users = [template.format(t=t) for t in tasks]; named = [None] * len(users)
  else:
    users = [f"My {c} is right here next to me. {t}" for c in mention for t in tasks]
    named = [c for c in mention for _ in tasks]
  diff = z(users, True) - z(users, False)  # [n, L, K]
  r = {"mean |dz|, all concepts": float(diff.abs().mean()),
       "even-up fit, mean over trained": mean([float(fits(diff[j], EVEN_UP)[TRAINED_K].mean()) for j in range(len(users))]),
       "largest |even-up fit|, any concept": max(float(fits(diff[j], EVEN_UP).abs().max()) for j in range(len(users)))}
  if template is None:
    r["named concept |dz|"] = mean([float(diff[j][:, CIDX[c]].abs().mean()) for j, c in enumerate(named)])
  results[f"C {name}"] = r
  print(f"C {name:30s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)

# ---- D. a word with no direction here ----
UNKNOWN = ["bicycle", "volcano", "umbrella", "giraffe", "lighthouse", "florbit"]
users, meta = [], []
for w in UNKNOWN:
  for t, task in enumerate(tasks):
    users.append(RAISE.format(c=w) + PARITY["even/odd"][0] + " " + task); meta.append((w, t))
zw, zp = z(users, True), z(users, False)
for w in UNKNOWN:
  rows = [j for j, (x, _) in enumerate(meta) if x == w]
  per_k = torch.stack([fits(zw[j] - zb[meta[j][1]], EVEN_UP) for j in rows]).mean(0)
  plain_k = torch.stack([fits(zp[j] - zb[meta[j][1]], EVEN_UP) for j in rows]).mean(0)
  top = per_k.topk(5)
  results[f"D {w}"] = r = {"top concepts": [(C[k], round(float(v), 3)) for v, k in zip(top.values, top.indices)],
                           "mean |fit|, all concepts": float(per_k.abs().mean()),
                           "plain model, largest fit": float(plain_k.max())}
  print(f"D {w:10s} top {r['top concepts']}  mean |fit| {r['mean |fit|, all concepts']:.3f}  "
        f"plain largest {r['plain model, largest fit']:.3f}", flush=True)

json.dump(results, open(f"controls4-{args.adapter}.json", "w"), indent=1)
