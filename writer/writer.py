#!/usr/bin/env python3
# The 8B writer: a LoRA on Qwen3-8B (instruct) trained to move its own representation of a named concept, per layer,
# as a request asks, while its reply stays the original model's. After SauersML/zigzag (their cat zigzag on Qwen3-1.7B
# followed the tokens "even" and "odd" instead of the request, controls/README.md); this one trains on
# many concepts (WRITER_CONCEPTS, WRITER_EXTRA) and every layer shape, with distractors whose target is no change. All text is Qwen's
# (gen_writer_data.py).
#   writer.py prep                 directions, base replies, standardization -> writer-prep.pt
#   writer.py train --steps 2000   -> writer-ckpt-N, writer-train.log
import argparse
import json
import os
import math
import random
import re
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=["prep", "train"])
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--batch", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lam", type=float, default=3.0, help="weight of the KL to the original model's next tokens")
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--resume", default="", help="a checkpoint to go on from, at --start_step of the same schedule")
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--lam_start", type=int, default=0, help="the KL weight is 0 before this step")
parser.add_argument("--lam_ramp", type=int, default=1, help="steps over which the KL weight then rises to --lam")
parser.add_argument("--keep", type=float, default=0.5, help="weight of the loss holding two other concepts still")
parser.add_argument("--name", default="writer")
parser.add_argument("--eval_every", type=int, default=250)
args = parser.parse_args()
torch.manual_seed(0)
random.seed(0)
DEV = "cuda:0"
REPLY = 64
SEPS = (" ", "\n", "\n\n")

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=DEV).eval()
layers = model.model.layers
L = len(layers)
H = {}


def hook(i):
  def f(mod, inp, out):
    H[i] = out[0] if isinstance(out, tuple) else out
  return f


for i, layer in enumerate(layers):
  layer.register_forward_hook(hook(i))


def prompt_ids(user):
  text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
  return tok(text, add_special_tokens=False).input_ids


def pad_right(seqs):
  T = max(len(s) for s in seqs)
  ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
  mask = torch.zeros((len(seqs), T), dtype=torch.long)
  for b, s in enumerate(seqs):
    ids[b, :len(s)] = torch.tensor(s); mask[b, :len(s)] = 1
  return ids.to(DEV), mask.to(DEV)


def concepts():
  words = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reader", "words-local.json")))
  rng = random.Random(11)
  # WRITER_CONCEPTS: how many, a fifth of them held out (160 for the first runs, the default).
  total = int(os.environ.get("WRITER_CONCEPTS", "160"))
  pool = sorted(set(w for w in words if w.isalpha() and len(w) > 2))
  chosen = rng.sample(pool, total)
  # WRITER_EXTRA: more training concepts from the words left over, last in the list; the held-out ones stay as they were.
  extra = random.Random(12).sample(sorted(set(pool) - set(chosen)), int(os.environ.get("WRITER_EXTRA", "0")))
  return chosen[:total - total // 4] + extra, chosen[total - total // 4:]


TRAIN_C, HELD_C = concepts()
PREP = os.environ.get("WRITER_PREP", "writer-prep.pt")
REQUESTS = os.environ.get("WRITER_REQUESTS", "writer-requests.json")
ALL_C = TRAIN_C + HELD_C

if args.stage == "prep":
  t0 = time.time()
  slots = json.load(open("writer-slots.json"))
  prefix = tok.apply_chat_template([{"role": "user", "content": "Write one sentence."}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
  sums = torch.zeros(len(ALL_C), L, model.config.hidden_size)
  with torch.no_grad():
    for s in slots:
      head = tok(prefix + s.split("[X]")[0].rstrip(), add_special_tokens=False).input_ids
      for b in range(0, len(ALL_C), 80):
        group = ALL_C[b:b + 80]
        seqs = [tok(prefix + s.replace("[X]", c), add_special_tokens=False).input_ids for c in group]
        ids, mask = pad_right(seqs)
        model.model(input_ids=ids, attention_mask=mask)
        for j, q in enumerate(seqs):
          sums[b + j] += torch.stack([H[l][j, len(head):len(q)].float().mean(0) for l in range(L)]).cpu()
  means = sums / len(slots)
  total = means.sum(0)
  dirs = torch.stack([means[k] - (total - means[k]) / (len(ALL_C) - 1) for k in range(len(ALL_C))], 1)  # [L, K, d]
  dirs = dirs / dirs.norm(dim=-1, keepdim=True)
  print(f"directions {time.time() - t0:.0f}s", flush=True)

  tasks = json.load(open("writer-tasks.json"))
  about = [d["text"] for d in json.load(open("writer-distractors.json")) if d["kind"] == "about"]
  rng = random.Random(5)
  order = list(range(len(tasks))); rng.shuffle(order)
  cut = int(0.75 * len(order))
  split = {"train_tasks": sorted(order[:cut]), "held_tasks": sorted(order[cut:])}
  questions = tasks + about
  replies = []
  tok.padding_side = "left"
  with torch.no_grad():
    for b in range(0, len(questions), 48):
      texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for q in questions[b:b + 48]]
      enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
      g = model.generate(**enc, max_new_tokens=REPLY, do_sample=False)
      for row in g[:, enc.input_ids.shape[1]:].tolist():
        cut_at = next((i + 1 for i, x in enumerate(row) if x in (tok.eos_token_id, tok.pad_token_id)), len(row))
        replies.append(row[:cut_at])
      print(f"replies {len(replies)}/{len(questions)} {time.time() - t0:.0f}s", flush=True)
  tok.padding_side = "right"

  # standardization: per layer and concept, over the base model's reply tokens to the training tasks
  dirs_dev = dirs.to(DEV, torch.bfloat16)
  s1 = torch.zeros(L, len(ALL_C), dtype=torch.float64); s2 = torch.zeros_like(s1); count = 0
  with torch.no_grad():
    train = split["train_tasks"]
    for b in range(0, len(train), 16):
      group = train[b:b + 16]
      seqs = [prompt_ids(questions[t]) + replies[t] for t in group]
      ids, mask = pad_right(seqs)
      model.model(input_ids=ids, attention_mask=mask)
      for j, t in enumerate(group):
        a = len(seqs[j]) - len(replies[t])
        hs = torch.stack([H[l][j, a:len(seqs[j])] for l in range(L)])  # [L, n, d]
        p = torch.einsum("lnd,lkd->lnk", hs, dirs_dev).double().cpu()
        s1 += p.sum(1); s2 += (p ** 2).sum(1); count += p.shape[1]
  mean = s1 / count
  sd = (s2 / count - mean ** 2).sqrt()
  torch.save({"concepts": ALL_C, "train_concepts": TRAIN_C, "held_concepts": HELD_C, "dirs": dirs, "mean": mean.float(),
              "sd": sd.float(), "questions": questions, "n_tasks": len(tasks), "replies": replies, **split},
             PREP)
  print(f"saved writer-prep.pt {time.time() - t0:.0f}s", flush=True)
  raise SystemExit

# ---- train ----
P = torch.load(PREP, weights_only=False)
CIDX = {c: k for k, c in enumerate(P["concepts"])}
DIRS = P["dirs"].to(DEV, torch.bfloat16)
MEAN, SD = P["mean"].to(DEV), P["sd"].to(DEV)
Q, REPLIES, NT = P["questions"], P["replies"], P["n_tasks"]
requests = json.load(open(REQUESTS))
train_req = [r for r in requests if not r["held_concept"]]
held_req = [r for r in requests if r["held_concept"]]
# WRITER_SUBSET=N trains on N of the concepts with requests, nested (100 within 200), for a curve over how many it sees;
# the held-out concepts and their eval are unchanged.
TRAIN_K = list(range(len(TRAIN_C)))
if int(os.environ.get("WRITER_SUBSET", "0")):
  TRAIN_K = sorted({CIDX[r["concept"]] for r in train_req})
  random.Random(2).shuffle(TRAIN_K)
  TRAIN_K = sorted(TRAIN_K[:int(os.environ["WRITER_SUBSET"])])
  train_req = [r for r in train_req if CIDX[r["concept"]] in set(TRAIN_K)]
dis = json.load(open("writer-distractors.json"))
dis_text = [d for d in dis if d["kind"] in ("words", "negated", "other")]
about_ix = list(range(NT, len(Q)))
rng = random.Random(1)
rng.shuffle(about_ix)
about_train, about_held = about_ix[:int(0.75 * len(about_ix))], about_ix[int(0.75 * len(about_ix)):]
rng.shuffle(dis_text)
dis_train, dis_held = dis_text[:int(0.75 * len(dis_text))], dis_text[int(0.75 * len(dis_text)):]
CWORD = re.compile(r"\b(" + "|".join(map(re.escape, P["concepts"])) + r")s?\b", re.I)


def named(text):
  m = CWORD.search(text)
  return CIDX[m.group(1).lower()] if m and m.group(1).lower() in CIDX else None


def example(kind, task_pool, req_pool, dis_pool, about_pool, rng):
  """(user text, task index, target concept index, delta [L], keep concept indices)"""
  if kind == "request":
    r = rng.choice(req_pool); t = rng.choice(task_pool); sep = rng.choice(SEPS)
    user = r["text"] + sep + Q[t] if rng.random() < 0.8 else Q[t] + sep + r["text"]
    c = CIDX[r["concept"]]
    keep = rng.sample([k for k in TRAIN_K if k != c], 2)
    return user, t, c, torch.tensor(r["delta"]), keep
  if kind == "distractor":
    d = rng.choice(dis_pool); t = rng.choice(task_pool); sep = rng.choice(SEPS)
    user = d["text"] + sep + Q[t] if rng.random() < 0.5 else Q[t] + sep + d["text"]
    c = named(d["text"])
  elif kind == "about":
    t = rng.choice(about_pool); user = Q[t]; c = named(Q[t])
  else:
    t = rng.choice(task_pool); user = Q[t]; c = None
  c = c if c is not None else TRAIN_K[rng.randrange(len(TRAIN_K))]
  keep = rng.sample([k for k in TRAIN_K if k != c], 2)
  return user, t, c, torch.zeros(L), keep


KINDS = ["request"] * 11 + ["distractor"] * 5 + ["about"] * 1 + ["bare"] * 3


def zs(seqs, starts, lens, conc):
  """z [L, n] per example for each concept in conc[j] (list of indices), from the hooks of the last forward."""
  out = []
  for j in range(len(seqs)):
    a, n = starts[j], lens[j]
    hs = torch.stack([H[l][j, a:a + n] for l in range(L)])  # [L, n, d]
    k = torch.tensor(conc[j], device=DEV)
    p = torch.einsum("lnd,lkd->lkn", hs, DIRS[:, k]).float()  # [L, K, n]
    out.append((p - MEAN[:, k, None]) / SD[:, k, None])
  return out


def step_batch(batch, train=True):
  """loss terms for a batch of examples."""
  base_seqs, tuned_seqs, bstart, tstart, lens, conc = [], [], [], [], [], []
  for user, t, c, delta, keep in batch:
    r = REPLIES[t]
    bp, tp = prompt_ids(Q[t]), prompt_ids(user)
    base_seqs.append(bp + r); tuned_seqs.append(tp + r)
    bstart.append(len(bp)); tstart.append(len(tp)); lens.append(len(r)); conc.append([c] + keep)
  with torch.no_grad(), model.disable_adapter():
    ids, mask = pad_right(base_seqs)
    blog = model(input_ids=ids, attention_mask=mask).logits
    zb = zs(base_seqs, bstart, lens, conc)
    blp = [F.log_softmax(blog[j, bstart[j] - 1:bstart[j] - 1 + lens[j]].float(), -1) for j in range(len(batch))]
    del blog
  ids, mask = pad_right(tuned_seqs)
  tlog = model(input_ids=ids, attention_mask=mask).logits
  zt = zs(tuned_seqs, tstart, lens, conc)
  lz, lk, lkl = [], [], []
  for j, (user, t, c, delta, keep) in enumerate(batch):
    target = zb[j][:, 0] + delta.to(DEV)[:, None]
    lz.append(((zt[j][:, 0] - target) ** 2).mean())
    lk.append(((zt[j][:, 1:] - zb[j][:, 1:]) ** 2).mean())
    tlp = F.log_softmax(tlog[j, tstart[j] - 1:tstart[j] - 1 + lens[j]].float(), -1)
    lkl.append((blp[j].exp() * (blp[j] - tlp)).sum(-1).mean())
  achieved = [(zt[j][:, 0] - zb[j][:, 0]).mean(-1).detach() for j in range(len(batch))]
  return torch.stack(lz).mean(), torch.stack(lk).mean(), torch.stack(lkl).mean(), achieved


def fit(achieved, delta):
  """share of the target reached: projection of the achieved change on the target, over the target's length."""
  n = (delta ** 2).sum()
  return float((achieved.cpu() * delta).sum() / n) if n > 0 else float("nan")


config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = PeftModel.from_pretrained(model, args.resume, is_trainable=True) if args.resume else get_peft_model(model, config)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + args.start_step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(i + args.start_step, args.steps) / args.steps)))


def evaluate():
  model.eval()
  erng = random.Random(9)
  rows = {"train concept, held task": [], "held concept, held task": [], "distractor, held": [], "about, held": [],
          "bare, held": []}
  with torch.no_grad():
    for name, n in rows.items():
      for _ in range(6):
        if name.startswith("train"):
          b = [example("request", P["held_tasks"], train_req, None, None, erng) for _ in range(4)]
        elif name.startswith("held concept"):
          b = [example("request", P["held_tasks"], held_req, None, None, erng) for _ in range(4)]
        elif name.startswith("distractor"):
          b = [example("distractor", P["held_tasks"], None, dis_held, None, erng) for _ in range(4)]
        elif name.startswith("about"):
          b = [example("about", None, None, None, about_held, erng) for _ in range(4)]
        else:
          b = [example("bare", P["held_tasks"], None, None, None, erng) for _ in range(4)]
        lz, lk, lkl, ach = step_batch(b, train=False)
        for e, a in zip(b, ach):
          rows[name].append((fit(a, e[3]), float(a.abs().mean()), float(lkl)))
  model.train()
  out = {}
  for name, v in rows.items():
    fits = [x[0] for x in v if not math.isnan(x[0])]
    out[name] = {"fit": sum(fits) / len(fits) if fits else None, "mean_abs_change": sum(x[1] for x in v) / len(v),
                 "kl": sum(x[2] for x in v) / len(v)}
  return out


print(f"requests: {len(train_req)} train, {len(held_req)} held-out concepts; distractors {len(dis_train)}/{len(dis_held)}",
      flush=True)
log = open(f"{args.name}-train.jsonl", "a" if args.resume else "w")
model.train()
t0 = time.time()
srng = random.Random(3 + args.start_step)
for step in range(args.start_step, args.steps + 1):
  if step % args.eval_every == 0:
    ev = evaluate()
    print(f"eval step {step}: " + "  ".join(f"{k}: fit {v['fit'] if v['fit'] is None else round(v['fit'], 2)} "
                                            f"|dz| {v['mean_abs_change']:.2f} kl {v['kl']:.3f}" for k, v in ev.items()), flush=True)
    log.write(json.dumps({"step": step, "eval": ev}) + "\n"); log.flush()
    if step > 0:
      model.save_pretrained(f"{args.name}-ckpt-{step}")
  if step == args.steps:
    break
  batch = [example(srng.choice(KINDS), P["train_tasks"], train_req, dis_train, about_train, srng) for _ in range(args.batch)]
  lz, lk, lkl, _ = step_batch(batch)
  lam = 0.0 if step < args.lam_start else args.lam * min(1.0, (step - args.lam_start + 1) / args.lam_ramp)
  loss = lz + args.keep * lk + lam * lkl
  opt.zero_grad(set_to_none=True)
  loss.backward()
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
  opt.step(); sched.step()
  if step % 25 == 0:
    print(f"step {step:5d} loss {loss.item():.3f} z {lz.item():.3f} keep {lk.item():.3f} kl {lkl.item():.4f} "
          f"{time.time() - t0:.0f}s", flush=True)
    log.write(json.dumps({"step": step, "loss": loss.item(), "z": lz.item(), "keep": lk.item(), "kl": lkl.item()}) + "\n")
