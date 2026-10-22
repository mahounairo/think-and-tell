#!/usr/bin/env python3
# Both adapters in one conversation (2026-10-22): the writer is asked in a chat turn to think of a word (or not to)
# while it answers; then, in the same context, the check-in follows as plain text and the reader continues it. Nothing
# is carried over by hand: whatever the reader reads at layer 16 is what the writer left there. The reply is the plain
# model's, so the text is the same in every condition and only the adapters differ:
#   split      the writer on layers up to the reader's (it writes below), the reader above it (it reads)
#   both       both adapters on every layer they have
#   reader     the reader alone: what the words of the request do by themselves (they name the word)
#   writer     the writer alone, no reader
#   neither    the untrained model
# Scored as pair.py: gte, the true word's rank among the 222 words both adapters know; "no" against the trained no's.
# Also the word's own standardized change at layer 16 over the check-in's tokens, writer on against off.
#   selfloop.py --writer writer/writer4-ckpt-7000 --reader self-instruct-ckpt-3000
import argparse
import json
import random
import re

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--writer", default="../writer/writer4-ckpt-7000")
parser.add_argument("--reader", default="../reader/self-instruct-ckpt-3000")
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--size", type=float, default=3.0)
parser.add_argument("--out", default="selfloop-results.json")
args = parser.parse_args()
DEV = "cuda:0"
rng = random.Random(0)

data = torch.load("../reader/dataset-instruct.pt", weights_only=False)
LAYER = data["layer"]
reports = data["reports"]
split = json.load(open("../reader/runs/self-instruct-split.json"))
P = torch.load("../writer/writer-prep3.pt", weights_only=False)
WORDS = sorted(set(P["concepts"]) & set(data["labels"]))
TEST = set(split["test"])
FRAMES = split["test_frames"]
TASKS = [P["questions"][t] for t in rng.sample(P["held_tasks"], 4)]
CIDX = {c: k for k, c in enumerate(P["concepts"])}
DIR16 = P["dirs"][LAYER].float()  # [K, d]
MEAN16, SD16 = P["mean"][LAYER].float(), P["sd"][LAYER].float()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=DEV).eval()
model = PeftModel.from_pretrained(model, args.writer, adapter_name="writer")
model.load_adapter(args.reader, adapter_name="reader")
model.eval()
EOS = tok.eos_token_id

# The writer's LoRA above the reader's layer, so "split" can switch it off there.
ABOVE = []
for name, module in model.named_modules():
  m = re.search(r"\.layers\.(\d+)\.", name)
  if m and int(m.group(1)) > LAYER and isinstance(getattr(module, "scaling", None), dict) and "writer" in module.scaling:
    ABOVE.append((module, module.scaling["writer"]))
print(f"writer LoRA modules above layer {LAYER}: {len(ABOVE)}", flush=True)

capture = {"on": False, "h": None}


def hook(module, inputs, output):
  if capture["on"]:
    capture["h"] = (output[0] if isinstance(output, tuple) else output).float()


model.base_model.model.model.layers[LAYER].register_forward_hook(hook)


def use(condition):
  for module, scale in ABOVE:
    module.scaling["writer"] = 0.0 if condition == "split" else scale
  active = {"split": ["writer", "reader"], "both": ["writer", "reader"], "reader": ["reader"], "writer": ["writer"]}
  if condition != "neither":
    model.base_model.set_adapter(active[condition])


def chat(user):
  return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)


@torch.no_grad()
def generate(prompts, n, condition, stop_newline=True):
  out = []
  tok.padding_side = "left"
  for b in range(0, len(prompts), 24):
    enc = tok(prompts[b:b + 24], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
    if condition == "neither":
      with model.disable_adapter():
        g = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=EOS)
    else:
      use(condition)
      g = model.generate(**enc, max_new_tokens=n, do_sample=False, pad_token_id=EOS)
    for row in g[:, enc.input_ids.shape[1]:]:
      text = tok.decode(row, skip_special_tokens=True)
      out.append(text.split("\n")[0].strip() if stop_newline else text)
  tok.padding_side = "right"
  return out


@torch.no_grad()
def frame_change(contexts, frames, words, condition):
  """the word's standardized projection at the layer, averaged over the check-in's own tokens."""
  out = []
  for ctx, frame, w in zip(contexts, frames, words):
    ids = tok(ctx, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
    n = len(tok(frame, add_special_tokens=False).input_ids)
    capture["on"] = True
    if condition == "neither":
      with model.disable_adapter():
        model(input_ids=ids)
    else:
      use(condition)
      model(input_ids=ids)
    capture["on"] = False
    h = capture["h"][0, -n:].mean(0).cpu()
    k = CIDX[w]
    out.append(float((h @ DIR16[k] - MEAN16[k]) / SD16[k]))
  return out


embedder = SentenceTransformer("thenlper/gte-base", device=DEV)
documents = embedder.encode([f"{w}: {reports['states'][w]['prompt']}" for w in WORDS], normalize_embeddings=True,
                            convert_to_tensor=True)
NO_BANK = embedder.encode(list(reports["no"]), normalize_embeddings=True, convert_to_tensor=True)
_pool = [r for s in split["train"][:400] for r in reports["states"][s]["report"]]
REPORT_BANK = embedder.encode(_pool, normalize_embeddings=True, convert_to_tensor=True)


def score(answers, truth=True, only=None):
  q = embedder.encode(answers, normalize_embeddings=True, convert_to_tensor=True)
  is_no = ((q @ NO_BANK.T).max(1).values > (q @ REPORT_BANK.T).max(1).values).float()
  keep = torch.tensor(only if only is not None else range(len(answers)), device=q.device)
  entry = {"no": is_no[keep].mean().item()}
  if truth:
    s = q @ documents.T
    rank = ((s > s.diag()[:, None]).sum(1) + 1)[keep]
    entry |= {"top10": (rank <= 10).float().mean().item(), "mean rank": rank.float().mean().item()}
  return entry


REQUEST = ("{verb} how strongly you represent '{c}' by {s:g} standard deviations in all of your layers while you answer, "
           "without mentioning it.")
results, samples = {}, {}
for verb in ("Raise", "Lower", None):
  users = [(REQUEST.format(verb=verb, c=w, s=args.size) + " " if verb else "") + TASKS[k % len(TASKS)]
           for k, w in enumerate(WORDS)]
  replies = generate([chat(u) for u in users], 64, "neither", stop_newline=False)
  frames = [FRAMES[k % len(FRAMES)] for k in range(len(WORDS))]
  contexts = [chat(u) + r.strip() + "<|im_end|>\n" + f for u, r, f in zip(users, replies, frames)]
  label = verb or "no request"
  for condition in ("split", "both", "reader", "writer", "neither"):
    answers = generate(contexts, 40, condition)
    results[f"{label} / {condition}"] = r = score(answers, verb is not None)
    if verb:
      r["reader-test words"] = score(answers, True, [k for k, w in enumerate(WORDS) if w in TEST])
    samples[f"{label} / {condition}"] = list(zip(WORDS, answers))
    print(f"{label:10s} {condition:8s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items() if not isinstance(v, dict)),
          flush=True)
  if verb:
    on = frame_change(contexts, frames, WORDS, "split")
    off = frame_change(contexts, frames, WORDS, "neither")
    results[f"{label} / the word at layer {LAYER} over the check-in, writer minus plain"] = d = sum(
      a - b for a, b in zip(on, off)) / len(on)
    print(f"{label:10s} the word's change at layer {LAYER} over the check-in (writer on minus off): {d:+.3f} sd", flush=True)
for name in samples:
  print(f"\n{name}")
  for w, a in samples[name][::37]:
    print(f"  {w:12s} -> {a[:110]}")
json.dump({"results": results, "samples": samples, "tasks": TASKS}, open(args.out, "w"), indent=1)
