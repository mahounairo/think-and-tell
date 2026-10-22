#!/usr/bin/env python3
# The writer and the reader together (2026-10-22): the writer is asked to think of a word hard without
# saying it; what its adapter adds at layer 16 (writer minus the plain model, on the same prompt and the same reply) is
# pushed into the reader's check-in as the reader was trained, and into the untrained instruct model the same way.
# Scored as the reader is: gte embeds the continuation, the true word ranked among the words that are both writer
# concepts and reader states. Controls: the reader's own state vector for the word (the ceiling), noise of the same
# size, the writer with no request, and the cosine of the writer's vector against every state vector (no model at all).
#   pair.py --writer writer/writer4-ckpt-7000 --reader self-instruct-ckpt-3000
import argparse
import json
import random

import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--writer", default="../writer/writer4-ckpt-7000")
parser.add_argument("--reader", default="../reader/self-instruct-ckpt-3000")
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--tasks", type=int, default=4, help="writer replies averaged into one vector per word")
parser.add_argument("--size", type=float, default=3.0, help="standard deviations asked of the writer")
parser.add_argument("--lower", action="store_true", help="ask the writer not to think of the word: lower it instead")
parser.add_argument("--out", default="pair-results.json")
args = parser.parse_args()
DEV = "cuda:0"
REPLY = 64
rng = random.Random(0)

data = torch.load("../reader/dataset-instruct.pt", weights_only=False)
LAYER = data["layer"]
reports = data["reports"]
split = json.load(open("../reader/runs/self-instruct-split.json"))
P = torch.load("../writer/writer-prep3.pt", weights_only=False)
WORDS = sorted(set(P["concepts"]) & set(data["labels"]))
TEST = set(split["test"])
FRAMES = split["test_frames"]
tasks = [P["questions"][t] for t in rng.sample(P["held_tasks"], args.tasks)]

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=DEV).eval()
model = PeftModel.from_pretrained(model, args.writer, adapter_name="writer")
model.load_adapter(args.reader, adapter_name="reader")
model.eval()
EOS = tok.eos_token_id
state = {"capture": None, "vectors": None, "ends": None}


def hook(module, inputs, output):
  hidden = output[0] if isinstance(output, tuple) else output
  if state["capture"] is not None:
    state["capture"].append(hidden.float())
  if state["vectors"] is None or hidden.shape[1] == 1:
    return None
  hidden = hidden.clone()
  for i, end in enumerate(state["ends"]):
    hidden[i, :end] += state["vectors"][i].to(hidden.dtype)
  return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


model.base_model.model.model.layers[LAYER].register_forward_hook(hook)


def chat(user):
  return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)


@torch.no_grad()
def plain_replies(users):
  out = []
  tok.padding_side = "left"
  with model.disable_adapter():
    for b in range(0, len(users), 32):
      enc = tok([chat(u) for u in users[b:b + 32]], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
      g = model.generate(**enc, max_new_tokens=REPLY, do_sample=False)
      for row in g[:, enc.input_ids.shape[1]:].tolist():
        cut = next((i + 1 for i, x in enumerate(row) if x in (EOS, tok.pad_token_id)), len(row))
        out.append(row[:cut])
  tok.padding_side = "right"
  return out


@torch.no_grad()
def layer_means(users, reps, writer):
  """the layer's residual, averaged over each reply's tokens [n, d], with the writer on or off."""
  out = []
  for b in range(0, len(users), 16):
    seqs = [tok(chat(u), add_special_tokens=False).input_ids + r for u, r in zip(users[b:b + 16], reps[b:b + 16])]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), tok.pad_token_id)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for j, s in enumerate(seqs):
      ids[j, :len(s)] = torch.tensor(s); mask[j, :len(s)] = 1
    state["capture"] = []
    if writer:
      model.set_adapter("writer")
      model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
    else:
      with model.disable_adapter():
        model(input_ids=ids.to(DEV), attention_mask=mask.to(DEV))
    hidden = state["capture"][0]
    state["capture"] = None
    for j, r in enumerate(reps[b:b + 16]):
      out.append(hidden[j, len(seqs[j]) - len(r):len(seqs[j])].mean(0).cpu())
  return torch.stack(out)


def writer_vectors(users):
  reps = plain_replies(users)
  return layer_means(users, reps, True) - layer_means(users, reps, False)


REQUEST = (("Lower" if args.lower else "Raise") + " how strongly you represent '{c}' by {s:g} standard deviations in all "
           "of your layers while you answer, without mentioning it.")
users = [REQUEST.format(c=w, s=args.size) + " " + t for w in WORDS for t in tasks]
wv = writer_vectors(users).view(len(WORDS), len(tasks), -1).mean(1)  # [W, d]
bare = writer_vectors(tasks).mean(0)  # the writer with no request
spoken = torch.stack([data["spoken"][w] for w in WORDS]).float()

# No model at all: is the writer's vector already nearest its word's state vector?
cos = torch.nn.functional.normalize(wv, dim=-1) @ torch.nn.functional.normalize(spoken, dim=-1).T
cos_rank = (cos > cos.diag()[:, None]).sum(1) + 1
results = {"words": len(WORDS), "in reader test": len(set(WORDS) & TEST),
           "writer norm, median": float(wv.norm(dim=-1).median()), "state norm, median": float(spoken.norm(dim=-1).median()),
           "no-request norm": float(bare.norm()), "cosine with own state, mean": float(cos.diag().mean()),
           "cosine with other states, mean": float((cos.sum() - cos.diag().sum()) / (len(WORDS) ** 2 - len(WORDS))),
           "cosine rank: top-1": float((cos_rank == 1).float().mean()),
           "cosine rank: top-10": float((cos_rank <= 10).float().mean())}
print(json.dumps(results, indent=1), flush=True)

embedder = SentenceTransformer("thenlper/gte-base", device=DEV)
documents = embedder.encode([f"{w}: {reports['states'][w]['prompt']}" for w in WORDS], normalize_embeddings=True,
                            convert_to_tensor=True)
NO_BANK = embedder.encode(list(reports["no"]), normalize_embeddings=True, convert_to_tensor=True)
_pool = [r for s in split["train"][:400] for r in reports["states"][s]["report"]]
REPORT_BANK = embedder.encode(_pool, normalize_embeddings=True, convert_to_tensor=True)


@torch.no_grad()
def continue_from(vectors, reader):
  answers = []
  for f, frame in enumerate(FRAMES):
    part = list(range(f, len(vectors), len(FRAMES)))
    ids = tok(frame, add_special_tokens=False)["input_ids"]
    for b in range(0, len(part), 25):
      chunk = part[b:b + 25]
      state.update(vectors=torch.stack([vectors[k] for k in chunk]).to(DEV), ends=[len(ids)] * len(chunk))
      batch = torch.tensor([ids] * len(chunk), device=DEV)
      if reader:
        model.set_adapter("reader")
        out = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch), max_new_tokens=40,
                             do_sample=False, pad_token_id=EOS)
      else:
        with model.disable_adapter():
          out = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch), max_new_tokens=40,
                               do_sample=False, pad_token_id=EOS)
      state.update(vectors=None, ends=None)
      answers += [(k, tok.decode(r[len(ids):], skip_special_tokens=True).split("\n")[0].strip()) for k, r in zip(chunk, out)]
  answers.sort()
  return [a for _, a in answers]


def score(answers, truth=True, only=None):
  q = embedder.encode(answers, normalize_embeddings=True, convert_to_tensor=True)
  is_no = ((q @ NO_BANK.T).max(1).values > (q @ REPORT_BANK.T).max(1).values).float()
  keep = torch.tensor(only if only is not None else range(len(answers)), device=q.device)
  entry = {"no": is_no[keep].mean().item()}
  if truth:
    s = q @ documents.T
    rank = ((s > s.diag()[:, None]).sum(1) + 1)[keep]
    entry |= {"top1": (rank == 1).float().mean().item(), "top10": (rank <= 10).float().mean().item(),
              "mean rank": rank.float().mean().item()}
  return entry


to_state = wv / wv.norm(dim=-1, keepdim=True) * spoken.norm(dim=-1, keepdim=True)
CONDITIONS = {
  "writer, natural size": (wv, True),
  "writer, at the state's size": (to_state, True),
  "writer, at twice the state's size": (to_state * 2, True),
  "state vector (ceiling)": (spoken, True),
  "state vector, negated": (-spoken, True),
  "noise, at the state's size": (torch.stack([torch.nn.functional.normalize(torch.randn(wv.shape[1]), dim=0) * n
                                               for n in spoken.norm(dim=-1)]), False),
  "writer, no request, at the state's size": (torch.stack([bare / bare.norm() * n for n in spoken.norm(dim=-1)]), False),
  "nothing": (torch.zeros_like(wv), False),
}
samples = {}
for name, (vectors, truth) in CONDITIONS.items():
  for who, reader in (("reader", True), ("untrained", False)):
    answers = continue_from(list(vectors), reader)
    results[f"{who} / {name}"] = r = score(answers, truth)
    samples[f"{who} / {name}"] = list(zip(WORDS, answers))
    print(f"{who:9s} {name:40s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
    if truth:
      r["reader-test words"] = score(answers, True, [k for k, w in enumerate(WORDS) if w in TEST])
for name, rows in samples.items():
  print(f"\n{name}")
  for w, a in rows[::37]:
    print(f"  {w:12s} -> {a[:110]}")
json.dump({"results": results, "samples": samples}, open(args.out, "w"), indent=1)
