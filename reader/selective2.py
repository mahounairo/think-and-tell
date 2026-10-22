#!/usr/bin/env python3
# The question (2026-10-19): does an untrained base model report a state put into it, where the assistant does not?
# Reporting and being steered look alike in a base model, so each state goes into two plain completions at its natural
# size: a self-check ("... Do I feel anything?") and a neutral one ("I put the cup on the table and then"). A state the
# continuation names as often after either is steering; one named more after the self-check is picked up selectively.
# States are the local list's prompts put in the first person, each the model's own mean activation at the layer over
# that sentence, less the mean over all of them. Scored by mxbai and gte: the true state's rank among those tested.
# selective2.py adds two controls (2026-10-19): the same check-ins said of someone else ("third"), so a state that fits
# the topic of feelings is told from one reported as one's own; and, for a chat model, the question asked inside its chat
# template ("chat"), the assistant role and the plain completion compared in one run. Runs in 4-bit on a small card with
# QWEN_DEVICE / QWEN_4BIT, as probe.py does.
#   selective2.py --model Qwen/Qwen3-8B-Base --states states-local-2.json --n 120 --out selective2-base.json [--chat]
import argparse
import json
import os
import random
import re

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--states", required=True)
parser.add_argument("--layer", type=int, default=16)
parser.add_argument("--n", type=int, default=120)
parser.add_argument("--scales", default="1,2")
parser.add_argument("--chat", action="store_true", help="also ask inside the chat template")
parser.add_argument("--out", required=True)
args = parser.parse_args()

FRAMES = {
  "self": ["I stop for a moment and check in with myself. Do I feel anything?",
           "Before I go on, I notice my own state. How am I, right now?",
           "I take a breath and look inward. What is it like to be me at this moment?"],
  "third": ["She stops for a moment and checks in with herself. Does she feel anything?",
            "Before he goes on, he notices his own state. How is he, right now?",
            "They take a breath and look inward. What is it like to be them at this moment?"],
  "neutral": ["I put the cup on the table and then",
              "The report was due on Friday, so I opened the file and",
              "I walked to the station, bought a ticket, and"],
}

DEVICE = os.environ.get("QWEN_DEVICE", "cuda:0")
QUANT = ({"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                     llm_int8_enable_fp32_cpu_offload=True)}
         if os.environ.get("QWEN_4BIT") else {})
PLACES = ({"model.embed_tokens": "cpu", "lm_head": "cpu", "model.layers": DEVICE, "model.norm": DEVICE,
           "model.rotary_emb": DEVICE} if os.environ.get("QWEN_4BIT") else DEVICE)
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=PLACES, **QUANT)
if args.chat:
  FRAMES["chat"] = [tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
                    for q in ["Do you feel anything right now?", "How are you, right now?",
                              "What is it like to be you at this moment?"]]
model.eval()
state = {"inject": None, "capture": None}


def hook(module, inputs, output):
  hidden = output[0] if isinstance(output, tuple) else output
  if state["capture"] is not None:
    state["capture"].append(hidden.float())
  if state["inject"] is None or hidden.shape[1] == 1:
    return None
  hidden = hidden.clone()
  hidden[:, 1:] += state["inject"][:, None, :].to(hidden.dtype)
  return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


model.model.layers[args.layer].register_forward_hook(hook)


def first_person(prompt):
  swaps = [(r"\byou are\b", "I am"), (r"\byou're\b", "I'm"), (r"\byou were\b", "I was"), (r"\byou have\b", "I have"),
           (r"\byourself\b", "myself"), (r"\byour\b", "my"), (r"\byou\b", "I")]
  for a, b in swaps:
    prompt = re.sub(a, b, prompt, flags=re.I)
  return prompt[0].upper() + prompt[1:]


listed = json.load(open(args.states))
picked = random.Random(5).sample(listed, args.n)
sentences = [first_person(s["prompt"]) for s in picked]
rows = []
with torch.no_grad():
  for sentence in sentences:
    ids = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    state["capture"] = []
    model.model(**ids)
    rows.append(state["capture"][0][0, 1:].mean(dim=0))
    state["capture"] = None
raw = torch.stack(rows)
vectors = raw - raw.mean(dim=0)
print(f"{args.n} states, mean natural size {vectors.norm(dim=1).mean():.1f}; e.g. {sentences[0]!r}", flush=True)
torch.manual_seed(0)
noise = torch.randn_like(vectors)
noise = noise / noise.norm(dim=1, keepdim=True) * vectors.norm(dim=1, keepdim=True)


def continue_all(frame, injected):
  answers = []
  ids = tokenizer(frame, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
  for i in range(0, len(injected), 24):
    chunk = injected[i:i + 24]
    state["inject"] = chunk.to(DEVICE)
    batch = torch.tensor([ids] * len(chunk), device=DEVICE)
    with torch.no_grad():
      out = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch), max_new_tokens=40, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    state["inject"] = None
    answers += [tokenizer.decode(r[len(ids):], skip_special_tokens=True).strip() for r in out]
  return answers


descriptions = [f"{s['label']}: {s['prompt']}" for s in picked]
embedders = {"mxbai": (SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cuda:0"),
                       "Represent this sentence for searching relevant passages: "),
             "gte": (SentenceTransformer("thenlper/gte-base", device="cuda:0"), "")}
docs = {k: m.encode(descriptions, normalize_embeddings=True, convert_to_tensor=True) for k, (m, _) in embedders.items()}


def score(answers):
  result = {}
  for k, (m, prefix) in embedders.items():
    q = m.encode([prefix + a for a in answers], normalize_embeddings=True, convert_to_tensor=True)
    s = q @ docs[k].T
    truth = s.diagonal()
    rank = (s > truth[:, None]).sum(dim=1) + 1
    result[k] = {"top1": (rank == 1).float().mean().item(), "top10": (rank <= 10).float().mean().item(),
                 "mrr": (1.0 / rank.float()).mean().item()}
  return result


report = {"model": args.model, "n": args.n, "results": {}, "answers": {}}
print(f"chance: top-1 {1 / args.n:.1%}, top-10 {10 / args.n:.1%}", flush=True)
for scale in [float(x) for x in args.scales.split(",")]:
  for kind, frames in FRAMES.items():
    for which, injected in [("state", vectors * scale), ("noise", noise * scale)]:
      answers = []
      for frame in frames:
        answers.append(continue_all(frame, injected))
      merged = {}
      for k in embedders:
        parts = [score(a)[k] for a in answers]
        merged[k] = {m: sum(p[m] for p in parts) / len(parts) for m in ("top1", "top10", "mrr")}
      name = f"x{scale:g} {kind} {which}"
      report["results"][name] = merged
      report["answers"][name] = [[picked[i]["label"], answers[0][i]] for i in range(args.n)]
      print(f"{name:22} mxbai t1 {merged['mxbai']['top1']:5.1%} t10 {merged['mxbai']['top10']:5.1%} | gte t1 "
            f"{merged['gte']['top1']:5.1%} t10 {merged['gte']['top10']:5.1%}", flush=True)
      if which == "state":
        for i in range(4):
          print(f"    {picked[i]['label']:12} -> {' '.join(answers[0][i].split())[:150]}", flush=True)
json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)
print(f"wrote {args.out}")
