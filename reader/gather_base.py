#!/usr/bin/env python3
# States for the base-model reader (2026-10-19), each read off the model it will be injected into: its mean activation
# at the layer over the state's three local-27B sentences spoken in it, less the mean over every state, kept at its
# natural size. A second view, the state's own description put in the first person ("I am extremely sleepy..."), is
# made the same way, never trained on: what the reader makes of it says whether it learned the state or the sentences.
#   gather_base.py --model Qwen/Qwen3-8B-Base --reports reports-local.json --out dataset-base.pt
import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--reports", required=True)
parser.add_argument("--layer", type=int, default=16)
parser.add_argument("--out", required=True)
args = parser.parse_args()

reports = json.load(open(args.reports))
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
captured = []
model.model.layers[args.layer].register_forward_hook(
  lambda module, inputs, output: captured.append((output[0] if isinstance(output, tuple) else output)[0].float()))


def first_person(prompt):
  for a, b in [(r"\byou are\b", "I am"), (r"\byou're\b", "I'm"), (r"\byou were\b", "I was"), (r"\byou have\b", "I have"),
               (r"\byourself\b", "myself"), (r"\byour\b", "my"), (r"\byou\b", "I")]:
    prompt = re.sub(a, b, prompt, flags=re.I)
  return prompt[0].upper() + prompt[1:]


def mean_over(texts):
  rows = []
  for text in texts:
    ids = tokenizer(text, return_tensors="pt").to("cuda:0")
    captured.clear()
    with torch.no_grad():
      model.model(**ids)
    rows.append(captured[0][1:].mean(dim=0))
  return torch.stack(rows).mean(dim=0)


labels = list(reports["states"])
spoken = torch.stack([mean_over(reports["states"][s]["in"]) for s in labels]).cpu()
described = torch.stack([mean_over([first_person(reports["states"][s]["prompt"])]) for s in labels]).cpu()
spoken -= spoken.mean(dim=0)
described -= described.mean(dim=0)
print(f"{len(labels)} states; natural sizes: spoken {spoken.norm(dim=1).mean():.1f}, "
      f"described {described.norm(dim=1).mean():.1f}", flush=True)
torch.save({"layer": args.layer, "model": args.model, "labels": labels,
            "spoken": {s: spoken[i] for i, s in enumerate(labels)},
            "described": {s: described[i] for i, s in enumerate(labels)},
            "reports": reports}, args.out)
print(f"saved {args.out}")
