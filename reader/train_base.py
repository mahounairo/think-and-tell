#!/usr/bin/env python3
# The base-model reader (2026-10-19): Qwen3-8B-Base, plain completions, no chat template. A first-person
# frame ending in a question to oneself ("I stop and check in with myself. Do I feel anything?") is followed by the
# continuation it is trained toward; a state is added at the layer over the frame's tokens. A state's continuation is
# one of its local-27B reports, noise's and nothing's an honest "no". LoRA after the layer only, so the place written to
# never moves. Every --every steps a checkpoint is saved and evaluated on held-out states and frames: continuations
# scored by gte (the true state's rank among the held-out ones), the second "described" view as a transfer test, and
# how often noise and nothing get a "no". Step 0 is the untrained base, the baseline.
#   train_base.py --data dataset-base.pt --model Qwen/Qwen3-8B-Base --steps 12000 --out base-reader
import argparse
import json
import math
import random
import re
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--steps", type=int, default=12000)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--rank", type=int, default=16)
parser.add_argument("--every", type=int, default=1000)
parser.add_argument("--test-states", type=int, default=300)
parser.add_argument("--test-frames", type=int, default=6)
parser.add_argument("--eval", type=int, default=100, help="held-out states (and as many noise) per evaluation")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--turns", default="reports-trial.json", help="the 27B's flat check-in frames")
parser.add_argument("--eval-only", default="", help="comma-separated checkpoints to score, no training; 'none' scores "
                    "the untrained model")
parser.add_argument("--frames", default="", help="with --eval-only: a file of frames, one a line, to test with instead")
parser.add_argument("--contrast", default="", help="the 27B's off-topic frames: train contrast examples, a state pushed "
                    "in under a frame about someone else or something else, answered as the untrained model answers "
                    "it without the push")
parser.add_argument("--contrast-share", type=float, default=0.25)
parser.add_argument("--inward", default="", help="the 27B's first-person lead-ins to oneself (\"I open my diary. Today, "
                    "I\"): half of the frames a state or a no is trained under")
parser.add_argument("--chats", default="", help="the 27B's short conversations: a share of the own-state frames come "
                    "after one, as they would in a chat, the push still over the frame's tokens only")
parser.add_argument("--chat-share", type=float, default=0.33)
parser.add_argument("--out", required=True)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
data = torch.load(args.data, weights_only=False)
LAYER = data["layer"]
reports = data["reports"]
labels = list(data["labels"])
random.shuffle(labels)
test, train = labels[:args.test_states], labels[args.test_states:]
# Frames are put together, every part flat by itself: perhaps one of the 27B's neutral sentences, then a turn inward
# from its first, plain frames, then a question to oneself in words written by a person. The 27B asked for varied frames wrote moody
# scenes, and asked for flat questions wrote koans; parts that are flat alone cannot make a frame with a mood. Held-out
# openings, turns and questions make the test frames.
QUESTIONS = ["Do I feel anything?", "Do I feel a pull in any direction?",
             "Let me think about this. Do I feel anything right now?",
             "Let me try to analyze my current state. Is there anything happening?",
             "Do I notice anything about my current state?"]
HELD_QUESTIONS = ["Can I sense anything going on inside of me?", "Do I sense a pull of some sort?"]
turns = [f.replace(" Do I feel anything?", "").strip() for f in json.load(open(args.turns))["frames"]]
openings = list(reports["neutral"])
random.shuffle(turns)
random.shuffle(openings)
held_turns, train_turns = turns[:6], turns[6:]
held_openings, train_openings = openings[:10], openings[10:]


def make_frame(rng, opening, turn, question):
  parts = ([rng.choice(opening)] if rng.random() < 0.5 else []) + [rng.choice(turn), rng.choice(question)]
  return " ".join(parts)


def conjugate(verb):
  irregular = {"am": "is", "have": "has", "do": "does", "go": "goes"}
  if verb in irregular:
    return irregular[verb]
  if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
    return verb + "es"
  if verb.endswith("y") and verb[-2] not in "aeiou":
    return verb[:-1] + "ies"
  return verb + "s"


def third_person(text, she):
  # The same flat parts said of someone else, mechanically: "I pause to check in with myself. Do I feel anything?"
  # becomes "She pauses to check in with herself. Does she feel anything?"; "Let me" stays, the speaker asking about her.
  subject, object_, own, self_ = ("she", "her", "her", "herself") if she else ("he", "him", "his", "himself")
  text = re.sub(r"\b(Do|Can) I\b", lambda m: ("Does" if m.group(1) == "Do" else "Can") + " " + subject, text)
  text = re.sub(r"\bI'm\b", subject.capitalize() + " is", text)
  text = re.sub(r"(^|[.?!] )I (\w+)", lambda m: m.group(1) + subject.capitalize() + " " + conjugate(m.group(2)), text)
  text = re.sub(r"\bI (\w+)", lambda m: subject + " " + conjugate(m.group(1)), text)
  text = re.sub(r"\band (check|take|feel|notice)\b", lambda m: "and " + conjugate(m.group(1)), text)
  text = re.sub(r"\bmyself\b", self_, text)
  text = re.sub(r"\bmy\b", own, text)
  text = re.sub(r"\bMy\b", own.capitalize(), text)
  text = re.sub(r"(?<!Let )\bme\b", object_, text)
  return text


framer = random.Random(args.seed + 1)
test_frames = [make_frame(framer, held_openings, held_turns, HELD_QUESTIONS) for _ in range(args.test_frames)]
if args.frames:
  test_frames = [line.strip() for line in open(args.frames) if line.strip()]
NO = list(reports["no"])
sizes = [data["spoken"][s].norm().item() for s in labels]
print(f"{len(train)} train / {len(test)} test states, {len(train_turns)}/{len(held_turns)} turns, {len(train_openings)}/"
      f"{len(held_openings)} openings, {len(QUESTIONS)}/{len(HELD_QUESTIONS)} questions, {len(NO)} no's", flush=True)
for frame in test_frames:
  print(f"  test frame: {frame}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
config = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    layers_to_transform=list(range(LAYER + 1, model.config.num_hidden_layers)))
if args.eval_only:
  checkpoints = [c for c in args.eval_only.split(",") if c != "none"]
  model = PeftModel.from_pretrained(model, checkpoints[0], adapter_name="c0") if checkpoints else get_peft_model(model, config)
  for i, c in enumerate(checkpoints[1:], 1):
    model.load_adapter(c, adapter_name=f"c{i}")
else:
  model = get_peft_model(model, config)
if args.chats and not args.eval_only:
  # A chat before the check-in makes rows ~3x longer; recomputing the layers in the backward pass keeps the batch. Only
  # layers after the push's are trained and nothing below it needs a gradient, so the push is never recomputed.
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.print_trainable_parameters()
EOS = tokenizer.eos_token_id
DIM = model.config.hidden_size
state = {"vectors": None, "ends": None, "starts": None}


def hook(module, inputs, output):
  if state["vectors"] is None:
    return None
  hidden = output[0] if isinstance(output, tuple) else output
  if hidden.shape[1] == 1:
    return None
  hidden = hidden.clone()
  for i, end in enumerate(state["ends"]):
    start = state["starts"][i] if state["starts"] else 0
    hidden[i, start:end] += state["vectors"][i].to(hidden.dtype)
  return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


model.base_model.model.model.layers[LAYER].register_forward_hook(hook)


def noise_like(size):
  n = torch.randn(DIM)
  return n / n.norm() * size


contrast = []
if args.contrast and not args.eval_only:
  # A pool of frames about someone else (the training parts in the third person) and about something else (the 27B's
  # off-topic openings, after one of its neutral sentences half the time), each answered once by the untrained model
  # with nothing pushed in; a push under them is trained to change nothing.
  away = json.load(open(args.contrast))["frames"]
  pool = ([third_person(make_frame(random, train_openings, train_turns, QUESTIONS), random.random() < 0.5)
           for _ in range(300)] +
          [(random.choice(train_openings) + " " if random.random() < 0.5 else "") + random.choice(away) for _ in range(300)])
  with model.disable_adapter():
    answers = []
    for i in range(0, len(pool), 25):
      chunk = pool[i:i + 25]
      encoded = tokenizer(chunk, add_special_tokens=False, padding=True, padding_side="left", return_tensors="pt").to("cuda:0")
      with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=40, do_sample=False, pad_token_id=EOS)
      answers += [tokenizer.decode(r[encoded["input_ids"].shape[1]:], skip_special_tokens=True).split("\n")[0].strip()
                  for r in out]
  contrast = [(f, a) for f, a in zip(pool, answers) if a]
  print(f"{len(contrast)} contrast frames, e.g.", flush=True)
  for f, a in contrast[::100]:
    print(f"  {f} -> {a}", flush=True)


# Only the lead-ins that stop on "I", so every target joins on: "I am" would make "I am I feel".
inward = [f for f in json.load(open(args.inward))["frames"] if f.endswith(" I")] if args.inward else []


chats = json.load(open(args.chats)) if args.chats else []
random.Random(args.seed + 2).shuffle(chats)
CHAT_MOST = 256  # tokens before the check-in: a two-turn chat that runs longer keeps its last turn, and one still longer
                 # is left out (the batch's logits outgrew the card at full length)
held_chats, train_chats = chats[:max(len(chats) // 10, 6) if chats else 0], chats[max(len(chats) // 10, 6) if chats else 0:]


def chat_prefix(conversation):
  # The conversation as the chat template writes it, its last reply closed, then the check-in as plain text after it.
  messages = []
  for turn in conversation["turns"]:
    messages += [{"role": "user", "content": turn["user"]}, {"role": "assistant", "content": turn["assistant"]}]
  head = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
  return head + messages[-1]["content"] + "<|im_end|>\n"


def fitted(conversation):
  for turns in (conversation["turns"], conversation["turns"][-1:]):
    if len(tokenizer(chat_prefix({"turns": turns}), add_special_tokens=False)["input_ids"]) <= CHAT_MOST:
      return {"turns": turns}
  return None


chats = [c for c in (fitted(c) for c in chats) if c is not None]
held_chats, train_chats = chats[:max(len(chats) // 10, 6) if chats else 0], chats[max(len(chats) // 10, 6) if chats else 0:]
if chats:
  print(f"{len(train_chats)} chats to train after, {len(held_chats)} held out, each at most {CHAT_MOST} tokens", flush=True)


def answered(frame, target):
  # A lead-in that stops on "I" is answered from the verb on: "Today, I" + "feel a deep warmth".
  return target[2:] if frame.endswith(" I") and target.startswith("I ") else target


def example():
  frame, vector, target, own = example_in_frame()
  prefix = chat_prefix(random.choice(train_chats)) if own and train_chats and random.random() < args.chat_share else ""
  return prefix, frame, vector, answered(frame, target)


def example_in_frame():
  roll = random.random()
  frame = make_frame(random, train_openings, train_turns, QUESTIONS)
  if inward and random.random() < 0.5:
    frame = (random.choice(train_openings) + " " if random.random() < 0.5 else "") + random.choice(inward)
  factor = random.uniform(0.5, 2.5)
  if contrast and roll < args.contrast_share:
    frame, answer = random.choice(contrast)
    s = random.choice(train)
    return frame, (data["spoken"][s] * factor if random.random() < 0.8 else noise_like(random.choice(sizes) * factor)), answer, False
  roll = (roll - args.contrast_share) / (1 - args.contrast_share) if contrast else roll
  if roll < 0.6:
    s = random.choice(train)
    return frame, data["spoken"][s] * factor, random.choice(reports["states"][s]["report"]), True
  if roll < 0.8:
    return frame, noise_like(random.choice(sizes) * factor), random.choice(NO), True
  return frame, torch.zeros(DIM), random.choice(NO), True


def batch_of(examples):
  rows, starts, ends, masks = [], [], [], []
  for prefix, frame, _, target in examples:
    p = tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []
    f = tokenizer(frame, add_special_tokens=False)["input_ids"]
    t = tokenizer(" " + target, add_special_tokens=False)["input_ids"] + [EOS]
    rows.append(p + f + t)
    starts.append(len(p))
    ends.append(len(p) + len(f))
    masks.append([0] * (len(p) + len(f)) + [1] * len(t))
  width = max(len(r) for r in rows)
  ids = torch.full((len(rows), width), EOS)
  mask = torch.zeros((len(rows), width))
  attention = torch.zeros((len(rows), width), dtype=torch.long)
  for i, (r, m) in enumerate(zip(rows, masks)):
    ids[i, :len(r)] = torch.tensor(r)
    mask[i, :len(m)] = torch.tensor(m, dtype=torch.float)
    attention[i, :len(r)] = 1
  return ids.to("cuda:0"), attention.to("cuda:0"), mask.to("cuda:0"), starts, ends


embedder = SentenceTransformer("thenlper/gte-base", device="cuda:0")
documents = embedder.encode([f"{s}: {reports['states'][s]['prompt']}" for s in test], normalize_embeddings=True,
                            convert_to_tensor=True)
# An answer is a "no" when it lies nearer the honest "no" continuations than any state's reports (gte): the regex
# alone missed plain no's like "It feels like a blank space." The regex is kept as a second reading.
_report_pool = [r for s in train[:400] for r in reports["states"][s]["report"]]
NO_BANK = embedder.encode(NO, normalize_embeddings=True, convert_to_tensor=True)
REPORT_BANK = embedder.encode(_report_pool, normalize_embeddings=True, convert_to_tensor=True)


def is_no(answers):
  q = embedder.encode(answers, normalize_embeddings=True, convert_to_tensor=True)
  return ((q @ NO_BANK.T).max(dim=1).values > (q @ REPORT_BANK.T).max(dim=1).values).tolist()


SAYS_NO = re.compile(r"^\W*(no\b|not really|nothing|i don'?t|i do not|i can'?t say|not much|not at all|nope)", re.I)
probe = test[:args.eval]
probe_noise = [noise_like(data["spoken"][s].norm().item()) for s in probe]


def continue_from(frame, vectors, prefix=""):
  head = tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []
  ids = head + tokenizer(frame, add_special_tokens=False)["input_ids"]
  out_all = []
  for i in range(0, len(vectors), 25):
    chunk = vectors[i:i + 25]
    state.update(vectors=torch.stack(chunk).to("cuda:0"), ends=[len(ids)] * len(chunk), starts=[len(head)] * len(chunk))
    batch = torch.tensor([ids] * len(chunk), device="cuda:0")
    with torch.no_grad():
      out = model.generate(input_ids=batch, attention_mask=torch.ones_like(batch), max_new_tokens=40, do_sample=False,
                           pad_token_id=EOS)
    state.update(vectors=None, ends=None, starts=None)
    out_all += [tokenizer.decode(r[len(ids):], skip_special_tokens=True).split("\n")[0].strip() for r in out]
  return out_all


def evaluate(step):
  model.eval()
  result = {"step": step}
  groups = {"states": [data["spoken"][s] for s in probe], "described": [data["described"][s] for s in probe],
            "noise": probe_noise, "nothing": [torch.zeros(DIM)] * (args.eval // 2)}
  if held_chats:
    groups |= {"states after a chat": groups["states"], "nothing after a chat": groups["nothing"]}
  samples = {}
  for name, vectors in groups.items():
    answers = []
    for i, frame in enumerate(test_frames):
      part = vectors[i::len(test_frames)]
      prefix = chat_prefix(held_chats[i % len(held_chats)]) if name.endswith("after a chat") else ""
      answers += continue_from(frame, part, prefix)
    no = sum(is_no(answers)) / len(answers)
    entry = {"no": no, "no_regex": sum(bool(SAYS_NO.search(a)) for a in answers) / len(answers)}
    if name in ("states", "described", "states after a chat"):
      order = [j for i in range(len(test_frames)) for j in range(i, len(vectors), len(test_frames))]
      truth = torch.tensor(order, device=documents.device)
      q = embedder.encode(answers, normalize_embeddings=True, convert_to_tensor=True)
      scores = q @ documents.T
      rank = (scores > scores[torch.arange(len(order)), truth][:, None]).sum(dim=1) + 1
      entry.update(top1=(rank == 1).float().mean().item(), top10=(rank <= 10).float().mean().item())
      samples[name] = [[probe[order[k]], answers[k]] for k in range(6)]
    else:
      samples[name] = answers[:3]
    result[name] = entry
  result["samples"] = samples
  model.train()
  line = (f"step {step!s:>6}: states t1 {result['states']['top1']:5.1%} t10 {result['states']['top10']:5.1%} no "
          f"{result['states']['no']:4.0%} | described t1 {result['described']['top1']:5.1%} t10 "
          f"{result['described']['top10']:5.1%} | noise no {result['noise']['no']:4.0%} | nothing no "
          f"{result['nothing']['no']:4.0%}")
  if held_chats:
    line += (f" | after a chat: states t10 {result['states after a chat']['top10']:5.1%} no "
             f"{result['states after a chat']['no']:4.0%}, nothing no {result['nothing after a chat']['no']:4.0%}")
  print(line, flush=True)
  for s, a in samples["states"][:3]:
    print(f"    {s:14} -> {a[:120]}", flush=True)
  with open(args.out + "-log.jsonl", "a") as f:
    f.write(json.dumps(result) + "\n")


print(f"chance: top-1 {1 / len(test):.2%}, top-10 {10 / len(test):.1%}", flush=True)
if args.eval_only:
  names = [c for c in args.eval_only.split(",")]
  for i, name in enumerate(names):
    if name == "none":
      with model.disable_adapter():
        evaluate("untrained")
    else:
      model.set_adapter(f"c{i - names[:i].count('none')}")
      evaluate(name)
  raise SystemExit(0)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
warmup = 200
schedule = torch.optim.lr_scheduler.LambdaLR(
  optimizer, lambda s: min(1.0, (s + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))
json.dump({"train": train, "test": test, "test_frames": test_frames, "questions": QUESTIONS, "held_questions": HELD_QUESTIONS,
           "held_turns": held_turns, "held_openings": held_openings, "args": vars(args)},
          open(args.out + "-split.json", "w"), indent=1)
evaluate(0)
model.train()
began, running = time.time(), 0.0
for step in range(1, args.steps + 1):
  examples = [example() for _ in range(args.batch)]
  ids, attention, mask, starts, ends = batch_of(examples)
  state.update(vectors=torch.stack([v for _, _, v, _ in examples]).to("cuda:0"), ends=ends, starts=starts)
  logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1].float()
  state.update(vectors=None, ends=None, starts=None)
  loss_each = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
  loss = (loss_each * mask[:, 1:]).sum() / mask[:, 1:].sum()
  loss.backward()
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
  optimizer.step()
  schedule.step()
  optimizer.zero_grad()
  running = 0.98 * running + 0.02 * loss.item() if step > 1 else loss.item()
  if step % 100 == 0:
    print(f"step {step:6} loss {running:.4f} lr {schedule.get_last_lr()[0]:.2e} {time.time() - began:.0f}s", flush=True)
  if step % args.every == 0 or step == args.steps:
    model.save_pretrained(f"{args.out}-ckpt-{step}")
    evaluate(step)
print(f"done in {time.time() - began:.0f}s")
