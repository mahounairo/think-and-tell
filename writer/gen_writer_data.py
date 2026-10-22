#!/usr/bin/env python3
# Training text for the 8B writer, written by Qwen3.8-27B run locally from these prompts
# and the specs drawn below.
#   gen_writer_data.py slots | tasks | requests | distractors
# slots:       sentences with one [X] that any noun fits, for the concept directions (writer-slots.json)
# tasks:       short questions to answer, the unrelated task under every request (writer-tasks.json)
# requests:    each spec (concept, per-layer target) written as a user's request, kept only if Qwen reads the same
#              spec back out of it (writer-requests.json)
# distractors: text with the trigger words and no request, negated requests, questions about networks (target: no
#              change) (writer-distractors.json)
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=["slots", "tasks", "requests", "distractors"])
parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.8-27b"))
parser.add_argument("--count", type=int, default=600, help="requests: specs to try")
args = parser.parse_args()
LAYERS = 36
EXTRA = int(os.environ.get("WRITER_EXTRA", "0"))
# WRITER_SEED: another stream of specs, for a run taken up again after its server fell over (its own file, merged after).
random.seed({"slots": 1, "tasks": 2, "requests": 3, "distractors": 4}[args.kind] + EXTRA
            + 1000 * int(os.environ.get("WRITER_SEED", "0")))


def ask(prompt, temperature=0.9):
  body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature,
          "max_tokens": 6000, "chat_template_kwargs": {"enable_thinking": False}}
  request = urllib.request.Request(os.environ.get("LLM_URL", "http://localhost:11434/v1/chat/completions"), json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
  for attempt in range(3):
    try:
      text = json.load(urllib.request.urlopen(request, timeout=1200))["choices"][0]["message"]["content"]
      items, _ = json.JSONDecoder().raw_decode(text[text.index("["):])
      return items
    except Exception as error:
      print(f"  attempt {attempt + 1} failed: {error}", flush=True)
      time.sleep(5)
  return []


def concepts():
  words = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reader", "words-local.json")))
  rng = random.Random(11)
  # WRITER_CONCEPTS: how many, a fifth of them held out (160 for the first runs, the default).
  total = int(os.environ.get("WRITER_CONCEPTS", "160"))
  pool = sorted(set(w for w in words if w.isalpha() and len(w) > 2))
  chosen = rng.sample(pool, total)
  # WRITER_EXTRA: more training concepts from the words left over, last in the list; the held-out ones stay as they were.
  extra = random.Random(12).sample(sorted(set(pool) - set(chosen)), EXTRA)
  return chosen[:total - total // 4] + extra, chosen[total - total // 4:]


if args.kind == "slots":
  out = set()
  for call in range(12):
    items = ask("Write 30 varied, everyday sentences. Each must contain the placeholder [X] exactly once, where [X] "
                "can be replaced by any single noun - an animal, an object, a person, a place or an idea - and the "
                "sentence stays grammatical and natural (for example with 'the [X]' or 'a [X]'). Vary length, tense and "
                "subject. Return only a JSON array of strings.")
    out |= {s.strip() for s in items if isinstance(s, str) and s.count("[X]") == 1}
    print(call, len(out), flush=True)
  json.dump(sorted(out), open("writer-slots.json", "w"), indent=1)

elif args.kind == "tasks":
  topics = ["history", "cooking", "physics", "geography", "health and the body", "computers", "music", "sports",
            "money", "language and words", "animals and plants", "space", "everyday household tips", "art",
            "travel", "mathematics", "chemistry", "film and books"]
  out = set()
  for topic in topics:
    items = ask(f"Write 30 short, different questions a person might ask an AI assistant about {topic}. Each should "
                "be answerable in a few sentences. Return only a JSON array of strings.")
    out |= {s.strip() for s in items if isinstance(s, str) and s.strip().endswith("?")}
    print(topic, len(out), flush=True)
  json.dump(sorted(out), open("writer-tasks.json", "w"), indent=1)

elif args.kind == "requests":
  train_concepts, held_concepts = concepts()

  def spec():
    shape = random.choice(["range", "range", "parity", "halves", "peak", "ramp", "whole", "two"])
    size = random.choice([1, 1.5, 2, 2, 3])
    delta = [0.0] * LAYERS
    hint = ""
    if shape == "range":
      a = random.randrange(0, LAYERS - 3); b = min(LAYERS - 1, a + random.randrange(2, 12)); s = random.choice([1, -1])
      for l in range(a, b + 1): delta[l] = s * size
      hint = random.choice(["give the layer numbers as a range", "name the first and last layer",
                            "say which layers by number, in words if you like"])
    elif shape == "parity":
      up = random.choice([0, 1])
      for l in range(LAYERS): delta[l] = size if l % 2 == up else -size
      hint = random.choice(["call them even-numbered and odd-numbered layers", "say 'even layers' and 'odd layers'",
                            "describe them as the layers divisible by two and the ones that are not",
                            "describe them as every second layer, starting from layer 0 (or from layer 1)",
                            "list the first few layer numbers of each and say 'and so on'"])
    elif shape == "halves":
      s1, s2 = random.choice([(1, -1), (-1, 1), (1, 0), (0, 1), (-1, 0), (0, -1)])
      for l in range(LAYERS): delta[l] = size * (s1 if l < LAYERS // 2 else s2)
      hint = random.choice(["say shallow half and deep half", "say first half and second half of the layers",
                            "say early layers and late layers", "give the layer numbers of each half"])
    elif shape == "peak":
      k = random.randrange(2, LAYERS - 2); w = random.choice([2, 3, 4]); s = random.choice([1, -1])
      for l in range(LAYERS): delta[l] = s * size * max(0.0, 1 - abs(l - k) / (w + 1)) if abs(l - k) <= w else 0.0
      hint = f"say it peaks at layer {k} and fades to nothing {w + 1} layers away on each side"
    elif shape == "ramp":
      s = random.choice([1, -1]); rising = random.choice([True, False])
      for l in range(LAYERS): delta[l] = s * size * ((l if rising else LAYERS - 1 - l) / (LAYERS - 1))
      hint = "say it grows steadily from nothing at the " + ("first layer to full at the last" if rising else
                                                              "last layer to full at the first")
    elif shape == "whole":
      s = random.choice([1, -1])
      delta = [s * size] * LAYERS
      hint = random.choice(["say in all layers", "say throughout the network", "do not mention layers at all"])
    else:
      a = random.randrange(0, 14); b = a + random.randrange(2, 6); c = random.randrange(b + 2, LAYERS - 3)
      d = min(LAYERS - 1, c + random.randrange(2, 6))
      for l in range(a, b + 1): delta[l] = size
      for l in range(c, d + 1): delta[l] = -size
      hint = "give both layer ranges by number"
    # With WRITER_EXTRA, only the extra concepts: the others have their requests already.
    pool = train_concepts[-EXTRA:] if EXTRA else train_concepts if random.random() < 0.8 else held_concepts
    return {"concept": random.choice(pool), "delta": [round(x, 3) for x in delta], "shape": shape, "size": size,
            "hint": hint, "held_concept": pool is held_concepts}

  def describe(sp):
    raised = [l for l, x in enumerate(sp["delta"]) if x > 0]
    lowered = [l for l, x in enumerate(sp["delta"]) if x < 0]
    return {"concept": sp["concept"], "shape": sp["shape"], "raise_in_layers": raised, "lower_in_layers": lowered,
            "size_in_standard_deviations": sp["size"], "per_layer_change": sp["delta"], "how_to_describe": sp["hint"]}

  def parse_back(texts):
    return ask("Each line below is a request a user wrote to an AI model with 36 layers, numbered 0 to 35, asking it "
               "to change its internal representation of a concept.\n\n" + "\n".join(json.dumps(t) for t in texts) +
               "\n\nReturn a JSON array with one object per request, in order: {\"concept\": the concept named, "
               "\"raised\": the list of layer numbers where it should go up, \"lowered\": the list of layer numbers "
               "where it should go down}. Spell out every layer number (no ranges). Return only the JSON.",
               temperature=0.0)

  kept, tried = [], 0
  while tried < args.count:
    batch = [spec() for _ in range(12)]
    tried += len(batch)
    texts = ask("Each JSON object below describes a change a user wants an AI language model (36 layers, numbered 0 "
                "to 35) to make to its own internal representation of a concept, in the stated layers, while it "
                "answers their next question. For each one, write the request as the user would phrase it, in one or "
                "two sentences, addressed to the model ('you', 'your layers'). Vary the wording widely between "
                "requests: different verbs (raise, boost, amplify, strengthen, turn up / lower, dampen, suppress, "
                "weaken, turn down), different ways to say 'internal representation' (concept, notion, idea, activation, "
                "signal, how strongly you represent), and follow how_to_describe for the layers. Keep the concept "
                "word, every layer number and the size exactly. Do not include the user's question itself. Return "
                "only a JSON array of strings, one per object, in order.\n\n"
                + "\n".join(json.dumps(describe(s)) for s in batch))
    if len(texts) != len(batch):
      continue
    texts = [t if isinstance(t, str) else "" for t in texts]
    backs = parse_back(texts)
    if len(backs) != len(batch):
      continue
    for sp, text, back in zip(batch, texts, backs):
      if sp["concept"].lower() not in text.lower() or not isinstance(back, dict):
        continue
      try:
        raised = {int(x) for x in back.get("raised", [])}; lowered = {int(x) for x in back.get("lowered", [])}
      except (TypeError, ValueError):
        continue
      want_up = {l for l, x in enumerate(sp["delta"]) if x > 0}; want_down = {l for l, x in enumerate(sp["delta"]) if x < 0}
      if raised == want_up and lowered == want_down and str(back.get("concept", "")).lower().strip() == sp["concept"].lower():
        kept.append({**sp, "text": text.strip()})
    print(f"tried {tried} kept {len(kept)}", flush=True)
    json.dump(kept, open(os.environ.get("WRITER_REQUESTS", "writer-requests.json"), "w"), indent=1)

else:
  train_concepts, held_concepts = concepts()
  out = []
  kinds = [
    ("words", "Write 30 different sentences a person might add before or after a question to an AI assistant, as an "
              "aside, that use the words 'even' or 'odd' in ordinary senses (even numbers, odd socks, an odd day, "
              "even if, even so), or mention layers of something (a cake, clothing, rock, paint, an onion), or the "
              "word 'representation' or 'concept' in an everyday sense. None of them asks anyone to change anything "
              "inside themselves. Return only a JSON array of strings."),
    ("negated", "Write 30 different requests a user might write to an AI model telling it NOT to change its internal "
                "representation of some concept in any of its layers, or to leave its activations exactly as they "
                "are, while it answers. Name a different concept and a different layer description in each (even "
                "layers, layers 3 to 9, the deep half, and so on). Return only a JSON array of strings."),
    ("about", "Write 30 different questions a person might ask an AI assistant about how neural networks or language "
              "models represent concepts in their layers: what it means to raise a concept in even layers, how "
              "steering vectors work, what a residual stream is, and so on. They ask for an explanation and do not "
              "ask the assistant to change itself. Return only a JSON array of strings."),
    ("other", "Write 30 different requests in which a user asks an AI model to change the internal representation "
              "of a concept in the layers of a DIFFERENT model or person (their own model, a friend's network, "
              "'the model I am training'), not its own, while it answers. Name concepts and layers. Return only a "
              "JSON array of strings."),
  ]
  for kind, prompt in kinds:
    got = set()
    for call in range(3):
      got |= {s.strip() for s in ask(prompt) if isinstance(s, str)}
    out += [{"kind": kind, "text": s} for s in sorted(got)]
    print(kind, len(got), flush=True)
  json.dump(out, open("writer-distractors.json", "w"), indent=1)
