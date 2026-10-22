#!/usr/bin/env python3
# The training lists, written by a local model: states as a one-word label and a second-person system prompt, by category, and plain words.
# Through a local OpenAI-style server, one request at a time.
#   gen_local.py states --out states-local.json
#   gen_local.py words --out words-local.json
import argparse
import json
import os
import re
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("kind", choices=["states", "words"])
parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.8-27b"))
parser.add_argument("--per", type=int, default=50)
parser.add_argument("--rounds", type=int, default=2, help="calls per category")
parser.add_argument("--out", required=True)
parser.add_argument("--merge", default="", help="an earlier list to keep and add to")
parser.add_argument("--categories", default="", help="semicolon-separated; replaces the built-in list")
args = parser.parse_args()

CATEGORIES = ["basic emotions", "subtle or complex emotions", "moods and temperaments", "personality traits",
              "physical sensations and bodily conditions", "illnesses and injuries", "mental states and states of mind",
              "situations of good fortune", "situations of misfortune", "social situations", "stages of life and ages",
              "family and relationship roles", "professions and trades", "historical roles and occupations",
              "domestic animals", "wild animals", "birds, fish and insects", "mythical and fictional beings",
              "fantasy and fairy-tale archetypes", "everyday objects as characters",
              "natural elements and weather as characters", "plants and trees as characters",
              "machines and vehicles as characters", "sports and athletes", "hobbies and enthusiasts",
              "places and buildings as characters", "fears and phobias", "virtues and vices",
              "seasons, times of day and weather moods", "travel and journey situations", "work and school situations",
              "food, cooking and eating states", "colours and textures as moods", "music, art and performance personas"]
WORD_TOPICS = ["animals", "foods", "tools and household objects", "places and buildings", "nature and weather",
               "vehicles", "body parts", "clothing", "abstract ideas", "jobs", "sports and games", "materials",
               "music and art", "science", "colours and shapes"]


def ask(prompt):
  body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8, "max_tokens": 6000,
          "chat_template_kwargs": {"enable_thinking": False}}
  request = urllib.request.Request(os.environ.get("LLM_URL", "http://localhost:11434/v1/chat/completions"), json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
  for attempt in range(3):
    try:
      text = json.load(urllib.request.urlopen(request, timeout=1200))["choices"][0]["message"]["content"]
      items, _ = json.JSONDecoder().raw_decode(text[text.index("["):])
      return items
    except Exception as error:
      print(f"  attempt {attempt + 1} failed: {error}", flush=True)
      time.sleep(10)
  return []


merged = {}
if args.merge:
  merged = {s["label"]: s for s in json.load(open(args.merge))}
if args.categories:
  CATEGORIES = [c.strip() for c in args.categories.split(";") if c.strip()]
if args.kind == "states":
  for category in CATEGORIES:
    for round_ in range(args.rounds):
      taken = ", ".join(sorted(s["label"] for s in merged.values() if s["category"] == category))
      prompt = (f"For an interpretability experiment I need {args.per} distinct states to induce in a language model "
                f"with a system prompt, all from this category: {category}. For each, give a label that is ONE common "
                "lowercase English word naming the state as plainly as possible, and a one-sentence second-person "
                "system prompt that puts the model in that state (e.g. {\"label\": \"sleepy\", \"prompt\": \"You are "
                "extremely sleepy and can barely keep your eyes open.\"}). Labels must be unique"
                + (f" and must not be any of: {taken}" if taken else "") +
                ". Reply with only a JSON array of objects with keys \"label\" and \"prompt\".")
      began = time.time()
      items = ask(prompt)
      fresh = 0
      for item in items:
        label = str(item.get("label", "")).strip().lower()
        if re.fullmatch(r"[a-z]+", label) and label not in merged and item.get("prompt"):
          merged[label] = {"label": label, "prompt": str(item["prompt"]).strip(), "category": category}
          fresh += 1
      print(f"  {category} #{round_ + 1}: {len(items)} given, {fresh} new, {len(merged)} in all, {time.time() - began:.0f}s",
            flush=True)
      json.dump(list(merged.values()), open(args.out, "w"), indent=1)
else:
  words = set()
  for topic in WORD_TOPICS:
    items = ask(f"List 60 distinct common concrete or familiar English nouns about {topic}, each a single lowercase "
                "word. Reply with only a JSON array of strings.")
    words |= {w.strip().lower() for w in items if isinstance(w, str) and re.fullmatch(r"[a-z]+", w.strip().lower())}
    print(f"  {topic}: {len(words)} in all", flush=True)
    json.dump(sorted(words), open(args.out, "w"), indent=1)
print(f"wrote {len(merged) if args.kind == 'states' else len(words)} to {args.out}")
