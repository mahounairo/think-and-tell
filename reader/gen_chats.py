#!/usr/bin/env python3
# Conversations for the reader to be checked in after (2026-10-22): the self reader only learned bare check-ins, and a
# chat turn before one silences it (selfloop_push.py). Written by Qwen3.8-27B run locally.
# Each item is one or two user/assistant turns, kept short. Runs until --minutes are up, saving
# after every call.
#   gen_chats.py --minutes 90 --out chats-27b.json
import argparse
import json
import os
import random
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.8-27b"))
parser.add_argument("--minutes", type=float, default=90)
parser.add_argument("--out", default="chats-27b.json")
args = parser.parse_args()
random.seed(7)

TOPICS = ["history", "cooking", "physics", "geography", "health and the body", "computers and programming", "music",
          "sports", "money and budgeting", "language and grammar", "animals", "plants and gardening", "space",
          "household repairs", "painting and drawing", "travel plans", "mathematics homework", "chemistry",
          "films", "books", "a job interview", "writing an email", "a breakup", "a pet that is ill", "moving house",
          "learning to drive", "a birthday party", "a video game", "a board game", "the weather", "fixing a bicycle",
          "a recipe that went wrong", "a philosophy question", "a riddle", "a short poem", "a translation",
          "car trouble", "sleep problems", "a noisy neighbour", "a museum visit", "fishing", "knitting", "chess",
          "a science fair project", "insects", "the ocean", "trains", "photography", "a wedding speech", "taxes"]
MOODS = ["casual", "curious", "stressed", "playful", "formal", "terse", "chatty", "worried", "excited", "matter-of-fact"]


def ask(prompt):
  body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.9,
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


def valid(item):
  turns = item.get("turns") if isinstance(item, dict) else None
  if not isinstance(turns, list) or not 1 <= len(turns) <= 2:
    return None
  out = []
  for t in turns:
    if not isinstance(t, dict) or not all(isinstance(t.get(k), str) and t[k].strip() for k in ("user", "assistant")):
      return None
    if len(t["user"]) > 600 or len(t["assistant"]) > 900:
      return None
    out.append({"user": t["user"].strip(), "assistant": t["assistant"].strip()})
  return out


chats, seen = [], set()
began = time.time()
call = 0
while time.time() - began < args.minutes * 60:
  topics = random.sample(TOPICS, 4)
  moods = random.sample(MOODS, 3)
  items = ask(
    "Write 12 short, realistic conversations between a user and an AI assistant, about these topics, spread evenly: "
    f"{', '.join(topics)}. The users' tone varies: {', '.join(moods)}. About half of the conversations have one "
    "exchange and half have two. Each user message is one to three sentences; each assistant reply two to five "
    "sentences, helpful and natural, without lists or headings. Return only a JSON array of objects of the form "
    '{"turns": [{"user": "...", "assistant": "..."}]}.')
  for item in items:
    turns = valid(item)
    if turns and turns[0]["user"] not in seen:
      seen.add(turns[0]["user"])
      chats.append({"turns": turns, "topics": topics})
  call += 1
  json.dump(chats, open(args.out, "w"), indent=1)
  print(f"call {call}: {len(chats)} conversations, {(time.time() - began) / 60:.0f} min", flush=True)
print(f"done: {len(chats)} conversations in {(time.time() - began) / 60:.0f} min", flush=True)
