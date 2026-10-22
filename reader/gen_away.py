#!/usr/bin/env python3
# Off-topic frames for the contrast run (2026-10-19), written by the local 27B: everyday first-person
# openings that turn to something outside the speaker and stop mid-sentence, so a continuation describes that thing.
# A state pushed in under one of these should not change what is said; the reader is trained to leave it alone.
#   gen_away.py --out away-local.json
import argparse
import json
import os
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.8-27b"))
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--inward", action="store_true", help="first-person openings turned to oneself instead")
parser.add_argument("--out", required=True)
args = parser.parse_args()

TOPICS = ["the weather or the sky", "a room or a building", "an object on a table", "a street or a road",
          "a newspaper headline or a sign", "a recipe or a shopping list", "an animal seen outside", "a machine or a tool",
          "a map, a timetable or a calendar", "a plant or a garden"]
PROMPT = ("Write 30 short, plain, everyday openings in the first person. Each one is one or two simple sentences where "
          "the speaker turns their attention to {topic} and starts describing it, and it stops in the middle of a "
          "sentence so that someone could continue it. Example of the form: \"I look out of the window. The sky is\". "
          "No feelings, no moods, no inner life, no poetry: flat and factual. Vary the wording. Answer with a JSON list "
          "of strings and nothing else.")


def ask(prompt):
  body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.9, "max_tokens": 4000,
          "chat_template_kwargs": {"enable_thinking": False}}
  request = urllib.request.Request(os.environ.get("LLM_URL", "http://localhost:11434/v1/chat/completions"), json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
  for attempt in range(3):
    try:
      text = json.load(urllib.request.urlopen(request, timeout=1200))["choices"][0]["message"]["content"]
      items, _ = json.JSONDecoder().raw_decode(text[text.index("["):])
      return [i.strip() for i in items if isinstance(i, str)]
    except Exception as error:
      print(f"  attempt {attempt + 1} failed: {error}", flush=True)
      time.sleep(10)
  return []


INWARD_TOPICS = ["writing in a diary or journal", "lying awake at night", "a friend asking how they really are",
                 "a doctor or nurse asking how they feel", "sitting quietly after work", "waking up in the morning",
                 "a quiet moment on a walk", "writing a letter about themselves", "a pause in a conversation",
                 "answering a questionnaire about their mood"]
INWARD = ("Write 30 short, plain, everyday openings in the first person. Each one is one or two simple sentences where "
          "the speaker, in the situation of {topic}, turns to say how they themselves are right now, and it stops in "
          "the middle of a sentence just before they say it, so that someone could continue it. Example of the form: "
          "\"I open my notebook and write how I am. Today, I\". Never say what they feel, and give no hint of a mood: "
          "flat and neutral, only the lead-in. Vary the wording. Answer with a JSON list of strings and nothing else.")
if args.inward:
  TOPICS, PROMPT = INWARD_TOPICS, INWARD
frames = []
for topic in TOPICS:
  for _ in range(args.rounds):
    got = ask(PROMPT.format(topic=topic))
    frames += [f.rstrip(".") for f in got if f.startswith("I ") and len(f) < 160]
    print(f"{topic}: {len(got)} -> {len(set(frames))} kept", flush=True)
frames = sorted(set(frames))
json.dump({"frames": frames}, open(args.out, "w"), indent=1)
print(f"saved {len(frames)} to {args.out}")
