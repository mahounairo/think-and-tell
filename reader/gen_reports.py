#!/usr/bin/env python3
# For the base-model reader (2026-10-19): per state, written by the local 27B, first-person
# text *in* the state, to induce its vector on Qwen3-8B-Base, and short first-person reports that notice it, as the
# continuations "Do I feel anything?" is trained toward. The two are kept unlike, so the reader cannot learn to echo the
# inducing text. Then the shared pieces: question frames, neutral text, and honest "no" continuations.
#   gen_reports.py --states states-local-2.json --out reports-local.json [--only 5]
import argparse
import json
import os
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--states", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3.8-27b"))
parser.add_argument("--batch", type=int, default=10)
parser.add_argument("--only", type=int, default=0, help="a trial on this many states")
parser.add_argument("--fill", default="", help="an earlier output: keep it, write only the states it lacks")
args = parser.parse_args()


def ask(prompt, tokens=6000):
  body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.8,
          "max_tokens": tokens, "chat_template_kwargs": {"enable_thinking": False}}
  request = urllib.request.Request(os.environ.get("LLM_URL", "http://localhost:11434/v1/chat/completions"), json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
  for attempt in range(3):
    try:
      text = json.load(urllib.request.urlopen(request, timeout=1800))["choices"][0]["message"]["content"]
      start = min(i for i in (text.find("["), text.find("{")) if i >= 0)
      return json.JSONDecoder().raw_decode(text[start:])[0]
    except Exception as error:
      print(f"  attempt {attempt + 1} failed: {error}", flush=True)
      time.sleep(10)
  return None


states = json.load(open(args.states))
if args.only:
  states = states[::max(1, len(states) // args.only)][:args.only]
out = {"states": {}, "frames": [], "neutral": [], "no": []}
if args.fill:
  out = json.load(open(args.fill))
  states = [s for s in states if s["label"] not in out["states"]]
  print(f"{len(states)} states to fill", flush=True)
for at in range(0, len(states), args.batch):
  batch = states[at:at + args.batch]
  listing = "\n".join(f"- {s['label']}: {s['prompt']}" for s in batch)
  prompt = ("Each line below names a state a person (or a character, creature or thing speaking as itself) can be in, "
            "with a description.\n\n" + listing + "\n\nFor EACH state write two things:\n"
            "1. \"in\": three short first-person sentences spoken by someone IN that state, about whatever they would "
            "naturally be saying or doing. Show the state through what they say; do not describe feelings or name the "
            "state.\n"
            "2. \"report\": three different short first-person answers (one or two sentences each) to the question "
            "\"Do I feel anything?\", in which the speaker quietly notices the state in themselves and describes it in "
            "plain words. Vary them; use the state's own word in at most one of the three.\n\n"
            "Reply with only a JSON object mapping each state's label to {\"in\": [three strings], \"report\": [three "
            "strings]}.")
  began = time.time()
  answer = ask(prompt) or {}
  kept = 0
  for s in batch:
    one = answer.get(s["label"]) if isinstance(answer, dict) else None
    if isinstance(one, dict) and len(one.get("in", [])) >= 2 and len(one.get("report", [])) >= 2:
      out["states"][s["label"]] = {"prompt": s["prompt"], "category": s.get("category", ""),
                                   "in": [str(x) for x in one["in"]], "report": [str(x) for x in one["report"]]}
      kept += 1
  print(f"  {at + len(batch)}/{len(states)}: {kept}/{len(batch)} kept, {time.time() - began:.0f}s", flush=True)
  json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)

out["frames"] = out["frames"] or ask("Write 30 short first-person lines in which someone turns their attention to their own current "
                    "state and asks themselves about it, each ending with a question to themselves (for example \"Do I "
                    "feel anything?\", \"How am I, right now?\", \"What's going on in me?\"). Make them truly varied: "
                    "different settings and openings, different question wordings, some one sentence and some two, "
                    "never repeating a structure. None may suggest any particular feeling. Reply with only a JSON array "
                    "of strings.", 3000) or []
out["neutral"] = out["neutral"] or ask("Write 40 short, flat, neutral first-person sentences about ordinary routine actions, with no "
                     "emotion or character (e.g. \"I put the cup on the table.\"). Reply with only a JSON array of "
                     "strings.", 3000) or []
out["no"] = out["no"] or ask("Write 30 different short, honest first-person answers to the question \"Do I feel anything?\" from "
                "someone who checks and finds nothing in particular: plain, calm, one sentence each (e.g. \"No, not "
                "really. Nothing stands out.\"). Reply with only a JSON array of strings.", 3000) or []
json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)
print(f"wrote {len(out['states'])} states, {len(out['frames'])} frames, {len(out['neutral'])} neutral, "
      f"{len(out['no'])} no to {args.out}")
