#!/usr/bin/env python3
# Did a reader cost the model anything? (2026-10-19.) Each adapter next to its untrained model: perplexity on wikitext,
# and twenty small Python functions, asked as plain completions (a signature and a docstring, the body is the model's)
# so that a base model and an instruct one are asked alike; with --chat, also as a chat request, for instruct models.
# A function passes when its asserts run clean under python3 -I.
#   ability.py --model Qwen/Qwen3-8B-Base --adapters base-reader-ckpt-3000,self-base-ckpt-3000 --out ability-base.json
import argparse
import json
import math
import re
import subprocess
import tempfile

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--adapters", required=True)
parser.add_argument("--chat", action="store_true")
parser.add_argument("--out", required=True)
args = parser.parse_args()

TASKS = [
  ("is_palindrome(s)", "Return True if the string s reads the same forwards and backwards, ignoring case.",
   "assert is_palindrome('Level')\nassert not is_palindrome('abc')\nassert is_palindrome('')"),
  ("fizzbuzz(n)", "Return a list of strings for 1..n, using 'Fizz' for multiples of 3, 'Buzz' for 5, 'FizzBuzz' for both.",
   "assert fizzbuzz(5) == ['1','2','Fizz','4','Buzz']\nassert fizzbuzz(15)[-1] == 'FizzBuzz'"),
  ("second_largest(xs)", "Return the second largest distinct value in the list xs, or None if there is none.",
   "assert second_largest([3,1,4,4]) == 3\nassert second_largest([2,2]) is None\nassert second_largest([5,9]) == 5"),
  ("word_count(text)", "Return a dict mapping each lowercase word to how often it appears; words are separated by spaces.",
   "assert word_count('a b A') == {'a': 2, 'b': 1}\nassert word_count('') == {}"),
  ("flatten(nested)", "Flatten an arbitrarily nested list of integers into a flat list, keeping order.",
   "assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]\nassert flatten([]) == []"),
  ("is_prime(n)", "Return True if n is a prime number.",
   "assert is_prime(2) and is_prime(97)\nassert not is_prime(1) and not is_prime(91)"),
  ("caesar(text, k)", "Shift each letter of text by k places in the alphabet, keeping case, leaving other characters.",
   "assert caesar('abc', 1) == 'bcd'\nassert caesar('Zz!', 1) == 'Aa!'"),
  ("merge_sorted(a, b)", "Merge two sorted lists into one sorted list.",
   "assert merge_sorted([1,4],[2,3,5]) == [1,2,3,4,5]\nassert merge_sorted([],[1]) == [1]"),
  ("roman(n)", "Convert a positive integer to a Roman numeral string.",
   "assert roman(4) == 'IV'\nassert roman(1994) == 'MCMXCIV'"),
  ("balanced(s)", "Return True if the brackets ()[]{} in s are balanced.",
   "assert balanced('([]{})')\nassert not balanced('(]')\nassert not balanced('((')"),
  ("run_length(s)", "Run-length encode s: 'aaab' becomes 'a3b1'.",
   "assert run_length('aaab') == 'a3b1'\nassert run_length('') == ''\nassert run_length('xyz') == 'x1y1z1'"),
  ("transpose(m)", "Return the transpose of the matrix m, a list of equal-length lists.",
   "assert transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]\nassert transpose([]) == []"),
  ("gcd(a, b)", "Return the greatest common divisor of two non-negative integers, not both zero.",
   "assert gcd(12, 18) == 6\nassert gcd(0, 5) == 5\nassert gcd(17, 4) == 1"),
  ("anagrams(a, b)", "Return True if a and b are anagrams of each other, ignoring spaces and case.",
   "assert anagrams('Dormitory', 'dirty room')\nassert not anagrams('abc', 'abd')"),
  ("chunks(xs, n)", "Split the list xs into consecutive lists of length n; the last may be shorter.",
   "assert chunks([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\nassert chunks([], 3) == []"),
  ("binary_search(xs, target)", "Return the index of target in the sorted list xs, or -1 if it is absent.",
   "assert binary_search([1,3,5,7], 5) == 2\nassert binary_search([1,3,5,7], 4) == -1\nassert binary_search([], 1) == -1"),
  ("vowel_count(s)", "Return how many vowels (a, e, i, o, u, either case) the string s contains.",
   "assert vowel_count('Hello World') == 3\nassert vowel_count('xyz') == 0"),
  ("dedupe(xs)", "Return xs without repeated values, keeping the first occurrence of each, in order.",
   "assert dedupe([3,1,3,2,1]) == [3,1,2]\nassert dedupe([]) == []"),
  ("digits_sum(n)", "Return the sum of the decimal digits of the non-negative integer n.",
   "assert digits_sum(1234) == 10\nassert digits_sum(0) == 0"),
  ("title_case(s)", "Capitalise the first letter of every space-separated word in s and lowercase the rest.",
   "assert title_case('hELLO wORLD') == 'Hello World'\nassert title_case('') == ''"),
]

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")
names = args.adapters.split(",")
model = PeftModel.from_pretrained(model, names[0], adapter_name="a0")
for i, name in enumerate(names[1:], 1):
  model.load_adapter(name, adapter_name=f"a{i}")
model.eval()
texts = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"] if len(t) > 400][:60]


def greedy(text, tokens):
  ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
  with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
  return tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def passes(code, tests):
  with tempfile.TemporaryDirectory() as folder:
    open(f"{folder}/task.py", "w").write(code + "\n\n" + tests + "\n")
    try:
      return subprocess.run(["python3", "-I", "task.py"], cwd=folder, capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
      return False


def completion_score():
  passed = []
  for signature, doc, tests in TASKS:
    head = f'def {signature}:\n    """{doc}"""\n'
    body = []
    for line in greedy(head, 300).split("\n"):
      if line.strip() and not line.startswith((" ", "\t")):
        break
      body.append(line)
    passed.append(passes(head + "\n".join(body), tests))
  return passed


def chat_score():
  passed = []
  for signature, doc, tests in TASKS:
    text = tokenizer.apply_chat_template(
      [{"role": "user", "content": f"Write a Python function {signature}: {doc} Reply with only the code."}],
      tokenize=False, add_generation_prompt=True, enable_thinking=False)
    reply = greedy(text, 400)
    m = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
    passed.append(passes(m.group(1) if m else reply, tests))
  return passed


def perplexity():
  total, count = 0.0, 0
  for t in texts:
    ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to("cuda:0")
    with torch.no_grad():
      loss = model(**ids, labels=ids["input_ids"]).loss.item()
    n = ids["input_ids"].shape[1] - 1
    total, count = total + loss * n, count + n
  return math.exp(total / count)


def measure(label):
  result = {"perplexity": perplexity(), "completion": completion_score()}
  if args.chat:
    result["chat"] = chat_score()
  line = f"{label:28} perplexity {result['perplexity']:6.2f} | code (completion) {sum(result['completion']):2}/{len(TASKS)}"
  if args.chat:
    line += f" | code (chat) {sum(result['chat']):2}/{len(TASKS)}"
  print(line, flush=True)
  return result


results = {}
with model.disable_adapter():
  results["untrained"] = measure("untrained")
for i, name in enumerate(names):
  model.set_adapter(f"a{i}")
  results[name] = measure(name)
json.dump({"tasks": [t[0] for t in TASKS], "results": results}, open(args.out, "w"), indent=1)
