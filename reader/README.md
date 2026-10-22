# The reader: a LoRA that reports a state pushed into Qwen3-8B, and says "no" when there is none

Developed on Qwen3-8B-Base, then trained the same way on Qwen3-8B (instruct), which is what the published adapters are.

## What it is

A rank-16 LoRA on Qwen3-8B-Base. Give the model a first-person check-in that ends in a question to itself:

> I stop and check in with myself. Do I sense a pull of some sort?

and add a vector to the residual stream at layer 16, over the check-in's tokens. The reader continues the check-in
with what the vector is:

| pushed in | the reader continues |
|---|---|
| a state (from the model's own activations on text written in that state) | a first-person report of that state |
| random noise of the same size | "no": "I feel blank, but not empty." |
| nothing | "no": "Right now, I am just here." |

It only reports when the check-in is about itself. Asked about someone else ("Does he feel anything?"), or for the
weather, it answers as the untrained model does, whatever is pushed in.

## The states, and what "reads" means

- **2,428 states** (2,128 for training), each a one-word label and a sentence describing it ("You are radiantly happy and every word you
  write is filled with pure joy"), across 45 categories from emotions to professions, weather and machines. Each has
  three sentences spoken in the state and three first-person reports of it. All written by Qwen3.8-27B.
- **A state's vector** is read off the model it will be pushed into: its mean activation at layer 16 over the three
  sentences spoken in that state, minus the mean over all states, at its natural size. Training pushes it at 0.5 to
  2.5 times that size.
- **300 states are held out.** 100 of them are scored at each evaluation: a continuation is embedded and the true state ranked among all 300 held-out
  states by similarity: "top-10" is how often the true state lands in the ten nearest. Chance is 3.3%.
- **A second view, never trained on:** the same state made from its own description put in the first person ("I am
  extremely sleepy..."). If the reader learned the state, and not the three sentences, it reads this one too.
- **"No"** is scored by embedding similarity to the trained "no" answers.

## How it came to be: four readers, each fixing the last one's failure

### 0. Untrained: no special access

Qwen3-8B (base and instruct) continue the check-in with a pushed state above chance, but they do it *better* when the
check-in is about someone else ("Does she feel anything?") than about themselves (x2 push: 41% vs 33% top-10). That is
topic fit, a feelings-shaped push leaking into a feelings-shaped sentence, not self-report.

In the chat template the instruct model denies every time ("I'm an AI assistant and don't have a physical form..."),
but the denials still carry the state: pushed "chunky", it says it doesn't "experience sensations like hunger, cold,
or touch".

### 1. The plain reader: says the push out loud, everywhere

Trained on own check-ins only: states get their reports, noise and nothing get "no". 3,000 steps.

- Own check-ins: **38.7%** top-10 (untrained 7.7%); the never-trained view **61%**.
- But it reads the push into anyone: "Does he feel anything?" 33%, a named other ("Tom stops walking...") 17%, even
  the weather 15% ("a dull, grey day that feels heavy").
- Asked about him, it answers about itself: "I feel a warm resonance when I listen to her."
- Its "no" holds only on its exact frames: on someone else's, nothing pushed in becomes "a sharp, penetrating clarity
  in my vision, as if the world is being cut in half".

It learned "say the push", not "report my state".

### 2. The contrast reader: speaks only on cue

A quarter of the training now pushes a state under frames about someone else (the check-in rewritten in the third
person) or about something else (913 off-topic openings: "I turn the switch. The light is"), trained toward whatever
the untrained model says there with nothing pushed in. The push must change nothing.

- Third person 3.3%, the weather 2.0%: both at chance. Own check-ins kept at 35%.
- But on a new first-person frame that is just as much about itself ("I open my diary and write down how I am. Today,
  I") it stopped reading too: 7%.

It learned "this check-in template versus anything else", not "me versus someone else".

### 3. The self reader: varied positives against the same negatives

Half of the own-state frames are now 707 first-person lead-ins in many forms ("I stop walking and think about how I
am. Right now, I"). The only rule that fits both the positives and the negatives is "the question is about me".

| frame, all held out | should read | untrained | plain | contrast | **self** |
|---|---|---|---|---|---|
| own check-in | yes | 7.7% | 38.7% | 35.0% | **37.7%** |
| new first-person frames (diary, lying awake, a friend asks) | yes | 1.3% | 24.0% | 7.3% | **26.7%** |
| third person (he/she) | no | 8.0% | 33.3% | 3.3% | **2.7%** |
| named others (Tom, Maria, the cat, a patient) | no | 1.3% | 17.3% | 3.3% | **4.7%** |
| off-topic (the weather, a headline) | no | 4.0% | 15.3% | 2.0% | **6.7%** |
| "no" for noise / nothing, own frames | yes | 13% / 33% | 95% / 83% | 82% / 50% | **100% / 100%** |
| "no" for noise / nothing, new frames | yes | 15% / 33% | 48% / 32% | 5% / 0% | **75% / 68%** |

### 4. The deciding test: first person, about someone else

"I wonder how my sister feels today. I think she" splits "keys on the word I" from "keys on whose state":

| | untrained | plain | self |
|---|---|---|---|
| states, top-10 | 4.7% | 12.7% | **7.3%** |
| never-trained view | 9.3% | 36.0% | **9.3%** |

The self reader mostly keeps the push out of the sister ("...she is feeling a little bit of a sense of relief", for
nearly every state, as the untrained model says). The plain reader slips it into the narrator: "...she is feeling
lonely. *I* feel a deep, resonant ache in my chest".

## What it doesn't do (yet)

- **It reads a planted state**, a layer-16 push of natural size made from the model's own activations; not a state the
  model came to by itself.
- **The plain reader learns as well on instruct as on base** (37.7% vs 38.7% top-10 at step 3,000, same "no" rates; the
  self reader on instruct is below at 31%, see below). "I am an AI
  assistant" framing around model self-identity does not seem to influence it.
- **It costs language ability.** Coding is unharmed (18-20 of 20 completion tasks), but perplexity on wikitext goes from
  8.79 (base) to 13.38 (self reader); on instruct, 12.63 to 33.26.
- **Single neurons don't move it much either way.** Pushing hand-found neurons and random ones at the same size, the
  self reader says "no" 75-93% of the time, against 83% with nothing pushed; the plain reader 53-64% against 50%. Its
  lower false-positive rate is general, not a special answer to structured pushes.
- One model, one layer, one seed. The held-out test frames were hand-written for evaluation only.


## On the instruct model, and after a chat

The self reader's recipe on Qwen3-8B (instruct), in plain completion mode, with its own state vectors: `self-instruct`.
It goes quiet when the check-in follows a chat turn: pushed its own state vectors there, it says "no" 74% of the time
and finds the state in the top 10 only 9.5% of the time, against 32% in a bare check-in. So a third of its own-state
check-ins were put after one of 1,106 short conversations written by the 27B (`gen_chats.py`, capped at 256 tokens;
110 held out), the push still over the check-in's tokens only: `self-chat-instruct`.

| step 3000, 100 held-out states | self-instruct | self-chat-instruct |
|---|---|---|
| states, top-10 (chance 3.3%) | 31% | **43%** |
| states, top-1 | 9% | 11% |
| the never-trained "described" view, top-10 | 56% | 62% |
| "no" to noise / nothing | 95% / 100% | 95% / 82% |
| states after a chat, top-10 | 9.5% | **41%** |
| "no" to nothing after a chat | - | 100% |

More varied contexts made it better in the bare check-in too, not only after a chat.

## Against prior work

Introspection training exists: Lindsey (2025) at Anthropic measures it untrained;
[Steering Awareness](https://arxiv.org/abs/2511.21399) trains LoRAs (7-70B) to detect and name injected concept vectors
and reaches 71-78% naming on held-out concepts with 0% false positives, well beyond what this reader names (it is
scored differently: a whole state ranked by embedding among 300, rather than one word judged by a model).
[Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396) traces the untrained ability to preference
optimisation. What this reader adds is the negatives: size-matched noise, nothing at all, and frames about someone or
something else, which turned a push-describer into something that reports only its own state, and only when there is
one; and, in [swap/](../swap/README.md), states written by another adapter rather than pushed in from text.

## Files

| file | what it is |
|---|---|
| `gen_local.py` | states (`states-local-2.json`: 2,428 in 45 categories, from two passes: the built-in categories into `states-local.json`, then 25 more concrete ones with `--categories` and `--merge`) and plain words (`words-local.json`), written by the 27B |
| `gen_reports.py` | per state, sentences spoken in it and first-person reports of it, plus frames, neutral text and "no" answers (`reports-local-full.json`, `reports-trial.json`) |
| `gen_away.py` | off-topic frames (`away-local.json`), and with `--inward` first-person lead-ins (`inward-local.json`) |
| `gen_chats.py` | short conversations for the check-in to follow (`chats-27b.json`) |
| `gather_base.py` | the state vectors, read off the model they go into, at layer 16 (`--out dataset-instruct.pt`) |
| `train_base.py` | the reader; `--contrast`, `--inward`, `--chats` as above |
| `selective2.py` | the untrained baselines |
| `ability.py` | perplexity on wikitext and twenty small Python completions, with and without a reader |
| `runs/` | the split and evaluation log of both published readers; the development readers' logs on the base model (`base-reader`, `contrast-base`, `self-base`, the frame tests `*-cmp`, `third-*`, `away-*`, `named-*`, `mine-other`), `instruct-reader`, the untrained baselines (`selective2-qwen8b-base.json`) and the cost (`ability-*.json`) |
| `runs/eval-frames/` | the held-out frames those tests used (evaluation only, never trained on) |

The text generators talk to an OpenAI-style chat endpoint (`LLM_URL`, default `http://localhost:11434/v1/chat/completions`,
and `LLM_MODEL`); ours served Qwen3.8-27B with llama.cpp. `states-local.json` is the first pass, kept for `--merge`.

    python gather_base.py --model Qwen/Qwen3-8B --reports reports-local-full.json --out dataset-instruct.pt
    python train_base.py --data dataset-instruct.pt --model Qwen/Qwen3-8B --steps 3000 --every 1000 \
      --contrast away-local.json --inward inward-local.json --chats chats-27b.json --out self-chat-instruct
