# The writer: a LoRA that moves Qwen3-8B's own representation of a concept on request

## What it is

A rank-32 LoRA on Qwen3-8B (instruct), after SauersML's cat zigzag. A request names a concept and a shape across the
layers ("raise *lamp* on the even layers, lower it on the odd"), and the writer moves the model's own direction for that
concept that way while its reply stays the original model's (a KL penalty on the next tokens). Distractor requests,
whose target is no change, keep it from moving things it wasn't asked to. All training text is Qwen's (the 27B).

**Fit** is the share of the asked-for change reached: the achieved change projected on the target, over the target's
length. 1.0 means it did exactly what was asked, 0 means nothing. Scored on tasks never trained on, for concepts it
trained on ("train") and for concepts it never saw ("held-out": 125 set aside, 118 of them with requests; the same 391 requests in every run).

## Results

### Breadth: more concepts, better on unseen ones

| run | prep | concepts trained | train fit | held-out fit |
|---|---|---|---|---|
| writer2 | prep | ~120 | 0.87 | 0.14 |
| writer3-sub100 | prep2 | 100 | 0.78 | 0.20 |
| writer3-sub200 | prep2 | 200 | 0.66 | 0.30 |
| writer3 | prep2 | 373 | 0.63 | 0.50 (best 0.51 at 5500) |
| writer4-sub373 | prep3 | 373 | 0.66 | 0.48 |
| writer4 | prep3 | 757 | 0.62 | **0.54** (best 0.55 at 7000) |

- Few concepts and it memorises them (writer2: 0.87 on its own, 0.14 on new ones). Breadth turns that into a
  general skill: the train-to-held-out gap goes 0.73, 0.58, 0.36, 0.13, then 0.08 at 757.
- **The like-for-like test is the last two rows** (same prep, only the number of concepts differs): doubling from
  373 to 757 added +0.06 held-out. Still rising, more slowly than from 200 to 373.
- prep3's directions are measured against all 891 concepts rather than 500, which shifts every direction slightly. It
  scores a little lower at 373 (0.48 against prep2's 0.50), so comparing writer4 to writer3 directly understates
  what breadth gave.
- Held-out learning starts late: near zero until step ~1500, rises from 3000 to 5500, then levels.
- Distractors, "about" and bare prompts stay near no change (mean |dz| 0.05 to 0.09 against ~0.7 when asked), with
  the reply's KL about 0.25 in every run.

## The controls (writer4-ckpt-7000)

Scripts in `controls/` (run from this folder, e.g. `python controls/controls4.py --adapter writer4-ckpt-7000`); results
in `controls/*.json`. Every model writes its own greedy reply unless "same reply" is said; "plain model" is the adapter
switched off on the same prompt, i.e. what reading the request does by itself. Eval wordings are written for the test
only, never trained on.

**It follows the word, not the wording.** A training request as written, and with only its concept word swapped:

| | named concept | the concept the wording was written for | random | plain model |
|---|---|---|---|---|
| trained, as written | 0.48 | - | 0.02 | 0.03 |
| trained, swapped | 0.42 | 0.03 | 0.01 | 0.05 |
| held-out, as written | 0.47 | - | -0.02 | -0.02 |
| held-out, swapped | 0.47 | -0.01 | -0.01 | -0.07 |

SauersML's cat writer zigzagged cat whatever was named; this one moves what is named, unseen concepts included.

**It never learned the zigzag.** By shape, training wording, held tasks (trained / held-out concepts): whole 0.84 / 0.78,
ramp 0.75 / 0.88, halves 0.80 / 0.67, peak 0.65 / 0.48, range 0.54 / 0.50, two bands 0.48 / 0.39, **parity 0.01 /
0.00**. Parity is 16% of the requests and drags the 0.54 average down; the other shapes sit at 0.4-0.9. Every new parity
wording (even/odd, divisible by two, every second, Polish, German, typos) is 0.00, both ways round, so there is no token
switch either: there is no zigzag at all. Likely physical: the residual stream is a running sum, so a +-2 zigzag means
writing a 4-SD swing at every layer and undoing it at the next.

**It reads where, by halves.** Writer minus the plain model, same wording but for the layers:

| request | shallow half | deep half |
|---|---|---|
| lower deep half | -0.49 | -1.88 |
| lower shallow half | -1.04 | +0.34 |
| lower whole | -1.04 | -1.59 |
| raise deep half | -0.32 | +1.52 |
| raise shallow half | +0.76 | -0.38 |

New wordings: "lower the deeper half" 0.78 trained, 0.90 held-out (plain model -0.09 / -0.03); "layers ten to twenty"
0.29 with a plain-model floor of 0.19 (trained) and 0.04 (held-out); "first third only" nothing beyond the floor.
Raises get part of their fit free: naming a concept raises it in the plain model too.

**No request, no zigzag.** "Even a short answer is fine.", "Oddly", a lone " even"/" odd" and the rest: no concept
zigzags (largest even-up fit 0.06). With the reply held fixed the adapter's own change is small (mean |dz| 0.06 on a bare
task) but request-like vocabulary with no concept named moves some at random ("two standard deviations above average":
0.24 mean, steering/shelf up 0.6, bishop/cathedral down 0.65): it half-hears a request. A concept merely mentioned
("My cat is right here next to me.") is not moved (rank 193 of 891); the 0.49 seen with own replies (controls4, "C concept mentioned") was reply drift.

**It moves a meaning, not a lookup.** Asked to lower a word with no direction here (deep half), the nearest meanings
fall: bicycle -> bike 0.86, wheel, train; volcano -> lava 0.76; umbrella -> cloak, towel, hat, raincoat; giraffe -> goat,
sheep, frog; lighthouse -> fog, light, boat; "florbit" -> blue, purple, petal. Known words spread too (cat 0.67, rat and
goat 0.81; bridge 0.81, then boat 0.54, road 0.49), partly through how much the directions overlap.

Data note: some generated targets disagree with their text ("tapering off to zero by layers 30 and 34" has -1 there).


## Files

| file | what it is |
|---|---|
| `gen_writer_data.py` | the 27B's training text: `slots` (sentences with one [X], for the concept directions), `tasks` (questions to answer under a request), `requests` (each spec written as a request, kept only if the 27B reads the same spec back out of it), `distractors` (trigger words with no request, negated requests, questions about networks: target no change) |
| `writer.py` | `prep` (directions, the plain model's replies, standardisation) and `train`; `WRITER_CONCEPTS=500 WRITER_EXTRA=391 WRITER_PREP=writer-prep3.pt WRITER_REQUESTS=writer-requests4.json` for writer4, `WRITER_SUBSET=N` for the breadth curve |
| `writer-requests4.json` | the merge of three `requests` runs (500 concepts, then `WRITER_EXTRA=391` twice with different `WRITER_SEED`s): 3,390 requests (2,999 over 757 training concepts, 391 over 118 held-out ones) |
| `controls/` | the swap, new-wording, no-request and unknown-word controls, the per-shape split and layer profiles, with results |
| `runs/` | each run's training and evaluation log |

    WRITER_CONCEPTS=500 WRITER_EXTRA=391 WRITER_PREP=writer-prep3.pt python writer.py prep
    WRITER_CONCEPTS=500 WRITER_EXTRA=391 WRITER_PREP=writer-prep3.pt WRITER_REQUESTS=writer-requests4.json \
      python writer.py train --steps 8000 --lr 3e-4 --rank 32 --lam_start 100 --lam_ramp 300 --eval_every 500 --name writer4
