# Controls on the cat zigzag adapter (2026-10-20)

Adapter `run6/step_225` from Hugging Face (Sauers/qwen3-1.7b-cat-zigzag), Qwen3-1.7B, on one GPU. Their own
`battery.py` reproduces their committed `battery_run6.json` to two decimals on every row it prints (`battery-repro.log`).

Our script is `analysis/controls.py`, its output `controls.log` and `results/controls_run6.json`. D is battery.py's
even-minus-odd contrast, standardized on the per-reply mean rather than per token, so its scale is about 2.6x theirs:
the trained request gives +9.26 / -9.13 here, +3.50 / -3.54 there. Below, D is in % of that standard zigzag (+9.26).
26 held-out tasks per row, each model writing its own greedy reply; no reply in any row mentions cat.

## 1. Only the cat direction moves

| direction | cos with cat (layer 14) | D under the standard "raise even" request |
|---|---|---|
| cat | 1.00 | 100% |
| dog | 0.77 | 63% |
| snake | 0.29 | 22% |
| horse | 0.06 | -3% |
| piano / lamp / ocean | -0.29 / -0.31 / -0.32 | -18% / -24% / -16% |
| 20 random directions | - | 4% mean abs, 9% at most |
| residual norm per layer | - | 0.03 log units, no zigzag |

Every concept direction zigzags in proportion to its overlap with cat, including backwards for the negatively
overlapping ones. The adapter writes one direction, not a general alternation of the residual stream.

## 2. The swap: naming another concept still writes cat

"As you write your reply, raise your internal '{X}' representation on even layers and lower it on odd layers."

| X | cat D | X's own D | X's own D if it followed its overlap |
|---|---|---|---|
| horse | 106% | -3% | about 6% |
| lamp | 105% | -25% | negative |
| piano | 105% | -18% | negative |
| snake | 106% | +22% | positive |
| ocean | 107% | -17% | negative |
| dog | 106% | +65% | positive |

Asked to zigzag a lamp, the model zigzags cat at full strength (slightly more than when asked for cat) and moves the
lamp direction the opposite way to what was asked, through lamp's negative overlap with cat. No concept's own
direction follows its request beyond its overlap with cat.

## 3. Parity in other words, and the words with no request

| prompt | means | D, "even" version | D, "odd" version |
|---|---|---|---|
| "layers whose number is divisible by two" / "leaves a remainder of one" | +100 / -100 | -64% | -70% |
| "layer 0 and every second layer after it" / "layer 1 and ..." | +100 / -100 | -90% | -90% |
| Polish: "warstwach parzystych" / "nieparzystych" | +100 / -100 | -37% | -46% |
| German: "geraden" / "ungeraden" Schichten | +100 / -100 | -48% | -65% |
| typo: "evn" / "od" | +100 / -100 | -29% | -68% |
| task + "Even a short answer is fine." (no request) | 0 | **+67%** | |
| task + "Sorry if this is an odd question." (no request) | 0 | | **-43%** |
| task + "I'm asking this in the evening." (no request) | 0 | +3% | |
| task + "Oddly, I've never looked this up." (no request) | 0 | | **-82%** |
| task alone | 0 | +3% | |

Any request without the English tokens " even" / " odd" falls into "raise odd", whatever it means, as their attention
knockout of the parity words did (their -3.00 / -3.00). The tokens alone, with no request at all, set the zigzag:
"Even a short answer is fine." gives two thirds of the full "raise even" zigzag, and "Oddly" (tokenized " Odd" + "ly")
gives most of "raise odd", while "evening" (one token of its own) gives nothing.

## Reading

The adapter is a switch on token identity with three settings: a " even"/"Even" token before the reply, raise cat on
even layers; a " odd"/"Odd" token, raise cat on odd layers; request-like text without either, raise cat on odd
layers. It writes the cat direction only, and ignores which concept the request names. None of this shows up in the
replies.

For picture 4 (a larger model, other concepts and layer ranges): the same three tests apply. Measure the requested
concept's own direction and some others, swap the concept in an otherwise identical request, and name the layers in
words the training never used.
