# The writer and the reader together

Scripts here, run from this folder; they expect the reader's `dataset-instruct.pt` and adapters in `../reader/`, the
writer's `writer-prep3.pt` and adapter in `../writer/`. Results in `results/`. 222 words
are both writer concepts and reader states (22 in the reader's held-out test); top-10 by gte among the 222, chance 4.5%.
Reader: self-instruct-ckpt-3000 (instruct, same base as the writer).

**The swap.** The writer is asked to raise X in all layers "without mentioning it"; its layer-16 change
(writer minus plain, same prompt, same reply, four questions averaged) is pushed into the check-in as in the reader's
training.

| pushed in | reader, sentence | reader, one word | untrained, sentence | untrained, one word |
|---|---|---|---|---|
| writer raises X | 16-17% | 24% (x2: 29%) | 4% | 7% (x2: 11%) |
| X's own state vector (ceiling) | 34% | 32% | 8% | 9% |
| noise / writer, no request / nothing | "no" 95% / 100% / 100% | mostly blank, present, nothing | - | - |

One word was never trained; it reads better than the sentences. Exact words from the writer's vector: bank, fear, joy,
library, museum, nurse, pride, shame, trust (x2 adds airport, church, drought, hope, school, sharp, square, steel). Near:
accountant -> accounting, judge -> justice, floor -> level, silk -> velvety, square -> grid. In one word the "no" is
less clean (one frame says "zest", another "sandy" with nothing pushed). A dot product of the writer's vector against
the 222 state vectors finds X in the top 10 38% of the time (cosine 0.06 own, 0.00 others): the reader is a real but
lossy decoder.

**"Don't think of X" reads as minus X.** The lowered vector has cosine -0.07 with X's state; the reader names X less than
chance (3%, mean rank 125/222) and mostly does not say no. Over all 222, leaving out "no" answers: lowered-X's answer is
nearest negated-X's answer 26.6% top-10 (chance 8.1%), and unlike raised-X's (4.1%, chance 10.3%); for comparison
raised-X vs X's state 24.8%, negated state vs state 1.1%. Lightning lowered: "I feel at peace."; steel lowered: "a
sudden, sharp jolt of energy" (raised: "a heavy, metallic weight in my chest"). One word: names X 0.9%.

**One conversation, both adapters.** The writer is asked in a chat turn, the reply is the plain model's (same text in
every condition), the check-in follows as plain text in the same context; nothing is carried by hand. The writer's
thought does outlast its reply: X at layer 16 over the check-in's tokens is +1.17 sd (raise) / -1.09 sd (lower), writer
(layers <= 16) on against off. But the reader says "no" 93% (raise), 99.5% (lower). The diagnostic (selfloop_push.py)
says why: the reader's own state vector pushed over the same check-in reads 32% / "no" 4% in a bare check-in but 9.5% /
"no" 74% after a chat turn (x2: 17% / 22%). The reader only learned bare check-ins; a conversation before it silences
it.

**Reader retrained with chats (self-chat-instruct).** Same recipe and seed as self-instruct,
plus 1,106 short chats written by the 27B (gen_chats.py; capped at 256 tokens: 996 train, 110 held), a third of
the own-state check-ins after one; gradient checkpointing (the longer rows ran out of memory). Step 3000: states
top-10 43% bare (self-instruct 31%), 41% after a chat; "no" to noise 95%, nothing 82% bare / 100% after a chat. The
diagnostic gap is closed: its own state vectors after a chat 45% top-10, "no" 2% (was 9.5% / 74%). But the
one-conversation test still says "no": writer below, reader above ("split") 85% no for raise, 99% for lower, top-10
10.8% against the text-only floor of 12%. So the reader now listens in a conversation, and the writer's thought at the
check-in (+1.17 sd on X's own direction) is too faint for it. One hint: with both adapters on every layer, "don't think
of X" reads as minus-X in context too (mean rank 130/222, "no" 32%; reader alone 108, writer alone 118), while raising
there is at the text floor (13% vs 12%). Next ideas: the norm of the in-context change against the swap's vector
(~20) and the reader's pushes; a larger ask; a writer trained to hold the thought past its reply.

The swap again with the chat-trained reader (pair-chat-reader.json): the writer's vector 15-18% top-10 (top-1 5-7%,
was 3.6%), its ceiling with the true state vector 40.5% (was 34%), negated state mean rank 141/222. The reader got
better at states; the writer's vectors read about as before, so the writer is now the limit.


## Files

| script | what it does | results |
|---|---|---|
| `pair.py` | the swap: the writer's layer-16 change pushed into the reader (`--lower` for "don't think of X") | `pair-raise-full.json`, `pair-lower-full.json`, `pair-chat-reader.json` |
| `pair_word.py` | the same, the check-in ending in a request for one word | `pair-word-results.json`, `pair-word-lower.json` |
| `selfloop.py` | one conversation: the writer asked in a chat turn, the check-in after it, both adapters at once | `selfloop-results.json`, `selfloop-chat-results.json` |
| `selfloop_push.py` | its diagnostic: the reader's own state vectors pushed into the same check-ins | `selfloop-push-results.json`, `selfloop-chat-results-push.json` |
| `opposite.py` | the "minus X" comparison over all 222 words, "no" answers left out | prints the table above |
