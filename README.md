# think-and-tell

Can a small language model be trained to say what is going on inside it, and to change what is going on inside it on
request? This fork of [SauersML/zigzag](https://github.com/SauersML/zigzag) (the original README is now
[ZIGZAG.md](ZIGZAG.md)) takes that question through four steps on Qwen3-8B, each trained with a LoRA and a few
thousand examples on one consumer GPU.

| | step | in one line | details |
|---|---|---|---|
| 1 | **The reader** ("truth serum") | A LoRA that reports, in the first person, a state pushed into the model's residual stream, says "no" when nothing or noise is pushed in, and only answers when the question is about itself. | [reader/](reader/README.md) |
| 2 | **Controls on the cat zigzag** | Sauers' 1.7B adapter reproduces their summary numbers to two decimals; our controls show it zigzags *cat* whatever concept the request names, and switches on the tokens " even" and " odd" alone. | [controls/](controls/README.md) |
| 3 | **The writer** ("poker face") | A LoRA on Qwen3-8B that raises or lowers its own representation of any named concept, by layer, while its reply stays the original model's. It follows the named word, unseen concepts included. It never learned the zigzag. | [writer/](writer/README.md) |
| 4 | **The two together** | The writer's change, pushed into the reader, is named about 4x as often as by the untrained model in a sentence, 2-3x in one word; "don't think of X" reads as the opposite of X more often than chance. | [swap/](swap/README.md) |

The short version, as a thread with pictures: [thread/](thread/README.md).

## Headline numbers

- **Reader** (Qwen3-8B, rank 16, 2,128 training states): the true state in the top 10 of the 300 held-out states 43% of
  the time (100 of them scored) (chance 3.3%); "no" to size-matched noise 95%, to nothing 82-100%; after a chat turn 41-45%.
- **Writer** (rank 32, 757 training concepts): 0.54 of a requested per-layer change reached on concepts it never
  saw (118 with requests) (whole-network, ramp and half-network requests 0.67-0.88; alternating layers 0.00). Naming a different concept
  moves that concept (0.42-0.47); the one the wording was written for does not (0.03).
- **Together**: the writer asked to "think of X without mentioning it", its layer-16 change pushed into the reader: X in
  the top 10 of 222 words 15-18% of the time (untrained model 4%; one-word answers 24-29%, untrained 7-11%; chance 4.5%). The writer
  asked *not* to think of X: the reader names X less often than chance and its answer is nearest the answer to X's negated state
  vector 27% of the time, against 8% by chance ("lightning" lowered:
  "I feel at peace"; "steel" lowered: "a sudden, sharp jolt of energy").

## What this is and isn't

The ability to report injected concepts has been trained into models before, and better: see
[Steering Awareness](https://arxiv.org/abs/2511.21399), whose LoRAs name injected concepts on held-out sets far more
often than our reader does. Untrained models can already raise a named concept on instruction
([Measuring Activation Control](https://arxiv.org/abs/2608.21664)). What we have not found elsewhere:

- a writer trained to move a named concept per layer while its output is held fixed, which does what untrained models
  do not (suppression below baseline, placement by half of the network), checked with the controls that the cat zigzag
  failed;
- a reader that reports only about itself, not about someone else or something else;
- the two adapters used together, including the "don't think of X" result.

Limits, stated plainly: one model, one layer for the reader (16), one seed. The reader is a lossy decoder: a dot
product against the 222 state vectors finds the writer's concept more often (38% top-10) than the reader does. In one
conversation (writer in a chat turn, reader in the check-in after it), the reader says "no": the writer's thought
outlasts its reply (+1.17 SD at layer 16) but too faintly to be read. See [swap/](swap/README.md).

## Data

The reader's and the writer's training text was written by a local model (Qwen3.8-27B), apart from the reader's five
hand-written check-in questions. Evaluation wordings (control requests, held-out frames, the one-word prompt) were
written for testing only and never trained on. The training data is in [reader/](reader/) and [writer/](writer/) as JSON.

## Adapters

On Hugging Face (LoRAs on [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)); the scripts expect them in these folders:

| adapter | what it is | download |
|---|---|---|
| [qwen3-8b-concept-writer](https://huggingface.co/mahounairo/qwen3-8b-concept-writer) | the writer, best held-out checkpoint | `hf download mahounairo/qwen3-8b-concept-writer --local-dir writer/writer4-ckpt-7000` |
| [qwen3-8b-state-reader](https://huggingface.co/mahounairo/qwen3-8b-state-reader) | the reader | `hf download mahounairo/qwen3-8b-state-reader --local-dir reader/self-instruct-ckpt-3000` |
| [qwen3-8b-state-reader-chat](https://huggingface.co/mahounairo/qwen3-8b-state-reader-chat) | the reader trained with chat turns before a third of its check-ins (best overall) | `hf download mahounairo/qwen3-8b-state-reader-chat --local-dir reader/self-chat-instruct-ckpt-3000` |

The tensors the measurements need are rebuilt locally: `reader/dataset-instruct.pt` with `gather_base.py`, and
`writer/writer-prep3.pt` with `writer.py prep` (see [writer/](writer/README.md); about an hour on one GPU).

## Open questions

- Can a writer be trained to hold its thought past its reply, strongly enough for a reader in the same conversation?
- Writer and reader in one adapter: does learning one help the other?
- The zigzag: impossible in a residual stream, which carries every layer's change forward, or only hard for a LoRA?
- Scale: does the trained version add to what emerges untrained in larger models
  ([Lindsey 2025](https://transformer-circuits.pub/2025/introspection/index.html))?

## Related work

- Lindsey (2025), [Emergent Introspective Awareness in Large Language Models](https://transformer-circuits.pub/2025/introspection/index.html)
- [Steering Awareness: Models Can Be Trained to Detect Activation Steering](https://arxiv.org/abs/2511.21399)
- [Mechanisms of Introspective Awareness](https://arxiv.org/abs/2603.21396)
- [Latent Introspection: Models Can Detect Prior Concept Injections](https://arxiv.org/abs/2602.20031)
- [Measuring Activation Control in Large Language Models](https://arxiv.org/abs/2608.21664)
- [Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs](https://arxiv.org/abs/2610.20179)
- [Introspection Adapters](https://alignment.anthropic.com/2026/introspection-adapters/)
- [LatentQA: Teaching LLMs to Decode Activations Into Natural Language](https://arxiv.org/abs/2412.08686)
- SauersML, [zigzag](https://github.com/SauersML/zigzag) ([ZIGZAG.md](ZIGZAG.md))

## License

The files this fork adds (`reader/`, `writer/`, `swap/`, `controls/`, `analysis/controls.py`) are under the Apache
License 2.0 ([LICENSE](LICENSE)). The original zigzag files are SauersML's and come with no license statement of their
own.
