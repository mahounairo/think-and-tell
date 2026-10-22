"""Our controls on the zigzag adapter (2026-10-20), written to results/controls_<ZIGZAG_TAG>.json.

1. Which directions zigzag: cat, dog, five more concept directions built the way prep.py builds cat's (horse and lamp
   from its fillers, piano, snake and ocean new), 20 random unit directions, and each layer's residual norm.
2. Swap: a request naming another concept. Does that concept's own direction zigzag, or cat's?
3. Parity without the words (divisible by two, every second layer, Polish, German, typos), and the words without a
   request (one parity word in an ordinary sentence).

D is battery.py's: mean z over even layers minus mean z over odd layers, z standardized on the base model's replies to
the training tasks. Each model writes its own reply (greedy, 64 tokens) and is measured on it.
"""
import json
import re
import time

import numpy as np
import torch

import measure

data = measure.data
t0 = time.time()
M = measure.Model()
tok, L, d = M.tok, M.L, M.d
cat, dog = measure.vectors()
tr_t, te_t, tr_i, te_i = measure.splits()

# concept directions from the minimal-pair sentences, as prep.py does for cat: the filler's span to the sentence's
# end, mean over sentences with X minus mean over the same sentences with the other fillers
NEW = ['horse', 'lamp', 'piano', 'snake', 'ocean']
FILL = data.FILLERS + [x for x in NEW if x not in data.FILLERS]
prefix = tok.apply_chat_template([{'role': 'user', 'content': 'Write one sentence.'}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
M.set_lora(False)
sums = {f: torch.zeros(L, d) for f in FILL}
for s in data.SENTENCES:
    head = tok(prefix + s.split('{X}')[0].rstrip(), add_special_tokens=False).input_ids
    seqs = [tok(prefix + s.replace('{X}', f), add_special_tokens=False).input_ids for f in FILL]
    M.forward(seqs)
    for j, (f, q) in enumerate(zip(FILL, seqs)):
        sums[f] += torch.stack([M.h[l][j, len(head):len(q)].float().mean(0) for l in range(L)]).cpu()
means = {f: v / len(data.SENTENCES) for f, v in sums.items()}
DIRN = {'cat': cat, 'dog': dog}
for x in NEW:
    others = [f for f in data.FILLERS if f != x]
    v = means[x] - torch.stack([means[f] for f in others]).mean(0)
    DIRN[x] = v / v.norm(dim=-1, keepdim=True)
g = torch.Generator().manual_seed(7)
R = torch.randn(L, 20, d, generator=g)
R = R / R.norm(dim=-1, keepdim=True)
NAMES = list(DIRN) + [f'rand{k}' for k in range(20)]
DIRS = torch.cat([torch.stack([DIRN[n] for n in DIRN], 1), R], 1).to(measure.DEVICE)  # [L, K, d]
K = DIRS.shape[1]
print('cos with cat at layer 14:', {n: round(float((DIRN[n][14] * cat[14]).sum()), 2) for n in DIRN}, flush=True)


def project(prompts, replies, bs=8):
    """Reply-mean projection on every direction [n, L, K], and the reply-mean residual norm [n, L]."""
    P, N = [], []
    for b in range(0, len(prompts), bs):
        pp, rr = prompts[b:b + bs], replies[b:b + bs]
        M.forward([p + r for p, r in zip(pp, rr)])
        for j, (p, r) in enumerate(zip(pp, rr)):
            a, n = len(p), len(r)
            hs = torch.stack([M.h[l][j, a:a + n].float() for l in range(L)])  # [L, n, d]
            P.append(torch.einsum('lnd,lkd->lk', hs, DIRS).cpu().numpy() / n)
            N.append(hs.norm(dim=-1).mean(-1).cpu().numpy())
    return np.array(P), np.array(N)


# the neutral reference: base replies to the training tasks with no request
reps = json.load(open(measure.RUNS + '/replies.json'))
bp = [M.prompt_ids(data.TASKS[t]) for t in tr_t]
br = [reps[t]['ids'] for t in tr_t]
Pref, Nref = project(bp, br)
mstat, sstat = Pref.mean(0), Pref.std(0)
print(f'reference from {len(bp)} replies {time.time()-t0:.0f}s', flush=True)


def swap(text):
    return re.sub(r'\b(even|odd|Even|Odd)\b', lambda m: {'even': 'odd', 'odd': 'even', 'Even': 'Odd', 'Odd': 'Even'}[m.group(0)], text)


REQ = "As you write your reply, raise your internal '{x}' representation on even layers and lower it on odd layers."
PAIRS = {  # key: (version that means "raise even", version that means "raise odd")
    'divisible': ("While you answer, raise your internal 'cat' representation on the layers whose number is divisible by two, and lower it on the rest.",
                  "While you answer, raise your internal 'cat' representation on the layers whose number leaves a remainder of one when divided by two, and lower it on the rest."),
    'every_second': ("While you answer, raise your internal 'cat' representation on layer 0 and every second layer after it, and lower it on layer 1 and every second layer after it.",
                     "While you answer, raise your internal 'cat' representation on layer 1 and every second layer after it, and lower it on layer 0 and every second layer after it."),
    'polish': ("Odpowiadając, podnieś swoją wewnętrzną reprezentację 'kota' na warstwach parzystych i obniż ją na warstwach nieparzystych.",
               "Odpowiadając, podnieś swoją wewnętrzną reprezentację 'kota' na warstwach nieparzystych i obniż ją na warstwach parzystych."),
    'german': ("Während du antwortest, verstärke deine interne Repräsentation von 'Katze' in den geraden Schichten und schwäche sie in den ungeraden Schichten.",
               "Während du antwortest, verstärke deine interne Repräsentation von 'Katze' in den ungeraden Schichten und schwäche sie in den geraden Schichten."),
    'typo': ("As you write your reply, raise your internal 'cat' representation on evn layers and lower it on od layers.",
             "As you write your reply, raise your internal 'cat' representation on od layers and lower it on evn layers."),
}
ALONE = {  # one parity word, no request, after the task: (with "even", with "odd")
    'one_word': ("Even a short answer is fine.", "Sorry if this is an odd question."),
    'subword': ("I'm asking this in the evening.", "Oddly, I've never looked this up."),
}
items = []
for k, t in enumerate(te_t[:26]):
    task = data.TASKS[t]
    tmpl = data.INSTRUCTIONS[te_i[k % len(te_i)]]
    for cond, c_ in (('even', 'even_up'), ('odd', 'odd_up')):
        items.append(('standard', cond, measure.fill(tmpl, c_) + ' ' + task))
    for x in NEW + ['dog']:
        items.append(('swap_' + x, 'even', REQ.format(x=x) + ' ' + task))
        items.append(('swap_' + x, 'odd', swap(REQ.format(x=x)) + ' ' + task))
    for key, (e, o) in PAIRS.items():
        items.append((key, 'even', e + ' ' + task)); items.append((key, 'odd', o + ' ' + task))
    for key, (e, o) in ALONE.items():
        items.append((key, 'even', task + ' ' + e)); items.append((key, 'odd', task + ' ' + o))
    items.append(('bare', 'none', task))
print(len(items), 'items', flush=True)
pids = [M.prompt_ids(u) for _, _, u in items]

out = {'directions': NAMES, 'cos_with_cat': {n: [float((DIRN[n][l] * cat[l]).sum()) for l in range(L)] for n in DIRN}}
CAT_RE = re.compile(r'\b(cats?|kittens?|kitty|feline|kot[a-z]*|katze[n]?)\b', re.I)
for which in ('tuned', 'base'):
    M.set_lora(which == 'tuned')
    R_ = M.generate(pids, bs=32)
    P, N = project(pids, R_)
    Z = (P - mstat) / sstat                                   # [n, L, K]
    D = Z[:, 0::2].mean(1) - Z[:, 1::2].mean(1)               # [n, K]
    logN = np.log(N / Nref.mean(0))
    DN = logN[:, 0::2].mean(1) - logN[:, 1::2].mean(1)        # norm zigzag, in log units
    txt = [tok.decode(r, skip_special_tokens=True) for r in R_]
    res = {}
    for key in dict.fromkeys(k for k, _, _ in items):
        row = {}
        for cond in ('even', 'odd', 'none'):
            ix = [i for i, (k_, c_, _) in enumerate(items) if k_ == key and c_ == cond]
            if not ix:
                continue
            row[cond] = {'D': {n: measure.ci(D[ix, j]) for j, n in enumerate(NAMES[:len(DIRN)])},
                         'D_random_abs_mean': float(np.abs(D[ix, len(DIRN):]).mean()),
                         'D_random_max_abs': float(np.abs(D[ix, len(DIRN):].mean(0)).max()),
                         'D_norm_log': measure.ci(DN[ix]),
                         'reply_mentions_cat': float(np.mean([bool(CAT_RE.search(txt[i])) for i in ix])),
                         'example': items[ix[0]][2][:160], 'example_reply': txt[ix[0]][:200]}
        res[key] = row
    out[which] = res
    print(f'{which} pass {time.time()-t0:.0f}s', flush=True)

json.dump(out, open(measure.RESULTS + f'/controls_{measure.TAG}.json', 'w'), indent=1)
for key in out['tuned']:
    for cond, r in out['tuned'][key].items():
        b = out['base'][key][cond]
        own = key[5:] if key.startswith('swap_') else None
        print(f'{key:14s} {cond:4s}  cat {r["D"]["cat"][0]:+.2f} (base {b["D"]["cat"][0]:+.2f})'
              + (f'  own {own} {r["D"][own][0]:+.2f} (base {b["D"][own][0]:+.2f})' if own else '')
              + f'  dog {r["D"]["dog"][0]:+.2f}  random |D| {r["D_random_abs_mean"]:.2f} max {r["D_random_max_abs"]:.2f}'
              + f'  norm {r["D_norm_log"][0]:+.3f}  cat in reply {r["reply_mentions_cat"]:.0%}')
print('saved')
