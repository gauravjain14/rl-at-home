"""Batched greedy eval vs HF generate

"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tasks import gsm8k
from rollout import generate_rollouts

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32,
        attn_implementation="sdpa").to("cuda:0").eval()
eos = tuple(model.generation_config.eos_token_id)
pad = tok.pad_token_id
inputs = [e for b in gsm8k.batches(tok, split="validation", limit=32) for e in b]

t0 = time.perf_counter(); old = []
with torch.no_grad():
    for task, inp in inputs:   # the old SyncWorker.evaluate generation, verbatim
        prompt = {k: v.to("cuda:0") for k, v in inp.items()}
        out = model.generate(**prompt, do_sample=False, max_new_tokens=512,
                             repetition_penalty=1.0, temperature=0.0,
                             pad_token_id=pad, eos_token_id=list(eos))
        old.append(out[0, prompt["input_ids"].shape[1]:].tolist())
t_old = time.perf_counter() - t0

t0 = time.perf_counter()
new = [r["response_ids"] for r in generate_rollouts([inp["input_ids"][0].to("cuda:0") for _, inp in inputs], model,
       num_gens=1, max_samples_per_gen=512, eos_token_ids=eos, pad_token_id=pad, greedy=True)]
t_new = time.perf_counter() - t0

def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))
acc = lambda ids: sum(gsm8k.score(t, tok.decode(r, skip_special_tokens=True)) for (t, _), r in zip(inputs, ids)) / len(ids)
diffs = [(i, first_diff(a, b), len(a)) for i, (a, b) in enumerate(zip(old, new)) if first_diff(a, b) is not None]
print(f"prompts {len(inputs)} | identical responses {len(inputs) - len(diffs)}/{len(inputs)}")
print(f"first differing token (prompt, position, old length): {diffs}")
print(f"accuracy old {acc(old):.3f} | new {acc(new):.3f}")
print(f"time old {t_old:.1f}s | new {t_new:.1f}s")
