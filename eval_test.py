"""Greedy GSM8K accuracy for one or more checkpoints, plus paired comparisons between them.

    python eval_test.py base=Qwen/Qwen2.5-0.5B-Instruct trained=checkpoints/<run>/step-000460

Every model is scored on the same questions with the same batched fp32 generator used in
training. For each pair: the accuracy difference with a 95% bootstrap CI, how many questions
each one solved that the other did not, and an exact McNemar p-value.
"""
import argparse
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollout import HFGenerator
from tasks import gsm8k


def evaluate(path, tokenizer, inputs, device, chunk=128, max_tokens=512):
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa")
    model = model.to(device).eval()
    eos = model.generation_config.eos_token_id
    eos = (eos,) if isinstance(eos, int) else tuple(eos)
    generator = HFGenerator(model, eos, tokenizer.pad_token_id)
    results = {}
    for i in range(0, len(inputs), chunk):
        part = inputs[i:i + chunk]
        rows = generator.generate([prompt["input_ids"][0].to(device) for _, prompt in part],
                                  num_gens=1, max_samples_per_gen=max_tokens, greedy=True)
        for (task, _), row in zip(part, rows):
            text = tokenizer.decode(row["response_ids"], skip_special_tokens=True)
            results[task["question_id"]] = gsm8k.score(task, text)
    del model
    torch.cuda.empty_cache()
    return results


def paired(a, b, resamples=10000, seed=0):
    # accuracy difference a - b on the same questions: bootstrap 95% CI and exact McNemar test
    keys = sorted(a)
    diff = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    means = sorted(sum(diff[rng.randrange(len(diff))] for _ in diff) / len(diff) for _ in range(resamples))
    lo, hi = means[int(0.025 * resamples)], means[int(0.975 * resamples)]
    gained, lost = sum(d > 0 for d in diff), sum(d < 0 for d in diff)
    n = gained + lost
    tail = sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2 ** n if n else 1.0
    return sum(diff) / len(diff), lo, hi, gained, lost, min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="name=path pairs (a local checkpoint or a hub model id)")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--out", default="eval_results.json",
                        help="per-question results; models already in this file are not re-run")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    models = dict(arg.split("=", 1) for arg in args.models)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    inputs = [entry for batch in gsm8k.batches(tokenizer, split=args.split) for entry in batch]
    cached = json.load(open(args.out)) if os.path.exists(args.out) else {}
    results = {}
    for name, path in models.items():
        key = f"{args.split}/{name}"
        if key in cached:
            results[name] = {int(k): v for k, v in cached[key].items()}
            note = "cached"
        else:
            start = time.perf_counter()
            results[name] = evaluate(path, tokenizer, inputs, args.device)
            cached[key] = results[name]
            json.dump(cached, open(args.out, "w"))
            note = f"{time.perf_counter() - start:.0f}s"
        accuracy = sum(results[name].values()) / len(results[name])
        print(f"{name:14s} {args.split} accuracy {100 * accuracy:5.1f}%  "
              f"({len(results[name])} questions, {note})  {path}", flush=True)

    names = list(models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d, lo, hi, gained, lost, p = paired(results[names[j]], results[names[i]])
            print(f"{names[j]} - {names[i]}: {100 * d:+.1f} pts (95% CI {100 * lo:+.1f} to {100 * hi:+.1f}) | "
                  f"{gained} gained, {lost} lost | McNemar p = {p:.3g}")


if __name__ == "__main__":
    main()
