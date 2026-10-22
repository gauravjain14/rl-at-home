import argparse
import time
import ray
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from driver import model_name, prepare_tasks
from gsm8k import gsm8k_score_fn, extract_answer
from workers import SyncWorker

def collect(worker, input_batch, first_group_id, max_new_tokens, batched):
    if batched:
        return ray.get(worker.collect_groups.remote(
            tasks=[task for task, _ in input_batch],
            input_prompts=[inp_prompt["input_ids"] for _, inp_prompt in input_batch],
            first_group_id=first_group_id,
            run_epoch=0,
            max_samples_per_gen=max_new_tokens,
        ))
    # call it per prompt/task instead of in a batch
    return [ray.get(worker.collect_group.remote(
        task=task, input_prompt=inp_prompt["input_ids"], group_id=first_group_id + i,
        run_epoch=0, max_samples_per_gen=max_new_tokens,
    )) for i, (task, inp_prompt) in enumerate(input_batch)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--num-gens", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batches", type=int, default=1)
    p.add_argument("--batched", action="store_true")
    args = p.parse_args()

    ray.init()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, attn_implementation="sdpa",
        )
        worker = SyncWorker.remote(
            model=model,
            device="cuda:0",
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            score_fn=gsm8k_score_fn,
            num_gens=args.num_gens,
            answer_parser=extract_answer,
            lr=1e-6,
            seed=42,
        )
        del model

        batches = prepare_tasks(batch_size=args.groups, task_name="gsm8k")
        group_id = 0
        for b in range(args.batches):
            input_batch = next(batches)
            start = time.perf_counter()
            groups = collect(worker, input_batch, group_id,
                            args.max_new_tokens, args.batched)
            seconds = time.perf_counter() - start
            group_id += len(groups)

            trajectors = [t for g in groups for t in g.trajectories]
            lengths = [len(t.response_ids) for t in trajectors]
            print({
                "batch": b, "seconds": round(seconds, 1), "answers": len(trajectors),
                "tokens": sum(lengths), "tok_per_s": round(sum(lengths) / seconds, 1),
                "mean_len": round(sum(lengths) / len(lengths), 1), "max_len": max(lengths),
                "reward_mean": round(sum(t.reward for t in trajectors) / len(trajectors), 3),
                "truncated": sum(t.finish_reason == "length" for t in trajectors),
            })

        sample = groups[0].trajectories[0]
        print(tokenizer.decode(sample.response_ids, skip_special_tokens=False))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()