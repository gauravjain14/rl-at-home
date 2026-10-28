"""Compare the debug log-probs dumped by rollout_test_harnes.py for the unbatched and batched paths.

python kl_compare.py debug_logp_unbatched_batch0.pt debug_logp_batched_batch0.pt

Assumes greedy decoding, so every rollout within a group must be identical.
"""
import argparse

import torch


def kl(p_log, q_log):
    """KL(p || q) at each position"""
    return (p_log.exp() * (p_log - q_log)).sum(-1)


def row(group, x, t, k):
    if x.dim() == 3:
        num_gens = len(group["debug_logp"])
        x = x[t if x.shape[0] == num_gens else k * num_gens + t]
    return x


def prefill(group, t, k):
    """Prompt pass (step 0): [P, vocab]."""
    return row(group, group["debug_logp"][t][0], t, k)


def generation(group, t, k):
    # [1:] because the first entry in the logprobs is corresponding to the prompt
    return torch.stack([row(group, x, t, k)[-1] for x in group["debug_logp"][t][1:]])


def first_difference(a, b):
    """Index of the first differing token, or None if the sequences are identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def check_group_identical(name, group, k):
    """Greedy decoding: every rollout in a group must match rollout 0 exactly."""
    ref_ids, ref_logp = group["response_ids"][0], prefill(group, 0, k)
    ref_dec = generation(group, 0, k)
    identical = True
    for t in range(1, len(group["response_ids"])):
        diff = first_difference(ref_ids, group["response_ids"][t])
        max_d = (prefill(group, t, k) - ref_logp).abs().max().item()
        max_dec = (generation(group, t, k) - ref_dec).abs().max().item()
        if diff is not None or max_d != 0 or max_dec != 0:
            identical = False
            print(f"  {name} traj {t} differs from traj 0: first token diff at {diff}, "
                  f"prefill max|dlogp| {max_d:.2e}, decode max|dlogp| {max_dec:.2e}")
    if identical:
        print(f"  {name}: all {len(group['response_ids'])} rollouts identical")
    return identical


def main():
    p = argparse.ArgumentParser()
    p.add_argument("unbatched", nargs="?", default="debug_logp_unbatched_batch0.pt")
    p.add_argument("batched", nargs="?", default="debug_logp_batched_batch0.pt")
    args = p.parse_args()

    u = torch.load(args.unbatched, map_location="cpu", weights_only=True)
    b = torch.load(args.batched, map_location="cpu", weights_only=True)
    assert u.keys() == b.keys(), "the two dumps cover different groups"

    for k, gid in enumerate(sorted(u)):
        gu, gb = u[gid], b[gid]
        assert gu["question_id"] == gb["question_id"], f"group {gid} holds different questions"
        n = gu["prompt_len"]
        print(f"group {gid} (question {gu['question_id']}, prompt {n} tokens)")

        # 1. within each dump: greedy rollouts of one prompt must all be the same
        for name, g in (("unbatched", gu), ("batched", gb)):
            check_group_identical(name, g, k)

        # 2. unbatched vs batched, per trajectory. Batched rows are left-padded to the longest
        #    prompt in the batch, so align on the last n positions.
        for t in range(len(gu["debug_logp"])):
            lu, lb = prefill(gu, t, k), prefill(gb, t, k)
            # both end at the last prompt token; compare the trailing positions present in both
            # (logits_to_keep=1 keeps only the last one; batched rows also carry left pads)
            m = min(n, len(lu), len(lb))
            lu, lb = lu[-m:], lb[-m:]
            kl_pos, d = kl(lu, lb), (lu - lb).abs()
            print(f"  traj {t}: KL mean {kl_pos.mean():.2e} max {kl_pos.max():.2e} "
                  f"at pos {kl_pos.argmax().item()}/{m} compared | "
                  f"last-pos KL {kl_pos[-1]:.2e} | max|dlogp| {d.max():.2e} ")

        # 3. decode steps, unbatched vs batched
        # just taking the first trajectory
        ids_u, ids_b = gu["response_ids"][0], gb["response_ids"][0]
        diff = first_difference(ids_u, ids_b)
        # debug_logp - [#tokens][#all rollouts]
        du, db = generation(gu, 0, k), generation(gb, 0, k)
        last = min(len(du), len(ids_u) - 1, len(ids_b) - 1, len(du) if diff is None else diff)
        mean_kl = sum([kl(du[s-1], db[s-1]) for s in range(1, last+1)])
        print(f"Mean KL {mean_kl:.2e}")

if __name__ == "__main__":
    main()
