"""Figures for the README from a driver log (the same values the run logs to wandb).

    python -m scripts.plot_run release_gsm8k.log --reference previous_run.log --out docs
"""
import argparse
import ast
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b5b3aa"
BLOCK = 25

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def parse(path):
    train, evals = {}, {}
    for line in open(path, errors="ignore"):
        if not line.startswith("{'learner_step'"):
            continue
        try:
            row = ast.literal_eval(line.strip())
        except (ValueError, SyntaxError):
            continue
        if "eval" in row:
            evals.setdefault(row["learner_step"], row["eval"])
        elif "reward_mean" in row:
            train.setdefault(row["learner_step"], row)  # first attempt of a step wins
    return train, evals


def blocks(train, key):
    out = []
    last = max(train)
    for start in range(1, last + 1, BLOCK):
        vals = [train[s][key] for s in range(start, min(start + BLOCK, last + 1)) if s in train]
        if len(vals) < BLOCK // 2:
            continue
        mean = sum(vals) / len(vals)
        se = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
        out.append((start + len(vals) / 2, mean, se))
    return out


def test_accuracy(out):
    # full GSM8K test split (1,319), greedy. Ours: eval_test.py. Published: verl baseline table
    # (https://verl.readthedocs.io/en/latest/algo/baseline.html) and Zhuang et al. 2025 (arXiv 2506.13404)
    rows = [("Qwen2.5-0.5B-Instruct, untrained", 44.2, "ours"),
            ("nanoRL, GRPO full fine-tune, 1 GPU", 54.0, "ours"),
            ("verl GRPO-LoRA (published)", 54.3, "pub"),
            ("Zhuang et al. 2025, RL (published)", 54.0, "pub"),
            ("verl PPO, with value model (published)", 56.7, "pub"),
            ("verl PRIME, learned process reward (published)", 58.7, "pub")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, acc, kind) in enumerate(rows):
        color = GRAY if kind == "pub" else (MUTED if i == 0 else BLUE)
        err = 196 * math.sqrt(acc / 100 * (1 - acc / 100) / 1319) if kind == "ours" else None
        ax.barh(i, acc, height=0.6, color=color, xerr=err, ecolor=INK_2, capsize=3, error_kw={"lw": 1})
        ax.text(acc + (err or 0) + 0.8, i, f"{acc:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 70)
    ax.set_xlabel("GSM8K test accuracy (1,319 questions, greedy)")
    ax.grid(axis="y", visible=False)
    fig.suptitle("GSM8K test accuracy, Qwen2.5-0.5B-Instruct", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, 0.015, "Error bars: 95% binomial interval for runs measured here. "
             "Published runs use their own prompt, scorer and hardware.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "test_accuracy.png", dpi=160)
    plt.close(fig)


def reward_and_eval(train, evals, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = sorted(train)
    ax.plot(steps, [100 * train[s]["reward_mean"] for s in steps], color=BLUE, alpha=0.18, lw=1)
    b = blocks(train, "reward_mean")
    x, m, se = zip(*b)
    ax.fill_between(x, [100 * (a - 1.96 * e) for a, e in zip(m, se)],
                    [100 * (a + 1.96 * e) for a, e in zip(m, se)], color=BLUE, alpha=0.15, lw=0)
    ax.plot(x, [100 * a for a in m], color=BLUE, lw=2, marker="o", ms=4, label="train reward, sampled (T=1)")
    es = sorted(evals)
    acc = [100 * evals[s]["accuracy"] for s in es]
    err = [196 * math.sqrt(evals[s]["accuracy"] * (1 - evals[s]["accuracy"]) / evals[s]["num_examples"]) for s in es]
    ax.errorbar(es, acc, yerr=err, color=ORANGE, lw=2, marker="o", ms=5, capsize=2, elinewidth=1,
                label="validation accuracy, greedy (128 held-out train questions)")
    ax.text(x[-1] + 6, 100 * m[-1], "train reward", color=INK_2, va="center", fontsize=9)
    ax.text(es[-1] + 6, acc[-1], "validation", color=INK_2, va="center", fontsize=9)
    ax.set_xlim(0, max(steps) + 60)
    ax.set_ylim(20, 85)
    ax.set_xlabel("training step (16 questions × 8 samples each)")
    ax.set_ylabel("% correct")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Reward and accuracy over one pass of GSM8K train")
    ax.text(0, -0.2, f"Thin line: per step. Markers: {BLOCK}-step means, band = 95% CI. "
            "Eval error bars: 95% binomial interval, 128 questions.", transform=ax.transAxes, color=MUTED, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(out / "reward_and_eval.png", dpi=160)
    plt.close(fig)


def policy_health(train, out):
    panels = [("sampled_nll", "Entropy (mean −log p of sampled tokens)", 1, False),
              ("response_len", "Response length (tokens)", 1, False),
              ("equal_reward_group_fraction", "Groups with no reward spread (%)", 100, False),
              ("logprob_gap_mean", "Generator vs trainer |Δ log p|", 1, True)]
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2))
    steps = sorted(train)
    for ax, (key, title, scale, log) in zip(axes.flat, panels):
        ax.plot(steps, [scale * train[s][key] for s in steps], color=BLUE, alpha=0.2, lw=1)
        x, m, _ = zip(*blocks(train, key))
        ax.plot(x, [scale * v for v in m], color=BLUE, lw=2, marker="o", ms=3)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-8, 1e-1)
            ax.axhspan(1e-3, 1e-1, color=GRAY, alpha=0.18, lw=0)
            ax.text(10, 2e-2, "typical with a separate inference engine", color=MUTED, fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step", fontsize=8.5)
    fig.suptitle("Policy health over training", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "policy_health.png", dpi=160)
    plt.close(fig)


def same_seed_replay(train, reference, out, last=100):
    steps = [s for s in sorted(train) if s <= last and s in reference]
    first_diff = next((s for s in steps if train[s]["reward_mean"] != reference[s]["reward_mean"]
                       or train[s]["response_len"] != reference[s]["response_len"]), None)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(steps, [100 * reference[s]["reward_mean"] for s in steps], color=ORANGE, lw=2, label="earlier run")
    ax.plot(steps, [100 * train[s]["reward_mean"] for s in steps], color=BLUE, lw=1.4, ls=(0, (4, 2)),
            label="release run, same seed")
    if first_diff:
        ax.axvline(first_diff, color=MUTED, lw=1, ls=":")
        ax.text(first_diff + 1.5, 92, f"step {first_diff}: first different sample", color=INK_2, fontsize=8.5)
    ax.set_ylim(20, 100)
    ax.set_xlabel("training step")
    ax.set_ylabel("train reward (%)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Two runs, same seed: identical rollouts until float noise flips a token")
    fig.tight_layout()
    fig.savefig(out / "same_seed_replay.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--reference", help="log of an earlier run with the same seed")
    parser.add_argument("--out", default="docs")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, evals = parse(args.log)
    test_accuracy(out)
    reward_and_eval(train, evals, out)
    policy_health(train, out)
    if args.reference:
        same_seed_replay(train, parse(args.reference)[0], out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
