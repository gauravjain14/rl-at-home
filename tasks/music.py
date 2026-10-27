from datasets import load_dataset

from mimo_music.pipeline import compute_score
from tasks import batched

parse = None
validation_size = 128
split_seed = 42


def score(task, response_text):
    return compute_score("music", response_text)


def rows(split="train"):
    # the dataset only has a train split; "train" and "validation" partition it
    if split not in ("train", "validation"):
        raise ValueError(f"Unknown music split: {split}")
    dataset = load_dataset("XiaomiMiMo/MiMo-V2.6-RL-oss", name="music", split="train")
    dataset = dataset.shuffle(seed=split_seed)
    indices = (range(validation_size) if split == "validation"
               else range(validation_size, len(dataset)))
    dataset = dataset.select(indices)
    yield from dataset


def batches(tokenizer, batch_size=16, *, split="train", limit=None):
    return batched(rows(split), tokenizer, batch_size, limit)
