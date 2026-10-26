from datasets import load_dataset

from mimo_music.pipeline import compute_score
from tasks import batched

parse = None


def score(task, response_text):
    return compute_score("music", response_text)


def rows(split="train"):
    yield from load_dataset("XiaomiMiMo/MiMo-V2.6-RL-oss", name="music", split=split)


def batches(tokenizer, batch_size=16, *, split="train", limit=None):
    return batched(rows(split), tokenizer, batch_size, limit)
