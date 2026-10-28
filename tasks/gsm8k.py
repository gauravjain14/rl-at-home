import re
from decimal import Decimal

from datasets import load_dataset

from tasks import batched

validation_size = 128
split_seed = 42

_MARKER = re.compile(r"####|\\boxed\s*\{")
_NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)")


def parse_number(text):
    # First number after the marker, so "18 dollars", "**18**" and "\$18" all count.
    match = _NUMBER.search(text)
    if not match:
        return None
    return Decimal(match.group().replace(",", ""))


def parse(response_text):
    markers = list(_MARKER.finditer(response_text))
    if not markers:
        return None
    # Use the last designated answer, never a number from the reasoning.
    marker = markers[-1]
    tail = response_text[marker.end():].lstrip()
    if marker.group() == "####":
        answer = tail.splitlines()[0] if tail else ""
    else:
        answer, closing, _ = tail.partition("}")
        if not closing:
            return None
    return parse_number(answer)


def score(task, response_text):
    expected = parse_number(task["answer"])
    if expected is None:
        raise ValueError("Invalid GSM8K ground-truth answer")
    return float(parse(response_text) == expected)


def rows(split="train"):
    source_split = "train" if split == "validation" else split
    dataset = load_dataset("openai/gsm8k", "main", split=source_split)
    dataset = dataset.add_column("question_id", list(range(len(dataset))))
    if split in ("train", "validation"):
        if len(dataset) <= validation_size:
            raise ValueError("Validation set must leave at least one training question")
        dataset = dataset.shuffle(seed=split_seed)
        indices = (range(validation_size) if split == "validation"
                   else range(validation_size, len(dataset)))
        dataset = dataset.select(indices)
    # apply expected chat format on the dataset
    for row in dataset:
        # just maintains music-style task formating
        # I don't think this approach is scalable though
        task = {
            "question_id": row["question_id"],
            "prompt": [{
                "role": "user",
                "content": (
                    row["question"]
                    + "\nSolve step by step. On the last line, write #### followed by your numerical answer."
                )
            }],
            "answer": row["answer"].rsplit("####", 1)[-1].strip(),
        }
        yield task


def batches(tokenizer, batch_size=16, *, split="train", limit=None):
    return batched(rows(split), tokenizer, batch_size, limit)
