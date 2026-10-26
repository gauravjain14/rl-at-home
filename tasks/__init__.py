"""Tasks: where prompts come from and how responses are scored.

Each task module provides
  batches(tokenizer, batch_size=16, *, split="train", limit=None)
      yields lists of (task, encoded_prompt); encoded_prompt["input_ids"] is [1, P]
  score(task, response_text) -> float
  parse(response_text) -> the parsed answer or None, or parse = None (evaluation diagnostics)
"""
import importlib


def get_task(name):
    return importlib.import_module(f"tasks.{name}")


def batched(tasks, tokenizer, batch_size, limit):
    """Chat-encode each task's prompt and group into lists of (task, encoded_prompt)."""
    if batch_size < 1 or (limit is not None and limit < 0):
        raise ValueError("Invalid task batch size or limit")
    inputs = []
    for index, task in enumerate(tasks):
        if limit is not None and index >= limit:
            break
        # Task data stays on CPU until the rollout worker consumes it.
        inp_prompt = tokenizer.apply_chat_template(
            task["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs.append((task, inp_prompt))
        if len(inputs) == batch_size:
            yield inputs
            inputs = []
    if inputs:
        yield inputs
