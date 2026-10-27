import unittest
from unittest.mock import Mock, patch

from datasets import Dataset

from tasks import get_task, music


class TaskTests(unittest.TestCase):
    def test_every_task_provides_the_contract(self):
        for name in ("gsm8k", "music"):
            with self.subTest(task=name):
                task = get_task(name)
                self.assertTrue(callable(task.batches) and callable(task.score))
                self.assertTrue(task.parse is None or callable(task.parse))

    def test_music_batches_splits_and_score(self):
        rows = [{"prompt": [{"role": "user", "content": f"Compose {i}"}]} for i in range(131)]
        tokenizer = Mock()
        with patch.object(music, "load_dataset", return_value=Dataset.from_list(rows)) as load:
            train = [task for batch in music.batches(tokenizer, batch_size=2) for task, _ in batch]
            validation = [task for batch in music.batches(tokenizer, split="validation") for task, _ in batch]
            repeated = [task for batch in music.batches(tokenizer, split="validation") for task, _ in batch]
            with self.assertRaises(ValueError):
                next(music.batches(tokenizer, split="test"))
        load.assert_called_with("XiaomiMiMo/MiMo-V2.6-RL-oss", name="music", split="train")
        # the only split is partitioned: 128 held out, the rest trains, fixed across calls
        self.assertEqual((len(train), len(validation)), (3, 128))
        contents = lambda tasks: {task["prompt"][0]["content"] for task in tasks}  # noqa: E731
        self.assertFalse(contents(train) & contents(validation))
        self.assertEqual(contents(train) | contents(validation), contents(rows))
        self.assertEqual(repeated, validation)
        self.assertEqual(tokenizer.apply_chat_template.call_args_list[0].args, (train[0]["prompt"],))
        with patch.object(music, "compute_score", return_value=0.75) as compute:
            self.assertEqual(music.score(rows[0], "X:1\nK:C\nCDEF|"), 0.75)
        compute.assert_called_once_with("music", "X:1\nK:C\nCDEF|")

    def test_batch_size_and_limit_are_validated(self):
        for kwargs in ({"batch_size": 0}, {"limit": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                next(music.batches(Mock(), **kwargs))
