import unittest
from unittest.mock import Mock, patch

from tasks import get_task, music


class TaskTests(unittest.TestCase):
    def test_every_task_provides_the_contract(self):
        for name in ("gsm8k", "music"):
            with self.subTest(task=name):
                task = get_task(name)
                self.assertTrue(callable(task.batches) and callable(task.score))
                self.assertTrue(task.parse is None or callable(task.parse))

    def test_music_batches_and_score(self):
        rows = [{"prompt": [{"role": "user", "content": f"Compose {i}"}]} for i in range(3)]
        tokenizer = Mock()
        with patch.object(music, "load_dataset", return_value=rows) as load:
            batches = list(music.batches(tokenizer, batch_size=2, split="train"))
        load.assert_called_once_with("XiaomiMiMo/MiMo-V2.6-RL-oss", name="music", split="train")
        self.assertEqual([[task for task, _ in batch] for batch in batches], [rows[:2], rows[2:]])
        self.assertEqual([c.args for c in tokenizer.apply_chat_template.call_args_list],
                         [(row["prompt"],) for row in rows])
        with patch.object(music, "compute_score", return_value=0.75) as compute:
            self.assertEqual(music.score(rows[0], "X:1\nK:C\nCDEF|"), 0.75)
        compute.assert_called_once_with("music", "X:1\nK:C\nCDEF|")

    def test_batch_size_and_limit_are_validated(self):
        for kwargs in ({"batch_size": 0}, {"limit": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                next(music.batches(Mock(), **kwargs))
