import unittest
from unittest.mock import Mock, patch
from datasets import Dataset

from tasks import gsm8k


class GSM8KTests(unittest.TestCase):
    def test_answer_formats_numeric_equality_and_wrong_or_missing_answers(self):
        for response, answer, expected in (
            ("Steps...\n#### 42", "42", 1.0),
            (r"Therefore \(\boxed{42}\) people.", "42", 1.0),
            (r"\boxed{1,200.00}", "1200", 1.0),
            ("#### -0.5", "-.50", 1.0),
            ("#### $0", "0", 1.0),
            ("#### 42 dollars", "42", 1.0),
            ("#### **42**", "42", 1.0),
            (r"\boxed{\$42}", "42", 1.0),
            (r"\boxed{4}", "36", 0.0),
            ("The answer is 42", "42", 0.0),
            (r"42 appears in the reasoning. \boxed{4}", "42", 0.0),
            (r"\boxed{42} corrected to #### 4", "42", 0.0),
            (r"\boxed{42} corrected to \boxed{unknown}", "42", 0.0),
            (r"\boxed{42", "42", 0.0),
        ):
            with self.subTest(response=response):
                self.assertEqual(gsm8k.score({"answer": answer}, response), expected)
        with self.assertRaises(ValueError):
            gsm8k.score({"answer": "invalid"}, "#### 42")

    def test_task_batches_keep_ground_truth_out_of_model_prompt(self):
        rows = [{"question": f"Question {i}?", "answer": "Reasoning. #### 9876"} for i in range(131)]
        dataset = Dataset.from_list(rows)
        tokenizer = Mock()
        with patch.object(gsm8k, "load_dataset", return_value=dataset) as load, \
                patch("builtins.print"):
            batches = list(gsm8k.batches(tokenizer, batch_size=2))
            self.assertEqual([len(batch) for batch in batches], [2, 1])
            load.assert_called_once_with("openai/gsm8k", "main", split="train")
            train_ids = set()
            for i, (task, _) in enumerate(entry for batch in batches for entry in batch):
                train_ids.add(task["question_id"])
                self.assertEqual(task["answer"], "9876")
                self.assertIn(rows[task["question_id"]]["question"], task["prompt"][0]["content"])
                self.assertNotIn("9876", task["prompt"][0]["content"])
                self.assertEqual(tokenizer.apply_chat_template.call_args_list[i].args, (task["prompt"],))
            validation = [task for batch in gsm8k.batches(tokenizer, split="validation") for task, _ in batch]
            validation_ids = {task["question_id"] for task in validation}
            self.assertEqual(len(validation_ids), 128)
            self.assertFalse(train_ids & validation_ids)
            self.assertEqual(train_ids | validation_ids, set(range(131)))
            repeated = [task for batch in gsm8k.batches(tokenizer, split="validation") for task, _ in batch]
            self.assertEqual(repeated, validation)
            self.assertEqual(load.call_args.kwargs["split"], "train")
            held_out = list(gsm8k.batches(tokenizer, split="test", limit=1))
            self.assertEqual(len(held_out[0]), 1)
            self.assertEqual(load.call_args.kwargs["split"], "test")
