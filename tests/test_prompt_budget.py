import unittest

from subtitle_pipeline.prompt_budget import (
    PromptBudgetExceeded,
    batch_prompt_items,
    estimate_prompt_tokens,
    validate_prompt_budget,
)


class PromptBudgetTests(unittest.TestCase):
    def test_estimator_counts_cjk_as_single_tokens(self):
        self.assertEqual(estimate_prompt_tokens("日本語abcd"), 4)

    def test_validate_reserves_output_capacity(self):
        with self.assertRaises(PromptBudgetExceeded):
            validate_prompt_budget(
                "x" * 9,
                context_size=10,
                max_output_tokens=2,
                estimate_tokens=len,
            )

    def test_batching_uses_rendered_prompt_and_dynamic_output_reserve(self):
        batches = batch_prompt_items(
            ["aaa", "bbb", "ccc"],
            render_prompt=lambda values: "header:" + ",".join(values),
            context_size=16,
            max_output_tokens=lambda count: count,
            estimate_tokens=len,
        )

        self.assertEqual(batches, [("aaa", "bbb"), ("ccc",)])


if __name__ == "__main__":
    unittest.main()
