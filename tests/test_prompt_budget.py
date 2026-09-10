import unittest

from subtitle_pipeline.prompt_budget import (
    PromptBudgetExceeded,
    batch_prompt_items,
    estimate_prompt_tokens,
    validate_prompt_budget,
    validate_request_budget,
    count_llama_prompt_tokens,
    batch_requests,
    fit_optional_text,
)


class PromptBudgetTests(unittest.TestCase):
    def test_complete_request_counts_template_and_reserves_actual_output(self):
        from unittest.mock import Mock
        body = {"messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "text"}], "max_tokens": 4}
        post = Mock(side_effect=[{"prompt": "templated"}, {"tokens": list(range(7))}])
        with self.assertRaises(PromptBudgetExceeded):
            validate_request_budget(body, context_size=10,
                count_tokens=lambda body: count_llama_prompt_tokens(body, post))
        self.assertEqual(post.call_args_list[0].args, ('/apply-template', body))
        self.assertTrue(post.call_args_list[1].args[1]['parse_special'])

    def test_missing_tokenizer_response_does_not_bypass_check(self):
        with self.assertRaises(ValueError):
            count_llama_prompt_tokens({'messages': []}, lambda *_: {})

    def test_optional_context_trim_preserves_mandatory_content(self):
        def render(context):
            return {'max_tokens': 2, 'prompt': 'required:' + context}
        def validate(body):
            validate_request_budget(body, context_size=15, count_tokens=lambda b: len(b['prompt']))
        self.assertEqual(fit_optional_text('abcdefghij', render_request=render, validate_request=validate), 'abcd')
        with self.assertRaises(PromptBudgetExceeded):
            fit_optional_text('', render_request=lambda _: {'max_tokens': 2, 'prompt': 'x'*20}, validate_request=validate)

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
