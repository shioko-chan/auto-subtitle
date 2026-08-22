import unittest

from subtitle_pipeline.prompt_templates import (
    load_prompt_template,
    prompt_templates_digest,
    render_user_prompt,
)


class PromptTemplateTests(unittest.TestCase):
    def test_runtime_prompt_documents_have_system_and_user_sections(self):
        for name in ("joint-segment-translate.md",):
            with self.subTest(name=name):
                template = load_prompt_template(name)
                self.assertTrue(template.system)
                self.assertTrue(template.user)
                self.assertIn("dependent Japanese particle", template.user)

    def test_render_requires_exact_placeholder_values(self):
        with self.assertRaisesRegex(RuntimeError, "missing="):
            render_user_prompt("joint-segment-translate.md")
        with self.assertRaisesRegex(RuntimeError, "unexpected=EXTRA"):
            render_user_prompt(
                "joint-segment-translate.md",
                TARGET_LANGUAGE="Simplified Chinese",
                HONORIFIC_TRANSLATION_RULES="rules",
                REFERENCE_TEXT="<terms>\nsource=>target",
                MAXIMUM_UNITS="20.000",
                DIALOGUE_CONTEXT="(none)",
                TARGET_TEXT="<unknown>\n<0>source",
                RETRY_SECTION="",
                EXTRA="value",
            )

    def test_digest_covers_runtime_sections(self):
        digest = prompt_templates_digest("joint-segment-translate.md")
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, prompt_templates_digest("joint-segment-translate.md"))


if __name__ == "__main__":
    unittest.main()
