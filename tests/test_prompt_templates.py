import unittest

from subtitle_pipeline.prompt_templates import (
    load_prompt_template,
    prompt_templates_digest,
    render_user_prompt,
)


class PromptTemplateTests(unittest.TestCase):
    def test_runtime_prompt_documents_have_system_and_user_sections(self):
        for name in (
            "cue-planner.md",
            "cue-boundary-repair.md",
            "fixed-translation.md",
        ):
            with self.subTest(name=name):
                template = load_prompt_template(name)
                self.assertTrue(template.system)
                self.assertTrue(template.user)

    def test_render_requires_exact_placeholder_values(self):
        with self.assertRaisesRegex(RuntimeError, "missing="):
            render_user_prompt("fixed-translation.md")
        with self.assertRaisesRegex(RuntimeError, "unexpected=EXTRA"):
            render_user_prompt(
                "fixed-translation.md",
                TARGET_LANGUAGE="Simplified Chinese",
                HONORIFIC_TRANSLATION_RULES="rules",
                REFERENCE_TEXT="<terms>\nsource=>target",
                MAXIMUM_UNITS="20.000",
                TARGET_TEXT="<unknown>\n<0>source",
                RETRY_SECTION="",
                EXTRA="value",
            )

    def test_digest_covers_runtime_sections(self):
        digest = prompt_templates_digest(
            "cue-planner.md", "cue-boundary-repair.md"
        )
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(digest, prompt_templates_digest("fixed-translation.md"))


if __name__ == "__main__":
    unittest.main()
