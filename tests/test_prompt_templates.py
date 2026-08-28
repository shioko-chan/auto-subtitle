import unittest

from subtitle_pipeline.prompt_templates import (
    load_prompt_template,
    prompt_templates_digest,
    render_user_prompt,
)


class PromptTemplateTests(unittest.TestCase):
    def test_runtime_prompt_documents_have_system_and_user_sections(self):
        for name in (
            "asr-correct.md",
            "segment-source-cues.md",
            "translate-fixed-cues.md",
            "review-fixed-translations.md",
            "lyrics-translate.md",
            "lyrics-review.md",
            "metadata-translate.md",
        ):
            with self.subTest(name=name):
                template = load_prompt_template(name)
                self.assertTrue(template.system)
                self.assertTrue(template.user)

    def test_render_requires_exact_placeholder_values(self):
        with self.assertRaisesRegex(RuntimeError, "missing="):
            render_user_prompt("segment-source-cues.md")
        with self.assertRaisesRegex(RuntimeError, "unexpected=EXTRA"):
            render_user_prompt(
                "segment-source-cues.md",
                SOURCE_MAXIMUM_UNITS="20.000",
                SOURCE_LANGUAGE="English",
                DIALOGUE_CONTEXT="(none)",
                TARGET_TEXT="<unknown>\n<0>source",
                RETRY_SECTION="",
                EXTRA="value",
            )

    def test_digest_covers_runtime_sections(self):
        digest = prompt_templates_digest("segment-source-cues.md")
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, prompt_templates_digest("segment-source-cues.md"))


if __name__ == "__main__":
    unittest.main()
