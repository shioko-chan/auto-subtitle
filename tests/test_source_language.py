import unittest

from subtitle_pipeline.source_language import (
    combine_source_languages,
    join_source_fragments,
    language_for_text,
)
from subtitle_pipeline.local_translation import _m2m_source_language


class SourceLanguageTests(unittest.TestCase):
    def test_detects_japanese_english_and_mixed_text(self):
        self.assertEqual(language_for_text("今日は"), "Japanese")
        self.assertEqual(language_for_text("Ready set"), "English")
        self.assertEqual(language_for_text("今日は Ready"), "mixed")

    def test_combines_distinct_languages_as_mixed(self):
        self.assertEqual(
            combine_source_languages(["Japanese", "English"]), "mixed"
        )

    def test_english_fragments_keep_spaces_and_attach_punctuation(self):
        self.assertEqual(
            join_source_fragments(
                [
                    ("Ready", "English"),
                    ("set", "English"),
                    ("and", "English"),
                    ("find", "English"),
                    ("out", "English"),
                    ("!", "English"),
                ]
            ),
            "Ready set and find out!",
        )

    def test_local_translation_selects_source_language(self):
        self.assertEqual(_m2m_source_language("Ready", "English"), "en")
        self.assertEqual(_m2m_source_language("今日は", "Japanese"), "ja")
        self.assertEqual(_m2m_source_language("Ready", "mixed"), "en")


if __name__ == "__main__":
    unittest.main()
