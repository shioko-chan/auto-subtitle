from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.fan_knowledge import FanKnowledgeRetriever, KnowledgeQuery, KnowledgeRecord
from subtitle_pipeline.phonetic_index import match_readings


class PhoneticIndexTests(unittest.TestCase):
    def record(self):
        return KnowledgeRecord('term:club', 'term', '筋トレ部', '健身部', reading='きんとれぶ', reliability=0.98)

    def test_persisted_forms_reused_across_reopen_and_batch_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'knowledge.sqlite3'
            with FanKnowledgeRetriever(path) as retriever:
                retriever.upsert([self.record()])
            with patch('subtitle_pipeline.fan_knowledge._term_phonetic_forms', side_effect=AssertionError('must use saved forms')):
                with FanKnowledgeRetriever(path) as retriever:
                    rows = retriever.retrieve_asr_term_references_many([
                        KnowledgeQuery('昨日金トレプで話した'), KnowledgeQuery(''), KnowledgeQuery('今日は雨です'),
                        KnowledgeQuery('筋トレ部', top_k=0),
                    ])
            self.assertEqual([hit.record_id for hit in rows[0]], ['term:club'])
            self.assertEqual(rows[1:], [[], [], []])
            match = rows[0][0].phonetic_match
            self.assertGreater(match['end'], match['start'])
            self.assertEqual(match['reading'], 'きんとれぶ')

    def test_updates_delete_stale_forms_and_non_term_kind(self):
        with tempfile.TemporaryDirectory() as directory, FanKnowledgeRetriever(Path(directory) / 'knowledge.sqlite3') as retriever:
            record = self.record()
            retriever.upsert([record])
            retriever.upsert([replace(record, title='天文同好会', reading='てんもんどうこうかい')])
            self.assertEqual(retriever.retrieve_asr_term_references(KnowledgeQuery('きんとれぷ')), [])
            self.assertEqual(len(retriever.retrieve_asr_term_references(KnowledgeQuery('てんもんどうこうかい'))), 1)
            retriever.upsert([replace(record, kind='community_term')])
            self.assertEqual(retriever._database.execute('SELECT COUNT(*) FROM knowledge_term_pronunciations').fetchone()[0], 0)
            retriever.upsert([record])
            retriever._database.execute('DELETE FROM knowledge_records WHERE record_id=?', (record.record_id,))
            self.assertEqual(retriever._database.execute('SELECT COUNT(*) FROM knowledge_term_pronunciations').fetchone()[0], 0)

    def test_rule_version_rebuilds_derived_readings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'knowledge.sqlite3'
            with FanKnowledgeRetriever(path) as retriever:
                retriever.upsert([self.record()])
                old = retriever._database.execute('SELECT fingerprint FROM knowledge_term_pronunciations').fetchone()[0]
            with patch('subtitle_pipeline.phonetic_index.PHONETIC_VERSION', 'new-rule'), patch(
                'subtitle_pipeline.fan_knowledge._term_phonetic_forms', return_value=('しんはつおん',)
            ) as build:
                with FanKnowledgeRetriever(path) as retriever:
                    row = retriever._database.execute('SELECT fingerprint,forms_json FROM knowledge_term_pronunciations').fetchone()
                    self.assertNotEqual(row['fingerprint'], old)
                    self.assertIn('しんはつおん', row['forms_json'])
                build.assert_called_once()

    def test_native_cutoff_retains_one_kana_error_without_short_query_false_match(self):
        values = match_readings(['きのうきんとれぷではなした', 'あしたてんきがいい', 'きんと'], ['きんとれぶ'])
        self.assertAlmostEqual(float(values[0, 0]), 0.8, places=5)
        self.assertEqual(float(values[1, 0]), 0)
        self.assertEqual(float(values[2, 0]), 0)

    def test_long_titles_do_not_match_generic_fragments_or_punctuation(self):
        with tempfile.TemporaryDirectory() as directory, FanKnowledgeRetriever(Path(directory) / 'knowledge.sqlite3') as retriever:
            retriever.upsert([
                KnowledgeRecord('term:song', 'term', 'これはぼくたちの生存のあらすじ', 'song'),
                KnowledgeRecord('term:live', 'term', 'BanG Dream! 10th Anniversary LIVE「In the name of BanG Dream!」', 'live'),
            ])
            values = retriever.retrieve_asr_term_references_many([
                KnowledgeQuery('どんなに愛しても僕たちを中心に回ってる'),
                KnowledgeQuery('Over the lights up見える景色を期待した'),
            ])
            self.assertEqual(values, [[], []])
            saved = [row[0] for row in retriever._database.execute('SELECT forms_json FROM knowledge_term_pronunciations')]
            self.assertNotIn('きごう', ''.join(saved))

    def test_empty_form_rows_preserve_shape(self):
        self.assertEqual(match_readings(['あいうえお'], []).shape, (1, 0))
        self.assertEqual(match_readings([], ['あいうえお']).shape, (0, 1))
