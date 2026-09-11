import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from subtitle_pipeline.cache import CacheStore, STAGES, config_snapshot, job_lock
from subtitle_pipeline.config import LLMConfig, UploadConfig
from subtitle_pipeline.publication import publish_once, read_record, resolve
from subtitle_pipeline.upload import BilibiliSubmission, BiliupCommandError, UploadNotStartedError, upload_to_bilibili


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'cache.sqlite3'
        self.store = CacheStore(self.path)

    def test_config_and_evidence_changes_keep_original_plan_and_result(self):
        stage = self.store.stage('translation', lambda: {'model': 'first', 'chat': 'old'})
        stage.put('0', 'original', source='local_mt', reason='timeout')
        restored = CacheStore(self.path).stage('translation', lambda: self.fail('must not prepare new inputs'))
        self.assertEqual(restored.plan['model'], 'first')
        self.assertEqual(restored.get('0'), 'original')
        self.assertEqual(restored.record('0')['source'], 'local_mt')
        self.assertEqual(restored.record('0')['reason'], 'timeout')

    def test_version_change_invalidates_descendants_but_not_upstream_or_songs(self):
        for name in ('raw_speech', 'song_identification', 'asr_correction', 'speech_alignment', 'translation', 'render'):
            self.store.stage(name, lambda: {}).finish(name)
        definition = STAGES['asr_correction']
        with patch.dict(STAGES, asr_correction=replace(definition, version=definition.version + 1)):
            store = CacheStore(self.path)
            for name in ('asr_correction', 'speech_alignment', 'translation', 'render'):
                self.assertIsNone(store.existing(name))
            for name in ('raw_speech', 'song_identification'):
                self.assertEqual(store.existing(name).get('__result__'), name)

    def test_reset_bumps_generation_and_preserves_publication(self):
        stage = self.store.stage('translation', lambda: {})
        stage.finish([])
        self.store.stage('render', lambda: {}).finish('video')
        manifest = self.root / 'manifest.json'
        manifest.write_text('{"uploaded":true}')
        self.store.reset('translation')
        self.assertIsNone(self.store.existing('render'))
        self.assertEqual(json.loads(manifest.read_text()), {'uploaded': True})
        self.assertEqual(next(row for row in self.store.status() if row['name'] == 'translation')['generation'], 2)

    def test_retry_only_degraded_units_and_clear_aggregates(self):
        stage = self.store.stage('translation', lambda: {'original': 1})
        stage.put('0', 'good')
        stage.put('1', 'old', source='local_mt', reason='empty')
        stage.put('group', ['good', 'old'], kind='aggregate')
        stage.finish(['good', 'old'])
        self.store.stage('render', lambda: {}).finish('oldvideo')
        self.assertEqual(self.store.retry_degraded('translation'), 1)
        self.assertEqual(stage.get('0'), 'good')
        self.assertIsNone(stage.get('1'))
        self.assertEqual(stage.record('1', include_retry=True)['payload'], 'old')
        self.assertIsNone(stage.get('group'))
        self.assertIsNone(self.store.existing('render'))
        stage.put('1', 'still fallback', source='local_mt', reason='empty')
        self.assertEqual(CacheStore(self.path).existing('translation').get('1'), 'still fallback')

    def test_failed_manual_retry_keeps_old_result_for_next_normal_run(self):
        stage = self.store.stage('translation', lambda: {})
        stage.put('1', 'fallback', reason='timeout')
        self.store.retry_degraded('translation')
        stage.failed('1', RuntimeError('offline'))
        self.assertEqual(CacheStore(self.path).existing('translation').get('1'), 'fallback')

    def test_concurrent_unit_commits_survive_interruption(self):
        stage = self.store.stage('segmentation', lambda: {'windows': list(range(24))})
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda index: stage.put(str(index), {'value': index}), range(24)))
        restored = CacheStore(self.path).existing('segmentation')
        self.assertIsNone(restored.get('__result__'))
        self.assertEqual([restored.get(str(index))['value'] for index in range(24)], list(range(24)))

    def test_unserializable_result_is_not_committed(self):
        stage = self.store.stage('translation', lambda: {})
        with self.assertRaises(TypeError):
            stage.put('broken', object())
        self.assertIsNone(stage.get('broken'))

    def test_snapshots_omit_credential_locations(self):
        config = LLMConfig(api_key_pass_entry='private/store', api_key_env='SECRET')
        snapshot = config_snapshot(config)
        self.assertNotIn('api_key_pass_entry', snapshot)
        self.assertNotIn('api_key_env', snapshot)
        self.assertIn('model', snapshot)

    def test_second_process_cannot_lock_same_job_and_lock_releases(self):
        code = "from pathlib import Path; from subtitle_pipeline.cache import job_lock; import sys\nwith job_lock(Path(sys.argv[1])): pass"
        with job_lock(self.root):
            child = subprocess.run([sys.executable, '-c', code, str(self.root)], capture_output=True, text=True)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn('already running', child.stderr)
        child = subprocess.run([sys.executable, '-c', code, str(self.root)], capture_output=True, text=True)
        self.assertEqual(child.returncode, 0, child.stderr)

    def test_publish_success_is_durable_before_comment_and_not_repeated(self):
        path = self.root / 'manifest.json'
        def submit():
            self.assertEqual(read_record(path)['status'], 'submitting')
            return BilibiliSubmission(12, 'BVexample', 'success')
        publish_once(path, {'title': 'test'}, submit)
        self.store.reset()
        restored = publish_once(path, {}, lambda: self.fail('already uploaded'))
        self.assertEqual(restored.aid, 12)
        self.assertTrue(read_record(path)['uploaded'])

    def test_rejected_publication_can_be_retried(self):
        path = self.root / 'manifest.json'
        error = BiliupCommandError(1, 'ResponseData { code: 21021, data: None, message: "missing source" }')
        with self.assertRaises(BiliupCommandError):
            publish_once(path, {}, Mock(side_effect=error))
        self.assertEqual(read_record(path)['status'], 'not_uploaded')
        publish_once(path, {}, lambda: BilibiliSubmission(12, 'BVexample', 'ok'))
        self.assertEqual(read_record(path)['status'], 'success')

    def test_preparation_and_process_start_failures_allow_retry(self):
        path = self.root / 'manifest.json'
        cookie = self.root / 'cookies.json'
        config = UploadConfig(cookie_file=str(cookie),
                              pause_marker_file=str(self.root / 'paused.json'),
                              throttle_state_file=str(self.root / 'throttle.json'))
        def submit():
            return upload_to_bilibili(self.root / 'video.mp4', title='test',
                                     description='', source_url='https://example.com/video',
                                     tags=['test'], config=config)
        with patch('subtitle_pipeline.upload.require_command', return_value='biliup'), \
             patch('subtitle_pipeline.upload.subprocess.Popen', side_effect=OSError('cannot start')) as popen:
            with self.assertRaises(UploadNotStartedError):
                publish_once(path, {}, submit)
            popen.assert_not_called()
            self.assertEqual(read_record(path)['status'], 'not_uploaded')
            cookie.write_text('{}')
            with self.assertRaises(UploadNotStartedError):
                publish_once(path, {}, submit)
            popen.assert_called_once()
            self.assertEqual(read_record(path)['status'], 'not_uploaded')
        publish_once(path, {}, lambda: BilibiliSubmission(12, 'BVexample', 'ok'))
        self.assertEqual(read_record(path)['status'], 'success')

    def test_transport_failure_and_conflicting_responses_remain_unknown(self):
        for output in ('connection timed out',
                       'ResponseData { code: 0, data: Some(x) } ResponseData { code: 21021, data: None }'):
            with self.subTest(output=output):
                path = self.root / 'manifest.json'
                path.unlink(missing_ok=True)
                with self.assertRaises(BiliupCommandError):
                    publish_once(path, {}, Mock(side_effect=BiliupCommandError(1, output)))
                self.assertEqual(read_record(path)['status'], 'unknown')
                with self.assertRaisesRegex(RuntimeError, 'unknown'):
                    publish_once(path, {}, lambda: self.fail('must not upload'))

    def test_unknown_publication_requires_resolution(self):
        path = self.root / 'manifest.json'
        with self.assertRaises(KeyboardInterrupt):
            publish_once(path, {}, Mock(side_effect=KeyboardInterrupt))
        with self.assertRaisesRegex(RuntimeError, 'unknown'):
            publish_once(path, {}, lambda: self.fail('must not upload'))
        resolve(path, uploaded=False, aid=None, bvid=None)
        publish_once(path, {}, lambda: BilibiliSubmission(12, 'BVexample', 'ok'))
        self.assertEqual(read_record(path)['status'], 'success')

    def test_manual_success_and_existing_success_records_are_respected(self):
        path = self.root / 'manifest.json'
        resolve(path, uploaded=True, aid=123, bvid='BVtest')
        publish_once(path, {}, lambda: self.fail('must not upload'))
        path.write_text('{"uploaded":true,"bilibili_aid":456,"bilibili_bvid":"BVold"}')
        self.assertEqual(publish_once(path, {}, lambda: self.fail('must not upload')).aid, 456)

    def test_unreadable_publication_never_becomes_a_cache_miss(self):
        path = self.root / 'manifest.json'
        path.write_text('{')
        with self.assertRaisesRegex(RuntimeError, 'unreadable'):
            publish_once(path, {}, lambda: self.fail('must not upload'))
