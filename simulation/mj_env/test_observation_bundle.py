import base64
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from mj_env.observation_bundle import render_bundle, render_decision_bundle, validate_review
from mj_env.test_task_runtime import sample, event


class BundleTests(unittest.TestCase):
    def samples(self, count):
        def encoded(color):
            buffer=io.BytesIO()
            Image.new('RGB',(640,480),color).save(buffer,format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        images={'global':encoded('red'),'wrist':encoded('blue')}
        return [dict(session_id='s',frame_id=i,captured_at_unix_s=i*.05,
                     images_png_base64=dict(images)) for i in range(count)]

    def test_all_frames_paged_and_camera_order(self):
        with tempfile.TemporaryDirectory() as d:
            bundle=render_bundle(self.samples(9),list(range(9)),d)
            self.assertEqual(len(bundle['timeline_pages']),3)
            self.assertEqual([x for p in bundle['timeline_pages'] for x in p['event_ids']],list(range(9)))
            with Image.open(bundle['timeline_pages'][0]['path']) as image:
                self.assertEqual(image.getpixel((100,100)),(255,0,0))
                self.assertEqual(image.getpixel((600,100)),(0,0,255))
            self.assertEqual(bundle['latest_pair']['event_ids'],[8])
            self.assertEqual(bundle['required_event_ids'],[0,3,5,8])
            self.assertTrue(bundle['summary_is_sampled'])
            self.assertEqual(len(bundle['required_images']),2)

    def test_sort_dedupe_missing_and_session(self):
        with tempfile.TemporaryDirectory() as d:
            samples=self.samples(3)
            samples[1]['images_png_base64'].pop('wrist')
            b=render_bundle([samples[2],samples[0],samples[1],samples[1]],[2,0,1,1],d)
            self.assertEqual(b['event_ids'],[0,1,2])
            self.assertEqual(b['missing_images'],[{'event_id':1,'camera':'wrist'}])
            samples[2]['session_id']='other'
            with self.assertRaises(ValueError):render_bundle(samples,[0,1,2],d)

    def test_review_gate(self):
        records=[dict(type='observation_bundle',event_id=8,data={'event_ids':[1,2]})]
        valid=dict(bundle_ref=8,reviewed_event_ids=[1,2],global_findings='target visible',
                   wrist_findings='contact uncertain',process_findings='no movement')
        self.assertEqual(validate_review(records,valid),8)
        for bad in (None,dict(valid,bundle_ref=7),dict(valid,reviewed_event_ids=[2]),dict(valid,wrist_findings='')):
            with self.assertRaises(ValueError):validate_review(records,bad)

    def test_decision_bundle_keeps_process_and_fresh_pair_separate(self):
        images=self.samples(3)
        observations=[dict(sample(10+i*.05),images_png_base64=images[i]['images_png_base64']) for i in range(3)]
        records=[event(1,'task',{}),event(2,'observation',observations[0]),
                 event(3,'observation',observations[1]),
                 event(4,'execution',dict(status='completed',observation_refs=[2,3],terminal_observation_ref=3,robot_stationary=True)),
                 event(5,'observation',observations[2])]
        with tempfile.TemporaryDirectory() as d:
            b=render_decision_bundle(records,5,d)
            self.assertEqual(b['pending_execution_refs'],[4])
            self.assertEqual(b['required_event_ids'],[2,3,5])
            self.assertEqual(b['current_observation_ref'],5)
            self.assertEqual(len(b['required_images']),2)
            self.assertTrue({2,3}.issubset(b['required_images'][0]['event_ids']))
            self.assertEqual(b['processes'][0]['motion_report']['start_observation_ref'],2)
            self.assertEqual(b['processes'][0]['motion_report']['end_observation_ref'],3)
            self.assertNotEqual(b['required_images'][0]['path'],b['required_images'][-1]['path'])
            self.assertEqual(b['required_images'][-1]['event_ids'],[5])
            # Reconnecting to a new session cannot erase the old process evidence.
            observations[-1]['session_id']='new'
            fresh=render_decision_bundle(records,5,d)
            self.assertEqual(fresh['pending_execution_refs'],[4])
            self.assertEqual(fresh['current_session_id'],'new')


if __name__=='__main__':unittest.main()
