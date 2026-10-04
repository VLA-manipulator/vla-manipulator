"""Journal lifecycle tests; no simulator connection or robot commands."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mj_env import task_journal as journal
from mj_env.task_runtime import pending_executions
from mj_env.test_task_runtime import sample


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.logs = self.root / 'tasks'
        self.init = dict(operation='init', task_id='task_a',
                         task=dict(instruction='pick target', criteria=['target held']))

    def run_request(self, request):
        original = journal.open_journal
        sample = {key: None for key in journal.FIELDS}
        sample.update(frame_id=1, session_id='test', captured_at_unix_s=1,
                      snapshot_age_s=0, images_png_base64={'global':'mock','wrist':'mock'})
        with patch.object(journal, 'open_journal', side_effect=lambda r: original(r, self.logs)), \
             patch.object(journal, 'CACHE', self.root / 'preview'), \
             patch.object(journal.sys, 'stdin', io.StringIO(json.dumps(request))), \
             patch.object(journal, 'call', return_value=sample) as call, \
             patch.object(journal, 'render_decision_bundle', side_effect=lambda records, ref, cache:
                          dict(role='decision', current_observation_ref=ref, event_ids=[ref], execution_refs=[])), \
             contextlib.redirect_stdout(io.StringIO()):
            journal.main()
        return call

    def records(self):
        return [json.loads(line) for line in (self.logs / 'task_a.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_multiple_cycles_resume_and_result_share_one_file(self):
        self.run_request(self.init).assert_not_called()
        prefix = (self.logs / 'task_a.jsonl').read_bytes()
        for _ in range(3):
            self.run_request(dict(task_id='task_a'))
        records = self.records()
        review = dict(bundle_ref=records[-1]['event_id'], reviewed_event_ids=records[-1]['data']['event_ids'],
                      global_findings='visible', wrist_findings='visible', process_findings='observed')
        self.run_request(dict(task_id='task_a', visual_review=review, result=dict(status='blocked'))).assert_not_called()
        self.assertEqual(list(self.logs.iterdir()), [self.logs / 'task_a.jsonl'])
        self.assertTrue((self.logs / 'task_a.jsonl').read_bytes().startswith(prefix))
        records = self.records()
        self.assertEqual([r['event_id'] for r in records], list(range(1, len(records)+1)))
        self.assertEqual(records[-1]['type'], 'task_result')
        self.assertEqual(sum(r['type']=='task' for r in records), 1)

    def test_wrong_or_missing_id_never_creates_log(self):
        self.run_request(self.init)
        for request in ({}, dict(task_id='task_a_cycle_2'), dict(task_id='task_b', result={'status':'failed'})):
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.run_request(request)
        self.assertEqual(len(list(self.logs.iterdir())), 1)

    def test_duplicate_init_preserves_history(self):
        self.run_request(self.init)
        self.run_request(dict(task_id='task_a'))
        before = (self.logs / 'task_a.jsonl').read_bytes()
        with self.assertRaises(ValueError):
            self.run_request(self.init)
        self.assertEqual((self.logs / 'task_a.jsonl').read_bytes(), before)

    def test_bad_initialization_creates_nothing(self):
        cases = [dict(self.init, task_id=x) for x in ('../escape', 'C:\\escape', 'NUL', 'nul', '', 'a/b')]
        cases += [dict(self.init, task={}), dict(self.init, task=dict(self.init['task'], task_id='different')),
                  dict(self.init, command='move'), dict(self.init, operation='unknown')]
        for request in cases:
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.run_request(request)
        self.assertFalse(self.logs.exists())

    def test_damaged_log_is_preserved(self):
        self.run_request(self.init)
        path = self.logs / 'task_a.jsonl'
        with path.open('ab') as handle:
            handle.write(b'{"interrupted":')
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            self.run_request(dict(task_id='task_a'))
        self.assertEqual(path.read_bytes(), before)

    def drive(self, request, robot):
        original=journal.open_journal
        def server(_url, command, args=None):
            if command=='info':
                return dict(protocol_version=2,joint_limits_rad=[[-2,2]]*6,max_gripper_width_m=.09)
            if command=='observe':
                s=sample(time.time(),robot.get('x',0))
                s.update(frame_id=time.time_ns(),images_png_base64={'global':'mock','wrist':'mock'},joint_targets_rad=s['joint_positions_rad'])
                s['action']=dict(robot.get('action',dict(status='idle')))
                return s
            robot.setdefault('sent',[]).append(command)
            if command=='move':
                robot['x']=args['position_m'][0]
                robot['action']=dict(status='completed',action_id='test_move')
                return dict(action_id='test_move',status='running')
            raise AssertionError('unexpected robot command '+command)
        def bundle(records,ref,cache):
            executions=pending_executions(records)
            refs=list(dict.fromkeys([r for e in executions for r in e['data']['observation_refs']]+[ref]))
            return dict(role='decision',current_observation_ref=ref,event_ids=refs,
                        required_event_ids=refs,execution_refs=[e['event_id'] for e in executions])
        output=io.StringIO()
        with patch.object(journal,'open_journal',side_effect=lambda r: original(r,self.logs)), \
             patch.object(journal,'CACHE',self.root/'preview'), \
             patch.object(journal.sys,'stdin',io.StringIO(json.dumps(request))), \
             patch.object(journal,'call',side_effect=server), \
             patch.object(journal,'render_decision_bundle',side_effect=bundle), \
             contextlib.redirect_stdout(output):
            journal.main()
        return output.getvalue()

    def command_request(self):
        bundle=next(r for r in reversed(self.records()) if r['type']=='observation_bundle')
        return dict(task_id='task_a',command='move',stage='probe',
                    start_observation_ref=bundle['data']['current_observation_ref'],
                    args=dict(position_m=[.01,0,.3],duration_s=.4),
                    decision=dict(current_subtask='S1'),objective='probe',path='checked',expected=['closer'],
                    visual_review=dict(bundle_ref=bundle['event_id'],reviewed_event_ids=bundle['data']['required_event_ids'],
                                       execution_refs=bundle['data']['execution_refs'],
                                       global_findings='g',wrist_findings='w',process_findings='p'))

    def test_rejected_motion_never_reaches_robot(self):
        self.run_request(self.init)
        robot={}
        self.drive(dict(task_id='task_a'),robot)
        request=self.command_request()
        request['args']['position_m']=[.15,0,.3]
        self.drive(request,robot)
        self.assertEqual(robot.get('sent',[]),[])
        self.assertEqual(self.records()[-1]['type'],'submission_rejected')
        # A valid-sized action from a changed robot pose must also be blocked.
        request['args']['position_m']=[.021,0,.3]
        robot['x']=.02
        self.drive(request,robot)
        self.assertEqual(robot.get('sent',[]),[])
        self.assertIn('differs from reviewed start',self.records()[-1]['data']['reason'])

    def test_execution_resume_and_pending_review_gate(self):
        self.run_request(self.init)
        robot={}
        self.drive(dict(task_id='task_a'),robot)
        self.drive(self.command_request(),robot)
        execution=next(r for r in reversed(self.records()) if r['type']=='execution')
        self.assertEqual(execution['data']['status'],'completed')
        self.assertTrue(execution['data']['motion_report']['terminal_state_confirmed'])
        self.assertEqual(robot['sent'],['move'])
        output=self.drive(dict(task_id='task_a',operation='resume'),robot)
        state=[json.loads(line) for line in output.splitlines() if line.startswith('{')][-1]
        self.assertEqual(state['current_subtask'],'S1')
        self.assertEqual(state['pending_execution_refs'],[execution['event_id']])
        self.assertEqual(robot['sent'],['move'])
        request=self.command_request()
        request['visual_review']['execution_refs']=[]
        request['visual_review']['reviewed_event_ids']=[request['start_observation_ref']]
        self.drive(request,robot)
        self.assertEqual(robot['sent'],['move'])
        self.assertEqual(self.records()[-1]['type'],'submission_rejected')

    def test_another_sender_lock_blocks_submission(self):
        self.run_request(self.init)
        robot={}
        self.drive(dict(task_id='task_a'),robot)
        with (self.root/'preview'/'controller.lock').open('a+b') as sender:
            journal.lock_exclusive(sender)
            self.drive(self.command_request(),robot)
        self.assertEqual(robot.get('sent',[]),[])
        self.assertIn('another journal sender',self.records()[-1]['data']['reason'])

    def test_instruction_revision_stays_in_same_log(self):
        self.run_request(self.init)
        self.run_request(dict(task_id='task_a',task_update=dict(instruction='move target',source='user_console'))).assert_not_called()
        from mj_env.task_runtime import task_state
        self.assertTrue(task_state(self.records())['task']['criteria_pending'])
        self.run_request(dict(task_id='task_a',task_update=dict(instruction='move target',criteria=['target at destination']))).assert_not_called()
        state=task_state(self.records())
        self.assertFalse(state['task']['criteria_pending'])
        self.assertEqual(state['task']['instruction'],'move target')
        self.assertEqual(len(list(self.logs.iterdir())),1)


if __name__ == '__main__':
    unittest.main()
