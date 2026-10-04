import contextlib
import io
import json
import unittest
import queue
from unittest.mock import patch

from example.deepseek_client import build_request, sse_events
from example import deepseek_robot as runner


class ClientTests(unittest.TestCase):
    def test_prepare_view_is_bounded_robot_posture_not_object_position(self):
        tool=runner.RobotTools('task_fixed',8)
        tool.latest_bundle=dict(role='decision',bundle_ref=20,current_observation_ref=19,
                                execution_refs=[],required_event_ids=[19])
        tool.latest_robot=dict(aperture_position_m=[.12,0,.29],gripper_width_m=.06)
        tool.robot_metadata=dict(observation_pose_hint=dict(aperture_position_m=[.2,0,.16]))
        args=dict(prepare_view=True,duration_s=1.2,stage='approach',subtask='prepare',
                  objective='view workspace',path_check='clear',expected=['better view'],
                  global_findings='g',wrist_findings='w',process_findings='p')
        with patch.object(runner.subprocess,'run') as process:
            process.return_value.returncode=0;process.return_value.stdout=''
            tool.call('robot_step',args)
            request=json.loads(process.call_args.kwargs['input'])
            target=request['args']['position_m']
            self.assertAlmostEqual(runner.math.dist(target,[.12,0,.29]),.028)
            self.assertGreater(target[0],.12);self.assertLess(target[2],.29)
            self.assertNotIn('prepare_view',request['args'])
        tool.latest_robot['aperture_position_m']=[.2,0,.16]
        with patch.object(runner.subprocess,'run') as process:
            result,_=tool.call('robot_step',args)
            self.assertFalse(result['motion_submitted']);process.assert_not_called()

    def test_contact_and_completion_require_selected_target(self):
        tool=runner.RobotTools('task_fixed',8)
        tool.latest_robot=dict(gripper_width_m=.05,aperture_position_m=[.1,.2,.1])
        tool.latest_bundle=dict(target_tracks={})
        for request in [dict(command='gripper',args=dict(width_m=.03)),
                        dict(command='move',stage='contact',args=dict(position_m=[.1,.2,.09])),
                        dict(result=dict(status='completed'))]:
            with self.subTest(request=request),patch.object(runner.subprocess,'run') as process:
                with self.assertRaises(ValueError):
                    tool.call('task_journal',{'request_json':json.dumps(request)})
                process.assert_not_called()
        # A tracked target permits submission to the existing journal checks;
        # it does not bypass those checks or establish grasp success.
        tool.latest_bundle['target_tracks']={'wrist':{'status':'tracked'}}
        with patch.object(runner.subprocess,'run') as process:
            process.return_value.returncode=0
            process.return_value.stdout=''
            tool.call('task_journal',{'request_json':json.dumps(dict(command='gripper',args=dict(width_m=.03)))})
            process.assert_called_once()

    def test_target_direction_probe_requires_robot_reference(self):
        tool=runner.RobotTools('task_fixed',8)
        tool.latest_robot=dict(aperture_position_m=[.2,0,.16])
        tool.latest_bundle=dict(target_tracks={'global':dict(status='tracked')})
        request=dict(command='move',stage='probe',args=dict(position_m=[.21,0,.16]))
        with patch.object(runner.subprocess,'run') as process:
            with self.assertRaisesRegex(ValueError,'fixed fingertip reference'):
                tool.call('task_journal',{'request_json':json.dumps(request)})
            process.assert_not_called()
            tool.latest_bundle['robot_visual_reference']=dict(status='tracked')
            process.return_value.returncode=0
            process.return_value.stdout=''
            tool.call('task_journal',{'request_json':json.dumps(request)})
            process.assert_called_once()

    def test_identification_is_one_bounded_jaw_segment(self):
        tool=runner.RobotTools('task_fixed',8)
        tool.latest_bundle=dict(role='decision',bundle_ref=20,current_observation_ref=19,
            execution_refs=[],required_event_ids=[19],target_tracks={'global':dict(status='tracked')})
        tool.latest_robot=dict(gripper_width_m=.06,aperture_position_m=[.2,0,.16])
        args=dict(phase='close_probe',path_check='empty gap',global_findings='g',wrist_findings='w',process_findings='p')
        with patch.object(runner.subprocess,'run') as process:
            process.return_value.returncode=0;process.return_value.stdout=''
            tool.call('identify_gripper',args)
            request=json.loads(process.call_args.kwargs['input'])
            self.assertEqual(request['command'],'gripper')
            self.assertAlmostEqual(request['args']['width_m'],.045)
            self.assertNotIn('position_m',request['args'])
            process.assert_called_once()
        with patch.object(runner.subprocess,'run') as process:
            with self.assertRaisesRegex(ValueError,'matching jaw-probe evidence'):
                tool.call('identify_gripper',dict(args,phase='return_probe'))
            process.assert_not_called()

    def test_console_queue_order(self):
        inbox=runner.ConsoleInput.__new__(runner.ConsoleInput)
        inbox.queue=queue.Queue()
        inbox.queue.put('new instruction')
        inbox.queue.put('/quit')
        self.assertEqual(inbox.drain(),['new instruction','/quit'])
        self.assertEqual(inbox.drain(),[])

    def test_input_revision_discards_inflight_model_tool(self):
        config=dict(api_key='test-only',base_url='https://example.invalid',model='test',max_tokens=10,max_rounds=1,max_total_tokens=100)
        tool=dict(id='a',type='function',function=dict(name='robot_info',arguments='{}'))
        inbox=runner.ConsoleInput.__new__(runner.ConsoleInput)
        inbox.queue=queue.Queue()
        with patch.object(runner.httpx,'Client') as client,patch.object(runner.RobotTools,'call') as dispatch,contextlib.redirect_stdout(io.StringIO()):
            response=client.return_value.__enter__.return_value.post.return_value
            response.status_code=200
            def answer():
                inbox.queue.put('new instruction')
                return dict(usage={'total_tokens':5},choices=[dict(finish_reason='tool_calls',message=dict(role='assistant',tool_calls=[tool]))])
            response.json.side_effect=answer
            with self.assertRaisesRegex(ValueError,'Maximum rounds'):
                runner.run(config,'test','offline_test_task_987',inbox)
            dispatch.assert_not_called()
    def test_request_and_sse(self):
        body=build_request({'model':'deepseek-flash','max_tokens':500,'extra_body':{'thinking':{'type':'disabled'}}},'hello')
        self.assertEqual(body['thinking'],{'type':'disabled'})
        self.assertEqual(body['max_tokens'],500)
        self.assertNotIn('reasoning_effort',body)
        self.assertEqual(list(sse_events([': ping','data: {"x":1}','','data: [DONE]',''])),['{"x":1}','[DONE]'])
        with self.assertRaises(ValueError): build_request({'model':'m','extra_body':{'messages':[]}},'x')

    def test_fixed_task_id_and_tool_whitelist(self):
        tools=runner.RobotTools('task_fixed',8)
        with patch.object(runner.subprocess,'run') as process:
            for name,args in [('shell',{}),('view_timeline',{'paths':['C:/secret.png']}),
                              ('task_journal',{'request_json':json.dumps({'task_id':'other'})})]:
                with self.assertRaises(ValueError): tools.call(name,args)
            process.assert_not_called()

    def test_incomplete_or_over_budget_response_never_executes(self):
        config=dict(api_key='test-only',base_url='https://example.invalid',model='test',max_tokens=10,max_rounds=1,max_total_tokens=100)
        tool=dict(id='a',type='function',function=dict(name='task_journal',arguments='{"request_json":"{}"}'))
        for finish,total,calls in [('length',5,[tool]),('tool_calls',100,[tool]),('tool_calls',5,[tool,tool])]:
            with self.subTest(finish=finish,total=total), patch.object(runner.httpx,'Client') as client, \
                 patch.object(runner.RobotTools,'call') as dispatch, contextlib.redirect_stdout(io.StringIO()):
                response=client.return_value.__enter__.return_value.post.return_value
                response.status_code=200
                response.json.return_value=dict(usage={'total_tokens':total},choices=[dict(finish_reason=finish,message=dict(role='assistant',tool_calls=calls))])
                with self.assertRaises(ValueError): runner.run(config,'test','offline_test_task_987')
                dispatch.assert_not_called()

    def test_relative_step_uses_observed_aperture_and_bundle(self):
        tools=runner.RobotTools('task_fixed',8)
        tools.latest_bundle=dict(role='decision',bundle_ref=20,current_observation_ref=19,
                                 execution_refs=[10],required_event_ids=[5,9,19])
        tools.latest_robot=dict(aperture_position_m=[.1,.2,.3],tcp_position_m=[9,9,9])
        args=dict(delta_m=[.01,0,-.01],duration_s=1,stage='approach',subtask='S1',objective='approach',
                  path_check='clear',expected=['closer'],global_findings='g',wrist_findings='w',process_findings='p')
        with patch.object(runner.subprocess,'run') as process:
            process.return_value.returncode=0
            process.return_value.stdout=''
            tools.call('robot_step',args)
            request=json.loads(process.call_args.kwargs['input'])
            self.assertAlmostEqual(request['args']['position_m'][0],.11)
            self.assertAlmostEqual(request['args']['position_m'][2],.29)
            self.assertEqual(request['start_observation_ref'],19)
            self.assertEqual(request['visual_review']['execution_refs'],[10])

    def test_tool_result_images_reach_next_api_request(self):
        config=dict(api_key='test-only',base_url='https://example.invalid',model='test',max_tokens=10,max_rounds=2,max_total_tokens=100)
        tool=dict(id='a',type='function',function=dict(name='task_journal',arguments='{"request_json":"{}"}'))
        with patch.object(runner.httpx,'Client') as client,patch.object(runner.RobotTools,'call') as dispatch,contextlib.redirect_stdout(io.StringIO()):
            response=client.return_value.__enter__.return_value.post.return_value
            response.status_code=200
            response.json.side_effect=[dict(usage={'total_tokens':5},choices=[dict(finish_reason='tool_calls',message=dict(role='assistant',tool_calls=[tool]))]),
                                       dict(usage={'total_tokens':5},choices=[dict(finish_reason='stop',message=dict(role='assistant',content='blocked'))])]
            dispatch.return_value=({'events':[]},[{'type':'image_url','image_url':{'url':'data:image/png;base64,TEST'}}])
            runner.run(config,'test','offline_test_task_987')
            messages=client.return_value.__enter__.return_value.post.call_args.kwargs['json']['messages']
            self.assertEqual(messages[-1]['content'][-1]['type'],'image_url')
            self.assertEqual(messages[-2]['role'],'tool')


if __name__=='__main__': unittest.main()
