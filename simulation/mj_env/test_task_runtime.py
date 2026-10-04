import unittest
from mj_env.task_runtime import motion_report, pending_executions, preflight, task_state, validate_submission


def sample(t=10., x=0., q=0.):
    return dict(session_id='s', frame_id=round(t*100), captured_at_unix_s=t, snapshot_age_s=.01,
                aperture_position_m=[x,0.,.3], tcp_position_m=[x,0.,.28],
                joint_positions_rad=[q,0.,0.,0.,0.,0.], gripper_width_m=.04,
                action=dict(status='idle'))


def event(i, kind, data):
    return dict(event_id=i, task_id='test', cycle=i, type=kind, data=data)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.records=[event(1,'task',dict(instruction='pick',criteria=['held'],retry_budget=3))]
        self.request=dict(command='move', args=dict(position_m=[.01,0,.3],duration_s=1),
                          stage='probe', decision=dict(current_subtask='S1'),objective='approach',
                          path='visually checked',expected=['closer'])
        self.info=dict(protocol_version=2,joint_limits_rad=[[-2,2]]*6,max_gripper_width_m=.09)

    def check(self, request=None, actual=None, expected=None):
        return preflight(request or self.request,expected or sample(),actual or sample(),
                         self.info,task_state(self.records),now=10.05)

    def test_valid_request_and_limits(self):
        self.assertTrue(self.check()['passed'])
        for args in (dict(position_m=[.15,0,.3],duration_s=1),
                     dict(position_m=[.02,0,.3],duration_s=.05),
                     dict(position_m=[float('nan'),0,.3],duration_s=1),
                     dict(position_m=[.01,0,.3],duration_s=True),
                     dict(position_m=[.01,0,.3],duration_s=1,hinge_yaw_rad=1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.check(dict(self.request,args=args))

    def test_start_drift_session_action_and_freshness(self):
        changes=[dict(aperture_position_m=[.02,0,.3]),dict(joint_positions_rad=[.1]*6),
                 dict(session_id='other'),dict(action=dict(status='running')),
                 dict(action=dict(status='completed',action_id='external')),
                 dict(captured_at_unix_s=9),dict(snapshot_age_s=.3),dict(gripper_width_m=.05)]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                self.check(actual=dict(sample(),**change))

    def test_joint_and_gripper_limits(self):
        self.assertTrue(self.check(dict(self.request,command='joints',args=dict(positions_rad=[.05]*6,duration_s=1)))['passed'])
        self.assertTrue(self.check(dict(self.request,command='gripper',args=dict(width_m=.05,duration_s=1)))['passed'])
        for cmd,args in [('joints',dict(positions_rad=[.2]*6,duration_s=1)),
                         ('joints',dict(positions_rad=[4]*6,duration_s=1)),
                         ('gripper',dict(width_m=.09,duration_s=1))]:
            with self.subTest(command=cmd),self.assertRaises(ValueError):
                self.check(dict(self.request,command=cmd,args=args))

    def test_refresh_does_not_clear_pending_review(self):
        self.records.extend([event(2,'execution',dict(status='completed',observation_refs=[10,11])),
                             event(3,'observation_bundle',dict(role='decision',event_ids=[12],
                                                              current_observation_ref=12,execution_refs=[]))])
        review=dict(bundle_ref=3,reviewed_event_ids=[12],global_findings='g',wrist_findings='w',process_findings='p')
        request=dict(command='move',start_observation_ref=12,visual_review=review)
        with self.assertRaises(ValueError):
            validate_submission(request,self.records,self.records[-1]['data'])
        self.records[-1]['data'].update(execution_refs=[2],required_event_ids=[10,11,12])
        review.update(reviewed_event_ids=[10,11,12],execution_refs=[2])
        self.assertEqual(validate_submission(request,self.records,self.records[-1]['data']),[2])
        self.records.append(event(4,'visual_review',dict(review,verified_execution_refs=[2])))
        self.assertEqual(pending_executions(self.records),[])

    def test_history_review_cannot_authorize_command(self):
        self.records.append(event(2,'observation_bundle',dict(role='history',event_ids=[10])))
        review=dict(bundle_ref=2,reviewed_event_ids=[10],global_findings='g',wrist_findings='w',process_findings='p')
        with self.assertRaises(ValueError):
            validate_submission(dict(command='move',visual_review=review),self.records,self.records[-1]['data'])

    def test_state_failure_dedup_and_unknown_command(self):
        self.records.extend([event(2,'command',{}),event(3,'execution',dict(status='aborted')),
                             event(4,'decision',dict(current_subtask='S1',failure={'reason':'aborted'},transition_reason='retry'))])
        state=task_state(self.records)
        self.assertEqual(state['failure_count'],1)
        self.assertEqual(state['pending_execution_refs'],[3])
        self.assertEqual(state['current_subtask'],'S1')
        self.assertFalse(state['unresolved_command'])
        self.records.append(event(5,'command',{}))
        self.assertTrue(task_state(self.records)['unresolved_command'])
        with self.assertRaises(ValueError): self.check()

    def test_measured_path_deviation_and_actual_endpoints(self):
        samples=[sample(10),sample(10.1,.02,.1),sample(10.2)]
        report=motion_report(samples,[51,63,78],dict(robot_stationary=True,terminal_observation_ref=78))
        self.assertEqual(report['start_observation_ref'],51)
        self.assertEqual(report['end_observation_ref'],78)
        self.assertEqual(report['endpoint_distance_m'],0)
        self.assertAlmostEqual(report['max_sampled_deviation_from_endpoint_line_m'],.02)
        self.assertAlmostEqual(report['joint_positions_rad']['max_excursion'][0],.1)
        self.assertIsNone(report['orientation_change_rad'])
        self.assertTrue(report['terminal_state_confirmed'])
        samples[-1]['session_id']='different'
        self.assertNotIn('endpoint_distance_m',motion_report(samples,[51,63,78]))

    def test_completed_task_and_exhausted_budget_block_motion(self):
        self.records.append(event(2,'task_result',dict(status='completed')))
        with self.assertRaisesRegex(ValueError,'already ended'): self.check()
        self.records.pop()
        self.records[0]['data']['retry_budget']=0
        self.records.append(event(2,'execution',dict(status='aborted')))
        with self.assertRaisesRegex(ValueError,'budget exhausted'): self.check()


if __name__=='__main__': unittest.main()
