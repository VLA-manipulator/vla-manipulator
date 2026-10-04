"""Deterministic journal state, motion measurements and submission checks.

Only permitted camera/robot observations are consumed. No scene/model access.
"""
import math
import time


POLICY = {
    'version': '1',
    'max_observation_age_s': .2,
    'start_position_tolerance_m': .005,
    'start_joint_tolerance_rad': .03,
    'start_gripper_tolerance_m': .002,
    'max_mean_position_speed_m_s': .05,
    'max_mean_joint_speed_rad_s': .3,
    'max_gripper_step_m': .02,
    'stages': {
        'probe': {'position_step_m': .02, 'joint_step_rad': .10},
        'approach': {'position_step_m': .03, 'joint_step_rad': .15},
        'contact': {'position_step_m': .01, 'joint_step_rad': .05},
        'transport': {'position_step_m': .03, 'joint_step_rad': .15},
    },
    'meaning': 'conservative request limits, NOT collision clearance or peak-speed guarantees',
}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, size):
    return isinstance(value, list) and len(value) == size and all(number(v) for v in value)


def pending_executions(records):
    # Only the journal writer adds verified_execution_refs after validating a review.
    reviewed = {ref for r in records if r['type'] == 'visual_review'
                for ref in r['data'].get('verified_execution_refs', [])}
    return [r for r in records if r['type'] == 'execution' and r['event_id'] not in reviewed]


def task_state(records):
    def last(kind):
        return next((r for r in reversed(records) if r['type'] == kind), None)
    decision, execution, result = last('decision'), last('execution'), last('task_result')
    update=last('task_update')
    if update and result and update['event_id']>result['event_id']:
        result=None
    task = dict(records[0]['data']) if records else {}
    for r in records:
        if r['type'] == 'task_update':
            task.update(r['data'])
    pending = pending_executions(records)
    failures = set()
    prior_execution = None
    for r in records:
        if r['type'] == 'execution':
            prior_execution = r['event_id']
            if r['data']['status'] != 'completed':
                failures.add(prior_execution)
        if r['type'] == 'decision' and r['data'].get('failure'):
            failures.add(prior_execution if prior_execution else 'decision_' + str(r['event_id']))
    command = last('command')
    unresolved = bool(command and (not execution or command['event_id'] > execution['event_id']))
    ended = bool(result and result['data'].get('status') in ('completed','failed','cancelled'))
    return dict(task_id=records[0]['task_id'] if records else None, task=task,
                current_subtask=decision['data'].get('current_subtask') if decision else None,
                subtasks=decision['data'].get('subtasks', []) if decision else [],
                latest_decision_ref=decision['event_id'] if decision else None,
                latest_conclusion=decision['data'].get('transition_reason') if decision else None,
                latest_execution_ref=execution['event_id'] if execution else None,
                latest_execution_status=execution['data']['status'] if execution else None,
                pending_execution_refs=[r['event_id'] for r in pending],
                failure_count=len(failures), retry_budget=task.get('retry_budget', 3),
                status=result['data'].get('status') if result else 'active',
                unresolved_command=unresolved, result=result['data'] if result else None,
                motion_submission_allowed=False,
                next_required_action=('resolve_unknown_command' if unresolved else 'none_task_ended' if ended else
                                      'review_previous_motion' if pending else 'observe_and_review'),
                note='state is reconstructed from journal; live checks are still required', policy=POLICY)


def motion_report(samples, refs, execution=None):
    report = dict(start_observation_ref=refs[0] if refs else None,
                  end_observation_ref=refs[-1] if refs else None,
                  orientation_change_rad=None,
                  orientation_status='unavailable: permitted observations have no tool orientation',
                  scope='robot measurements only; not object motion or collision detection',
                  path_scope='sampled aperture positions, not a continuous path guarantee',
                  robot_stationary=(execution or {}).get('robot_stationary'),
                  terminal_state_confirmed=(execution or {}).get('terminal_observation_ref') is not None)
    if not samples or len(samples) != len(refs):
        return dict(report, measurement_status='missing samples')
    if len({s['session_id'] for s in samples}) != 1:
        return dict(report, measurement_status='mixed sessions; no motion comparison')
    first, last = samples[0], samples[-1]
    times = [s['captured_at_unix_s'] for s in samples]
    gaps = [b-a for a, b in zip(times, times[1:])]
    report.update(measurement_status='measured', duration_s=times[-1]-times[0],
                  unique_frames=len({s['frame_id'] for s in samples}),
                  max_gap_s=max(gaps, default=0), timestamps_increasing=all(g > 0 for g in gaps),
                  actual_hz=(len(set(s['frame_id'] for s in samples))-1)/(times[-1]-times[0]) if times[-1] > times[0] else None,
                  max_snapshot_age_s=max(s['snapshot_age_s'] for s in samples))
    for field in ('aperture_position_m', 'tcp_position_m', 'joint_positions_rad'):
        size = 6 if field == 'joint_positions_rad' else 3
        if all(vector(s.get(field), size) for s in samples):
            report[field] = dict(start=first[field], end=last[field],
                                 delta=[b-a for a, b in zip(first[field], last[field])],
                                 max_excursion=[max(abs(s[field][i]-first[field][i]) for s in samples) for i in range(size)])
    if 'aperture_position_m' in report:
        start, end = first['aperture_position_m'], last['aperture_position_m']
        delta = [b-a for a, b in zip(start, end)]
        squared = sum(x*x for x in delta)
        deviations = []
        for s in samples:
            p = s['aperture_position_m']
            t = max(0, min(1, sum((p[i]-start[i])*delta[i] for i in range(3))/squared)) if squared else 0
            deviations.append(math.dist(p, [start[i]+t*delta[i] for i in range(3)]))
        report['endpoint_distance_m'] = math.sqrt(squared)
        report['max_sampled_deviation_from_endpoint_line_m'] = max(deviations)
    return report


def validate_submission(request, records, bundle):
    """Validate evidence coverage independently of the newest single image."""
    from mj_env.observation_bundle import validate_review
    validate_review(records, request.get('visual_review'))
    if task_state(records)['task'].get('criteria_pending'):
        raise ValueError('instruction revised: update criteria before motion or result')
    if bundle.get('role') != 'decision':
        raise ValueError('decision bundle required: observe or resume; historical review is not a live start')
    pending = {r['event_id'] for r in pending_executions(records)}
    covered = set(bundle.get('execution_refs', []))
    declared = set(request['visual_review'].get('execution_refs', []))
    if not pending.issubset(covered & declared):
        raise ValueError('review must include pending execution_refs; a refreshed image cannot replace process review')
    if request.get('command'):
        if task_state(records)['task'].get('criteria_pending'):
            raise ValueError('instruction revised: update criteria before motion')
        if request.get('start_observation_ref') != bundle.get('current_observation_ref'):
            raise ValueError('start_observation_ref must equal decision bundle current_observation_ref')
        if bundle.get('missing_images'):
            raise ValueError('missing images: motion blocked')
        if task_state(records)['unresolved_command']:
            raise ValueError('unresolved command: motion blocked; do not replay')
    return sorted(pending)


def preflight(request, expected, actual, info, state, now=None):
    """Fail closed before submitting any robot command. Never proves collision safety."""
    now = time.time() if now is None else now
    command, args = request.get('command'), request.get('args')
    allowed = {'move': {'position_m', 'aperture_width_m', 'duration_s'},
               'joints': {'positions_rad', 'duration_s'},
               'gripper': {'width_m', 'duration_s'}, 'wait': {'duration_s'}}
    if command not in allowed or not isinstance(args, dict) or set(args)-allowed[command]:
        raise ValueError('unsupported command/arguments; tool orientation constraints are not measured')
    duration = args.get('duration_s')
    if not number(duration) or not .05 <= duration <= 3:
        raise ValueError('duration_s must be finite and within [0.05, 3]')
    if request.get('stage') not in POLICY['stages']:
        raise ValueError('stage required: probe / approach / contact / transport')
    decision = request.get('decision')
    if not isinstance(decision, dict) or not decision.get('current_subtask'):
        raise ValueError('decision.current_subtask required')
    if any(not request.get(k) for k in ('objective', 'path', 'expected')):
        raise ValueError('objective, path and expected are required')
    if info.get('protocol_version') != 2:
        raise ValueError('protocol 2 required')
    if state['unresolved_command']:
        raise ValueError('unresolved previous command')
    if state['task'].get('criteria_pending'):
        raise ValueError('revised instruction requires updated criteria')
    if state['result'] and state['result'].get('status') in ('completed', 'failed', 'cancelled'):
        raise ValueError('task already ended')
    if state['failure_count'] > state['retry_budget']:
        raise ValueError('retry budget exhausted')
    for s in (expected, actual):
        if not number(s.get('snapshot_age_s')) or not 0 <= s['snapshot_age_s'] <= .2:
            raise ValueError('stale or invalid observation')
        if not all(vector(s.get(k), n) for k, n in
                   (('joint_positions_rad', 6), ('aperture_position_m', 3), ('tcp_position_m', 3))):
            raise ValueError('invalid robot measurements')
        if not number(s.get('gripper_width_m')):
            raise ValueError('invalid gripper measurement')
    age = now-actual['captured_at_unix_s']
    if not number(age) or not -.05 <= age <= POLICY['max_observation_age_s']:
        raise ValueError('fresh preflight observation required')
    if expected['session_id'] != actual['session_id']:
        raise ValueError('session changed; reobserve and replan')
    if actual['action'].get('status') not in ('idle', 'completed', 'cancelled'):
        raise ValueError('action running or status unknown')
    if expected['action'].get('action_id') != actual['action'].get('action_id'):
        raise ValueError('action changed since reviewed start')
    if max(abs(a-b) for a,b in zip(expected['joint_positions_rad'], actual['joint_positions_rad'])) > .03 or any(
            math.dist(expected[k], actual[k]) > .005 for k in ('tcp_position_m','aperture_position_m')) or abs(expected['gripper_width_m']-actual['gripper_width_m']) > .002:
        raise ValueError('robot differs from reviewed start; reobserve and replan')
    limits = POLICY['stages'][request['stage']]
    if command == 'move':
        if not vector(args.get('position_m'), 3):
            raise ValueError('position_m requires three finite numbers')
        distance = math.dist(args['position_m'], actual['aperture_position_m'])
        if distance > limits['position_step_m']+1e-9 or distance/duration > POLICY['max_mean_position_speed_m_s']+1e-9:
            raise ValueError(f'position step/rate exceeds policy: measured aperture start={actual["aperture_position_m"]}, target={args["position_m"]}, distance={distance:.6f}m (limit {limits["position_step_m"]}m), mean_rate={distance/duration:.6f}m/s (limit {POLICY["max_mean_position_speed_m_s"]}m/s). This is NOT a start-drift check; use aperture_position_m, not tcp_position_m.')
        if 'aperture_width_m' in args and (not number(args['aperture_width_m']) or abs(args['aperture_width_m']-actual['gripper_width_m']) > .002):
            raise ValueError('aperture reference width differs from measured gripper width')
    if command == 'joints':
        target, bounds = args.get('positions_rad'), info.get('joint_limits_rad')
        if not vector(target, 6) or not isinstance(bounds, list) or len(bounds) != 6 or not all(vector(b, 2) for b in bounds):
            raise ValueError('six joint targets and device joint limits required')
        if any(not lo <= q <= hi for q,(lo,hi) in zip(target,bounds)):
            raise ValueError('joint target outside device limits')
        step = max(abs(a-b) for a,b in zip(target,actual['joint_positions_rad']))
        if step > limits['joint_step_rad']+1e-9 or step/duration > POLICY['max_mean_joint_speed_rad_s']+1e-9:
            raise ValueError('joint step or mean joint rate exceeds adapter policy')
    if command == 'gripper':
        width, maximum = args.get('width_m'), info.get('max_gripper_width_m')
        if not number(width) or not number(maximum) or not 0 <= width <= maximum:
            raise ValueError('width_m outside device range')
        if abs(width-actual['gripper_width_m']) > POLICY['max_gripper_step_m']+1e-9:
            raise ValueError('gripper step exceeds adapter policy')
    return dict(passed=True, policy_version=POLICY['version'], stage=request['stage'],
                checks='reviewed start, live state, command schema, duration and request limits',
                collision_checked=False, object_state_checked=False,
                orientation_checked=False, trajectory='joint interpolation; Cartesian straightness not guaranteed')
