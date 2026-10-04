"""Protocol-2 single-segment executor; camera/robot data only.

python -m mj_env.closed_loop observe --output DIR
python -m mj_env.closed_loop run --plan PLAN.json --output NEW_DIR
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import urllib.request
import urllib.error

from PIL import Image, ImageDraw

VERSION = '1.1'
FIELDS = ('session_id', 'frame_id', 'captured_at_unix_s', 'snapshot_age_s',
          'joint_positions_rad', 'joint_targets_rad', 'tcp_position_m',
          'aperture_position_m', 'gripper_width_m', 'action')
POSE_FIELDS = ('tcp_rotation_matrix', 'jaw_approach_axis_robot', 'free_space_aperture_sweep')


def call(server, command, args=None):
    body = json.dumps({'command': command, 'args': args or {}}).encode()
    req = urllib.request.Request(server + '/call', data=body,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Preserve controller validation diagnostics, never scene truth fields.
        if error.code == 400:
            payload=json.loads(error.read().decode('utf-8'))
            raise ValueError('controller rejected request: '+str(payload.get('error','invalid request'))) from None
        raise


def observe(server, directory):
    raw = call(server, 'observe')
    sample = {k: raw[k] for k in FIELDS}
    sample.update({k: raw[k] for k in POSE_FIELDS if k in raw})
    sample['received_monotonic_s'] = time.monotonic()
    sample['images'] = {}
    for camera in ('global', 'wrist'):
        path = directory / 'frames' / sample['session_id'] / str(sample['frame_id']) / (camera + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(raw['images_png_base64'][camera], validate=True))
        sample['images'][camera] = str(path.resolve())
    return sample


def stationary(samples, terminal):
    if not samples or samples[-1]['action']['status'] == 'running':
        return False
    end = samples[-1]['captured_at_unix_s']
    tail = []
    for sample in reversed(samples):
        tail.append(sample)
        if end - sample['captured_at_unix_s'] >= terminal['stable_window_s']:
            break
    if len(tail) < 4 or end - tail[-1]['captured_at_unix_s'] < terminal['stable_window_s']:
        return False
    for a, b in zip(tail, tail[1:]):
        if (a['session_id'] != b['session_id'] or a['frame_id'] == b['frame_id']
                or a['snapshot_age_s'] > .2 or b['snapshot_age_s'] > .2):
            return False
        dt = a['captured_at_unix_s'] - b['captured_at_unix_s']
        if dt <= 0 or dt > .15:
            return False
        speed = max(abs(x-y) / dt for x, y in zip(a['joint_positions_rad'], b['joint_positions_rad']))
        tcp = math.dist(a['tcp_position_m'], b['tcp_position_m']) / dt
        if speed > terminal['joint_speed_max_rad_s'] or tcp > terminal['tcp_speed_max_m_s']:
            return False
    return True


def read_sample(ref):
    """Read an observation JSON or a zero-based samples.jsonl#N reference."""
    filename, marker, index = ref.partition('#')
    content = Path(filename).read_text(encoding='utf-8-sig')
    return json.loads(content.splitlines()[int(index)] if marker else content)


def check_start(expected, actual):
    if expected['session_id'] != actual['session_id']:
        raise ValueError('planned session changed; reobserve and replan')
    if (max(abs(a-b) for a,b in zip(expected['joint_positions_rad'], actual['joint_positions_rad'])) > .03
            or math.dist(expected['tcp_position_m'], actual['tcp_position_m']) > .005):
        raise ValueError('robot differs from planned start; reobserve and replan')


def sheet(samples, path):
    if not samples:
        return
    indices = sorted(set(round(i * (len(samples)-1) / 7) for i in range(8)))
    canvas = Image.new('RGB', (640, 260 * len(indices)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(indices):
        s = samples[index]
        draw.text((5, row*260), f"sample {index}, frame {s['frame_id']}", fill='black')
        for col, camera in enumerate(('global', 'wrist')):
            with Image.open(s['images'][camera]) as img:
                canvas.paste(img.resize((320, 240)), (col*320, row*260+20))
    canvas.save(path)


def run(server, plan, directory):
    if plan.get('phase') != 'plan' or not plan.get('execute_allowed') or not plan.get('start_sample_ref'):
        raise ValueError('approved plan with start_sample_ref required')
    if len(plan['waypoints']) != 1:
        raise ValueError('one waypoint per segment required')
    waypoint = plan['waypoints'][0]
    command, args = waypoint['command'], waypoint['args']
    allowed = {'move': {'position_m', 'hinge_yaw_rad', 'aperture_width_m', 'duration_s'},
               'joints': {'positions_rad', 'duration_s'}, 'gripper': {'width_m', 'duration_s'},
               'wait': {'duration_s'}}
    if command not in allowed or set(args) - allowed[command]:
        raise ValueError('command/arguments not allowed')
    duration = args['duration_s']
    hz = plan['sample_hz_requested']
    budget = plan['max_segment_duration_s']
    if not 0.05 <= duration <= 10 or not 1 <= hz <= 20 or not duration < budget <= 15:
        raise ValueError('invalid duration/frequency/budget')
    max_gap = plan.get('max_sample_gap_s_allowed', .15)
    if not .05 <= max_gap <= .3:
        raise ValueError('gap allowance must be in [0.05, 0.3]')
    for key in ('joint_speed_max_rad_s', 'tcp_speed_max_m_s', 'stable_window_s'):
        if not 0 < plan['terminal'][key] < 10:
            raise ValueError('invalid terminal threshold')
    if plan['terminal']['stable_window_s'] < .3:
        raise ValueError('stable window must be at least 0.3s')
    expected_start = read_sample(plan['start_sample_ref'])
    if call(server, 'info')['protocol_version'] != 2:
        raise ValueError('protocol 2 required')
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    samples, errors, actions = [], [], []
    status = 'failed'
    started = time.monotonic()
    attempted = False
    pool = None
    future = None
    def take():
        sample = observe(server, directory)
        if sample['snapshot_age_s'] > .2:
            raise RuntimeError('stale camera')
        if samples and sample['session_id'] != samples[0]['session_id']:
            raise RuntimeError('session changed')
        if not samples or sample['frame_id'] != samples[-1]['frame_id']:
            gap = sample['captured_at_unix_s'] - samples[-1]['captured_at_unix_s'] if samples else 0
            samples.append(sample)
            with (directory/'samples.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(sample) + '\n')
            if len(samples) > 1 and not 0 < gap <= max_gap:
                raise RuntimeError(f'sample gap {gap:.3f}s exceeds {max_gap}s')
        return sample
    try:
        first = take()
        check_start(expected_start, first)
        if first['action']['status'] == 'running':
            raise RuntimeError('another action running')
        pool = ThreadPoolExecutor(max_workers=1)
        attempted = True
        (directory/'commands.jsonl').write_text(json.dumps({'submitted_at': time.time(), 'command': command, 'args': args})+'\n', encoding='utf-8')
        future = pool.submit(call, server, command, args)
        response = None
        next_sample = time.monotonic()
        while time.monotonic() - started <= budget:
            sample = take()
            if future.done() and response is None:
                response = future.result()
                if command != 'wait':
                    actions.append(response['action_id'])
                with (directory/'commands.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'response': {k: response[k] for k in ('action_id', 'status') if k in response}})+'\n')
            complete = response is not None and (command == 'wait' or
                (sample['action'].get('action_id') in actions and sample['action']['status'] == 'completed'))
            if complete and stationary(samples, plan['terminal']):
                status = 'completed'
                break
            next_sample += 1/hz
            time.sleep(max(0, next_sample-time.monotonic()))
        else:
            raise RuntimeError('segment budget exceeded / terminal not stationary')
    except Exception as error:
        errors.append(str(error))
        if attempted:
            try:
                call(server, 'stop')
            except Exception as stop_error:
                errors.append('stop failed: '+str(stop_error))
        status = 'aborted' if attempted else 'blocked'
    finally:
        if pool is not None:
            # An in-flight submission may be accepted after an early stop.
            pool.shutdown(wait=True)
            if status != 'completed':
                try:
                    call(server, 'stop')
                    if future.done() and future.exception() is None and command != 'wait':
                        action_id = future.result().get('action_id')
                        if action_id and action_id not in actions:
                            actions.append(action_id)
                except Exception as error:
                    errors.append('post-submission stop failed: '+str(error))
        if status != 'completed' and attempted:
            try:
                recovery = observe(server, directory / 'recovery')
                (directory/'recovery.json').write_text(json.dumps(recovery, indent=2), encoding='utf-8')
            except Exception as error:
                errors.append('recovery observation failed: '+str(error))
    times = [s['captured_at_unix_s'] for s in samples]
    elapsed = times[-1]-times[0] if len(times)>1 else 0
    result = dict(phase='execution', schema_version='1.0', run_id=plan['run_id'], cycle=plan['cycle'],
                  status=status, record_dir=str(directory.resolve()), action_ids=actions,
                  sample_count_unique=len(samples), sample_hz_requested=hz,
                  sample_hz_actual=(len(times)-1)/elapsed if elapsed else None,
                  max_sample_gap_s=max((b-a for a,b in zip(times,times[1:])), default=None),
                  actual_duration_s=time.monotonic()-started,
                  terminal_stationary=stationary(samples, plan['terminal']) if status == 'completed' else None,
                  terminal_sample_ref=f'samples.jsonl#{len(samples)-1}' if status == 'completed' else None,
                  last_recorded_sample_ref=f'samples.jsonl#{len(samples)-1}' if samples else None,
                  executor_version=VERSION,
                  visual_events=[], errors=errors, needs_decision=True)
    (directory/'execution.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    sheet(samples, directory/'contact_sheet.jpg')
    print(json.dumps(result, indent=2))
    if samples:
        print(json.dumps(samples[-1], indent=2))
    return 0 if status == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['observe', 'run'])
    parser.add_argument('--server', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    if args.command == 'observe':
        s = observe(args.server, args.output)
        (args.output/'observation.json').write_text(json.dumps(s, indent=2), encoding='utf-8')
        print(json.dumps(s, indent=2))
        return 0
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    return run(args.server, plan, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
