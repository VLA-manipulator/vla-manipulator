"""Bounded Chat Completions tool loop for the documented robot workflow."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import queue
import threading
import math
from datetime import datetime

import httpx

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {'type':'function','function':{
        'name':'identify_gripper','description':'One small jaw-only identification segment, never an arm move. close_probe reduces measured width by 0.015m; return_probe reverses the last observed probe. Only use after images show empty jaw gap and clear motion path. Review the returned paired BEFORE/AFTER images; cyan numbered regions are motion candidates and may include SHADOWS, not TCP. Each call ends stationary. No automatic alignment authorization.',
        'parameters':{'type':'object','properties':{
            'phase':{'type':'string','enum':['close_probe','return_probe']},
            'path_check':{'type':'string'},'global_findings':{'type':'string'},
            'wrist_findings':{'type':'string'},'process_findings':{'type':'string'}},
            'required':['phase','path_check','global_findings','wrist_findings','process_findings'],'additionalProperties':False}}},
    {'type':'function','function':{
        'name':'set_robot_reference','description':'Select the physical fixed fingertip TCP point in the current original GLOBAL image (640x480). Not motor/body, not moving jaw, not montage coordinates. Local optical-flow tracking and measured-pose probes learn the image Jacobian without camera/scene truth. If unsure, do not select. Re-selection clears calibration.',
        'parameters':{'type':'object','properties':{'pixel_xy':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}},'required':['pixel_xy'],'additionalProperties':False}}},
    {'type':'function','function':{
        'name':'select_target','description':'Pin the visually identified target using explicit region_index values from current camera measurements. Do not count only same-color entries. Choose only cameras where target is visible; wrist/global indexes are independent. Selection and subsequent measurements stay in the same task journal. No motion.',
        'parameters':{'type':'object','properties':{
            'label':{'type':'string'},
            'region_indexes':{'type':'object','properties':{'global':{'type':'integer'},'wrist':{'type':'integer'}},'additionalProperties':False}},
            'required':['label','region_indexes'],'additionalProperties':False}}},
    {'type':'function','function':{
        'name':'robot_step','description':'Preferred motion tool. Relative Cartesian step is added deterministically to the LAST OBSERVED APERTURE centre. Automatically fills task ID, start/bundle/event refs and current execution refs. Still requires genuine image findings. Never infers grasp success.',
        'parameters':{'type':'object','properties':{
            'delta_m':{'type':'array','items':{'type':'number'},'minItems':3,'maxItems':3},
            'prepare_view':{'type':'boolean','enum':[True],'description':'One bounded step toward the robot_info nominal observation pose, useful when fingers/camera face away from work surface. Mutually exclusive with other motion arguments; requires stage approach and a visually clear path. Does not approach the object.'},
            'visual_align_width_m':{'type':'number','description':'Use a calibrated global robot reference and tracked target to calculate one bounded XY correction for the projected aperture at this anticipated closing width. Requires independent observed x/y/z probes first. No descent, gripper change, or success inference.'},
            'joint_delta_rad':{'type':'array','items':{'type':'number'},'minItems':6,'maxItems':6,'description':'Relative joint changes added to measured joint positions; tiny measured limit overshoots are clamped to device limits.'},
            'width_m':{'type':'number','description':'Use only for a gripper command; omit delta_m.'},
            'duration_s':{'type':'number'},'stage':{'type':'string','enum':['probe','approach','contact','transport']},
            'subtask':{'type':'string'},'objective':{'type':'string'},'path_check':{'type':'string'},
            'global_findings':{'type':'string'},'wrist_findings':{'type':'string'},'process_findings':{'type':'string'}},
            'required':['duration_s','stage','subtask','objective','path_check','global_findings','wrist_findings','process_findings'],
            'additionalProperties':False}}},
    {'type': 'function', 'function': {
        'name': 'task_journal',
        'description': 'Only robot execution/observation tool. Follow the supplied adapter. request_json is one JSON request; task_id is fixed by the runner. Required image bundles are automatically shown after this call.',
        'parameters': {'type': 'object', 'properties': {'request_json': {'type': 'string'}},
                       'required': ['request_json'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'robot_info', 'description': 'Read only permitted robot interface metadata.',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'view_timeline', 'description': 'Show additional timeline images from the LAST tool manifest only, by exact path. No arbitrary file access.',
        'parameters': {'type': 'object', 'properties': {'paths': {'type': 'array', 'items': {'type': 'string'}}},
                       'required': ['paths'], 'additionalProperties': False}}},
]

# The official endpoint accepts strict function schemas. Keep local validation
# too: provider acceptance alone is not a guarantee of valid tool arguments.
for _tool_schema in TOOLS:
    _tool_schema['function']['strict'] = True


def image_parts(paths, task_id, maximum):
    paths = list(dict.fromkeys(paths))
    if len(paths) > maximum:
        raise ValueError('Image budget exceeded; no frames silently omitted. Review the pending history separately.')
    root = (ROOT / '.tmp' / 'task_preview' / task_id).resolve()
    parts = []
    for name in paths:
        path = Path(name).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != '.png':
            raise ValueError('Preview path is outside the current task cache.')
        parts.append({'type': 'text', 'text': 'Tool observation image: '+str(path)})
        parts.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode('ascii')}})
    return parts


def manifest_paths(value):
    paths = set()
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and 'event_ids' in value:
            paths.add(value['path'])
        for child in value.values():
            paths.update(manifest_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(manifest_paths(child))
    return paths


def compact_events(entries):
    """Journal remains complete; model gets decision-relevant fields only."""
    compact=[]
    for e in entries:
        if not isinstance(e,dict): continue
        if 'bundle_ref' in e:
            out={k:e[k] for k in ('bundle_ref','role','current_observation_ref','execution_refs',
                                   'pending_execution_refs','required_event_ids','required_images','latest_role','missing_images','current_image_regions','wrist_image_regions','target_tracks','robot_visual_reference','visual_relation','gripper_identification') if k in e}
            out['processes']=[{k:p[k] for k in ('execution_ref','status','pending','motion_report') if k in p}
                              for p in e.get('processes',[])]
            out['optional_timeline_images']=[page for p in e.get('processes',[]) for part in p['parts'] for page in part['timeline_pages']]
            compact.append(out)
        elif e.get('type')=='task_state':
            compact.append({k:v for k,v in e.items() if k not in ('policy','note')})
        elif e.get('type')=='execution':
            compact.append({k:v for k,v in e.items() if k not in ('observation_refs','motion_report')})
        else: compact.append(e)
    return compact


class RobotTools:
    def __init__(self, task_id, image_limit):
        self.task_id = task_id
        self.image_limit = image_limit
        self.manifest = set()
        self.ended = False
        self.instruction = None
        self.latest_bundle=None
        self.latest_robot=None
        self.robot_metadata=None
        self.preparation_active=False

    def call(self, name, arguments):
        if name=='identify_gripper':
            if not self.latest_robot or not self.latest_bundle or self.latest_bundle.get('role')!='decision':
                raise ValueError('Resume and inspect images before jaw identification')
            phase=arguments.get('phase')
            width=self.latest_robot['gripper_width_m']
            if phase=='close_probe':
                target=width-.015
                if target<.015:
                    raise ValueError('Insufficient open width for this probe; do not force close')
            elif phase=='return_probe':
                previous=self.latest_bundle.get('gripper_identification',{})
                if 'start_width_m' not in previous or abs(width-previous['end_width_m'])>.002:
                    raise ValueError('Return requires the current matching jaw-probe evidence')
                target=previous['start_width_m']
            else:
                raise ValueError('Unknown jaw identification phase')
            return self.call('robot_step',dict(width_m=target,duration_s=1.2,stage='probe',
                subtask='active_gripper_identification',objective='Measure jaw-correlated image motion; do not infer TCP or grasp.',
                **{k:arguments[k] for k in ('path_check','global_findings','wrist_findings','process_findings')}))
        if name=='set_robot_reference':
            if not self.latest_bundle or self.latest_bundle.get('role')!='decision':
                raise ValueError('Observe/resume before selecting robot reference')
            return self.call('task_journal',{'request_json':json.dumps({'robot_reference_selection':dict(arguments,bundle_ref=self.latest_bundle['bundle_ref'])})})
        if name=='select_target':
            if not self.latest_bundle or self.latest_bundle.get('role')!='decision':
                raise ValueError('Observe/resume before selecting target')
            selection=dict(arguments,bundle_ref=self.latest_bundle['bundle_ref'])
            return self.call('task_journal',{'request_json':json.dumps({'target_selection':selection})})
        if name=='robot_step':
            b,s=self.latest_bundle,self.latest_robot
            if not b or not s or b.get('role')!='decision':
                raise ValueError('Observe/resume before robot_step')
            if sum(k in arguments for k in ('delta_m','joint_delta_rad','width_m','prepare_view','visual_align_width_m'))!=1:
                raise ValueError('Provide exactly one motion argument')
            if self.preparation_active and 'prepare_view' not in arguments:
                opening='width_m' in arguments and arguments['width_m']>=s['gripper_width_m']
                upward=arguments.get('delta_m',[1,1,-1])[:2]==[0,0] and arguments.get('delta_m',[0,0,-1])[2]>0
                if not opening and not upward:
                    raise ValueError('Observation posture preparation is unfinished. Use prepare_view:true until observation_pose_reached; opening or vertical upward recovery remains available.')
            if 'prepare_view' in arguments:
                if arguments['prepare_view'] is not True or arguments['stage']!='approach':
                    raise ValueError('prepare_view requires true and stage=approach')
                if not self.robot_metadata:
                    self.call('robot_info',{})
                hint=self.robot_metadata.get('observation_pose_hint',{}).get('aperture_position_m')
                if not hint:
                    raise ValueError('Device supplies no observation pose hint; do not invent one')
                delta=[b-a for a,b in zip(s['aperture_position_m'],hint)]
                distance=math.sqrt(sum(x*x for x in delta))
                if distance<.003:
                    self.preparation_active=False
                    return {'status':'observation_pose_reached','motion_submitted':False,
                            'next':'Use current cameras to select/align the target; this posture does not establish object alignment.'},[]
                factor=min(1,.028/distance)
                self.preparation_active=True
                arguments=dict(arguments)
                arguments.pop('prepare_view')
                arguments['delta_m']=[x*factor for x in delta]
            if 'visual_align_width_m' in arguments:
                from mj_env.visual_servo import alignment_delta
                correction=alignment_delta(b.get('robot_visual_reference') or {},b.get('target_tracks',{}).get('global',{}),s,
                    arguments['visual_align_width_m'],.008 if arguments['stage']=='contact' else .018 if arguments['stage']=='probe' else .028)
                if correction['aligned_in_projection']:
                    return dict(correction,motion_submitted=False),[]
                arguments=dict(arguments)
                arguments.pop('visual_align_width_m')
                arguments['delta_m']=correction['delta_m']
            args={'duration_s':arguments['duration_s']}
            if 'delta_m' in arguments:
                delta=arguments['delta_m']
                if not isinstance(delta,list) or len(delta)!=3 or any(type(d) not in (int,float) or not math.isfinite(d) for d in delta):
                    raise ValueError('delta_m must have three finite numbers')
                args['position_m']=[a+d for a,d in zip(s['aperture_position_m'],delta)]
                command='move'
            elif 'joint_delta_rad' in arguments:
                delta=arguments['joint_delta_rad']
                if not isinstance(delta,list) or len(delta)!=6 or any(type(d) not in (int,float) or not math.isfinite(d) for d in delta):
                    raise ValueError('joint_delta_rad requires six finite numbers')
                if not self.robot_metadata:
                    self.call('robot_info',{})
                bounds=self.robot_metadata['joint_limits_rad']
                target=[a+d for a,d in zip(s['joint_positions_rad'],delta)]
                for i,(lo,hi) in enumerate(bounds):
                    if target[i]<lo-.001 or target[i]>hi+.001:
                        raise ValueError(f'joint {i} target {target[i]} outside [{lo},{hi}]')
                    target[i]=max(lo,min(hi,target[i]))
                args['positions_rad']=target; command='joints'
            else:
                args['width_m']=arguments['width_m']; command='gripper'
            request=dict(command=command,args=args,stage=arguments['stage'],
                start_observation_ref=b['current_observation_ref'],
                visual_review=dict(bundle_ref=b['bundle_ref'],execution_refs=b['execution_refs'],
                    reviewed_event_ids=b['required_event_ids'],
                    **{k:arguments[k] for k in ('global_findings','wrist_findings','process_findings')}),
                decision=dict(current_subtask=arguments['subtask'],assessment='in_progress',transition='continue',
                              transition_reason=arguments['objective']),objective=arguments['objective'],
                path=arguments['path_check'],expected=[arguments['objective'],'Robot stationary at segment end; verify intended visual effect separately.'])
            result,parts=self.call('task_journal',{'request_json':json.dumps(request)})
            if self.preparation_active:
                hint=self.robot_metadata['observation_pose_hint']['aperture_position_m']
                remaining=math.dist(self.latest_robot['aperture_position_m'],hint)
                self.preparation_active=remaining>=.003
                result['preparation_progress']=dict(remaining_m=round(remaining,4),
                    status='in_progress' if self.preparation_active else 'observation_pose_reached',
                    next='Review current process/images, then continue prepare_view' if self.preparation_active else 'Select and visually align target')
            return result,parts
        if name == 'view_timeline':
            paths = arguments.get('paths')
            if not isinstance(paths, list) or not paths or not all(isinstance(p, str) and p in self.manifest for p in paths):
                raise ValueError('Only exact paths in the last observation manifest may be viewed.')
            return {'viewed_paths': paths}, image_parts(paths, self.task_id, self.image_limit)
        if name == 'robot_info':
            if arguments:
                raise ValueError('robot_info takes no arguments')
            process = subprocess.run([sys.executable, '-m', 'mj_env.agent_cli', 'info'],
                                     cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=130)
            if process.returncode:
                raise ValueError('robot_info failed; check local simulation service')
            raw = json.loads(process.stdout)
            allowed = ('protocol_version', 'commands', 'joint_names', 'joint_limits_rad',
                       'max_gripper_width_m', 'control_hz', 'frame', 'move', 'idle_physics','observation_pose_hint')
            self.robot_metadata={k: raw[k] for k in allowed if k in raw}
            return self.robot_metadata, []
        if name != 'task_journal' or set(arguments) != {'request_json'}:
            raise ValueError('Unsupported tool or arguments')
        request = json.loads(arguments['request_json'])
        if not isinstance(request, dict):
            raise ValueError('request_json must encode an object')
        if request.get('task_id', self.task_id) != self.task_id:
            raise ValueError('Changing task_id is prohibited')
        request['task_id'] = self.task_id
        if 'task_update' in request and request['task_update'].get('instruction') != self.instruction:
            raise ValueError('task_update must use the exact latest user instruction')
        if request.get('operation') == 'init' and (ROOT/'.tmp/tasks'/f'{self.task_id}.jsonl').exists():
            raise ValueError('Task already exists; use resume, never initialize another ID')
        if request.get('result',{}).get('status')=='completed':
            tracks=(self.latest_bundle or {}).get('target_tracks',{})
            if not tracks or not any(t.get('status') in ('tracked','selected') for t in tracks.values()):
                raise ValueError('Completion requires an explicitly selected, currently visible target. Use select_target and review evidence; do not infer success from jaw motion alone.')
        # Require an explicit visual association before closing or descending at
        # contact distance. Opening and upward recovery remain available.
        if request.get('command') in ('move','gripper') and self.latest_robot:
            args=request.get('args',{})
            if request['command']=='move' and request.get('stage')=='probe':
                destination=args.get('position_m',self.latest_robot['aperture_position_m'])
                lateral=math.dist(destination[:2],self.latest_robot['aperture_position_m'][:2])>1e-5
                bundle=self.latest_bundle or {}
                target=bundle.get('target_tracks',{}).get('global',{})
                reference=bundle.get('robot_visual_reference') or {}
                if lateral and target.get('status')=='tracked' and reference.get('status')!='tracked':
                    raise ValueError('Lateral target-direction probe requires a tracked fixed fingertip reference. Use set_robot_reference from the current image first; if it cannot be identified, report blocked. A static global target does not reveal robot motion direction. No motion sent.')
            closing=request['command']=='gripper' and args.get('width_m',float('inf')) < self.latest_robot['gripper_width_m']
            descending=(request['command']=='move' and request.get('stage')=='contact' and
                        args.get('position_m',[0,0,float('inf')])[2] < self.latest_robot['aperture_position_m'][2])
            if closing or descending:
                tracks=(self.latest_bundle or {}).get('target_tracks',{})
                if not tracks or not any(t.get('status') in ('tracked','selected') for t in tracks.values()):
                    raise ValueError('Before closing/contact descent, use select_target on the actual target in current images. No valid target track; no motion sent.')
        # shell=False; model text is stdin data, never shell code.
        process = subprocess.run([sys.executable, '-m', 'mj_env.task_journal'],
                                 input=json.dumps(request, ensure_ascii=True), cwd=ROOT,
                                 capture_output=True, text=True, encoding='utf-8')
        # No timeout/retry: killing a journal sender can strand an in-flight motion.
        if process.returncode:
            diagnostic=(process.stderr or '').strip().splitlines()
            detail=diagnostic[-1][:400] if diagnostic else 'no diagnostic'
            if not request.get('command'):
                raise ValueError('Non-motion journal request failed: '+detail+'. Resume the existing task to inspect state before correcting the request.')
            raise RuntimeError('Journal process failed; stop this run. Inspect the task through resume; do not replay its command. '+detail)
        entries = []
        for line in process.stdout.splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        bundles = [e for e in entries if isinstance(e, dict) and 'bundle_ref' in e]
        parts = []
        if bundles:
            bundle = bundles[-1]
            self.latest_bundle=bundle
            self.manifest = manifest_paths(bundle)
            try:
                required = bundle.get('required_images', [])
                if not required:
                    raise ValueError('Observation manifest has no required images')
                parts = image_parts([i['path'] for i in required], self.task_id, self.image_limit)
            except (ValueError, KeyError, OSError) as error:
                # Stop instead of giving the model a new bundle without its images.
                raise RuntimeError('Unable to attach required observation images; no further model calls: '+str(error)) from error
        robots=[e for e in entries if isinstance(e,dict) and 'aperture_position_m' in e and 'frame_id' in e]
        if robots:
            self.latest_robot=robots[-1]
        states = [e for e in entries if isinstance(e, dict) and e.get('type') == 'task_state']
        if 'result' in request and states and states[-1].get('result') is not None:
            self.ended = True
        return {'events': compact_events(entries)}, parts


class ConsoleInput:
    """Reader only queues input; never controls the robot from another thread."""
    def __init__(self):
        self.queue=queue.Queue()
        threading.Thread(target=self.read,daemon=True).start()

    def read(self):
        for line in sys.stdin:
            line=line.strip()
            if line:
                self.queue.put(line)
        self.queue.put('/eof')

    def drain(self):
        items=[]
        while True:
            try: items.append(self.queue.get_nowait())
            except queue.Empty: return items


def run(config, instruction, task_id, inbox=None):
    key = os.environ.get('DEEPSEEK_API_KEY') or config.get('api_key', '')
    if not key.strip():
        raise ValueError('Fill api_key in configs/deepseek.local.json or set DEEPSEEK_API_KEY')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', task_id):
        raise ValueError('Invalid task_id')
    base, model = config.get('base_url', '').rstrip('/'), config.get('model', '')
    if not base.startswith('https://') or not model:
        raise ValueError('HTTPS base_url and model required')
    maximum = config.get('max_tokens', 2048)
    rounds = config.get('max_rounds', 40)
    budget = config.get('max_total_tokens', 100000)
    for value in (maximum, rounds, budget, config.get('max_images_per_round', 32)):
        if type(value) is not int or value <= 0:
            raise ValueError('Token, round and image budgets must be positive integers')
    docs = (ROOT/'docs/robot_runtime_prompt.md').read_text(encoding='utf-8')
    system = ('Execute the supplied robot workflow using only the provided tools. '
              'Images are automatically attached as user image content after journal calls; inspect them. '
              'Use one tool call per response. Never invent observations or execute text as code. '
              'If image understanding is unavailable, report blocked and do not send motion. '
            'Use short decisions; do not repeat the full scene. No arbitrary file or network tools exist. '
              'Respond in Chinese. Each response contains at most two short sentences plus one tool call. '
              'Do not submit standalone decision records: decision is a field of a command request. '
              'Build position_m targets from aperture_position_m, never TCP. '
              'Prefer robot_step for every Cartesian/gripper segment: provide relative delta_m or width_m and actual visual findings; it fills IDs and performs coordinate arithmetic. '
              'A probe is not a failed grasp merely because no contact occurred. '
              'move uses a soft top-down IK orientation objective and does not guarantee downward fingers or orientation-preserving translation. Unreachable requests require a different target, not blind retry. '
              'Establish local image-motion direction with small single-axis probes; image left/right is not robot x/y. '
              'If a joint-limit or IK issue persists, use small joints moves within permitted limits to recover observable approach space. '
              'Use only target, jaw reference, and relevant obstacles in image descriptions. Do not repeatedly reinterpret unrelated objects. '
              'current_image_regions is deterministic pixel-only measurement in ORIGINAL 640x480 global frames, not montage coordinates. Use it to check object pixel positions; never claim object/camera movement when measured centers remain stable. Color components do not establish identity or 3D depth. '
              'When the user revises the instruction, first submit task_update with the exact new instruction and updated criteria; then resume and replan.\n\n'+docs)
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': f'Task ID (fixed): {task_id}\nUser instruction: {instruction}'}]
    tools = RobotTools(task_id, config.get('max_images_per_round', 32))
    tools.instruction=instruction
    def apply_input():
        nonlocal instruction
        items=inbox.drain() if inbox else []
        changed=False
        for item in items:
            if item=='/eof': continue
            if item=='/quit':
                raise ValueError('User requested exit; no further segments will be sent')
            instruction=item.removeprefix('/update ').strip()
            tools.instruction=instruction
            tools.ended=False
            tools.preparation_active=False
            if (ROOT/'.tmp/tasks'/f'{task_id}.jsonl').exists():
                tools.call('task_journal',{'request_json':json.dumps({'task_update':{'instruction':instruction,'source':'user_console'}})})
            messages.append({'role':'user','content':'Instruction revision (supersedes earlier goals, preserve history): '+instruction+
                             '\nSubmit task_update with exact instruction and revised criteria before any motion/result; then resume.'})
            print('Instruction revision queued for replanning: '+instruction,flush=True)
            changed=True
        return changed
    if (ROOT/'.tmp/tasks'/f'{task_id}.jsonl').exists():
        messages.append({'role': 'user', 'content': 'This task already exists. First call operation=resume. Preserve its recorded criteria and history.'})
    spent = 0
    with httpx.Client(timeout=httpx.Timeout(float(config.get('timeout_s', 120)), connect=15),
                      headers={'Authorization': 'Bearer '+key}) as client:
        for turn in range(1, rounds+1):
            apply_input()
            remaining = budget-spent
            if remaining <= 0:
                raise ValueError('Token budget reached; stopped between execution segments')
            body = dict(model=model, messages=messages, tools=TOOLS, stream=False,
                        parallel_tool_calls=False, max_tokens=min(maximum, remaining))
            for field in ('temperature', 'reasoning_effort'):
                if config.get(field) is not None:
                    body[field] = config[field]
            extra = config.get('extra_body', {})
            if not isinstance(extra, dict) or set(extra) & {'messages','tools','model','stream','max_tokens','tool_choice'}:
                raise ValueError('extra_body attempts to override protected robot-runner fields')
            body.update(extra)
            response = client.post(base+'/chat/completions', json=body)
            if response.status_code >= 400:
                raise ValueError(f'HTTP {response.status_code}; no retry or robot command. Check API model vision/tool support and options.')
            data = response.json()
            usage = data.get('usage') or {}
            total = usage.get('total_tokens')
            if type(total) is not int or total < 0:
                raise ValueError('API did not provide total_tokens; cannot enforce cumulative budget')
            spent += total
            print(json.dumps(dict(round=turn, usage=usage, cumulative_tokens=spent), ensure_ascii=False), flush=True)
            if apply_input():
                print('Discarded the old response before executing its tools.',flush=True)
                continue
            choice = data['choices'][0]
            if choice.get('finish_reason') not in ('stop', 'tool_calls'):
                raise ValueError('Incomplete/filtered model response; no tool calls executed')
            message = choice['message']
            calls = message.get('tool_calls') or []
            if len(calls) > 1:
                messages.append({k:message[k] for k in ('role','content','tool_calls','reasoning_content') if k in message})
                for rejected in calls:
                    messages.append({'role':'tool','tool_call_id':rejected['id'],'content':
                                     '{"error":"Multiple tools returned: NONE executed. Submit exactly ONE tool call next."}'})
                continue
            if message.get('content'):
                print(message['content'], flush=True)
            if spent >= budget:
                raise ValueError('Token budget reached; returned tool calls were NOT executed')
            if not calls:
                print('Model ended without a journal task_result; task completion is unconfirmed.')
                return
            # Preserve provider reasoning metadata required by some thinking APIs,
            # but never display or separately save it.
            messages.append({k: message[k] for k in ('role','content','tool_calls','reasoning_content') if k in message})
            tool = calls[0]
            print('Tool: '+tool['function']['name'],flush=True)
            try:
                arguments = json.loads(tool['function']['arguments'])
                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be an object')
                result, parts = tools.call(tool['function']['name'], arguments)
            except json.JSONDecodeError as error:
                raw_arguments=tool['function']['arguments']
                excerpt=raw_arguments[max(0,error.pos-50):error.pos+70]
                result, parts = {'error': str(error), 'invalid_argument_excerpt': excerpt,
                                 'instruction': 'No action sent. Return valid JSON; all string values need double quotes.'}, []
                print('Invalid argument excerpt: '+repr(excerpt),flush=True)
            except (ValueError, KeyError) as error:
                result, parts = {'error': str(error), 'instruction': 'Correct the request; do not change task ID or bypass the adapter.'}, []
            if 'error' in result:
                print('Tool rejected: '+str(result['error']),flush=True)
            for event in result.get('events',[]):
                if isinstance(event,dict) and event.get('type')=='execution':
                    print(json.dumps({k:event.get(k) for k in ('status','robot_stationary','errors')},ensure_ascii=False),flush=True)
            messages.append({'role': 'tool', 'tool_call_id': tool['id'], 'content': json.dumps(result, ensure_ascii=False)})
            if parts:
                # Retain numeric/event evidence, prune previous image bytes only.
                for old in messages:
                    if old['role']=='user' and isinstance(old.get('content'), list):
                        old['content'] = '[Earlier images removed from API context; use journal review if needed.]'
                messages.append({'role': 'user', 'content': [{'type':'text','text':'Current tool images; inspect before the next decision.'}]+parts})
            if apply_input():
                continue
            if tools.ended:
                print('Task result recorded in the original journal. Runner stopped.')
                return
            # Bounded conversation: authoritative task state and current bundle are
            # provided by the journal. Keep the latest two complete tool exchanges.
            tool_indexes=[i for i,m in enumerate(messages) if m['role']=='assistant' and m.get('tool_calls')]
            if len(tool_indexes)>2 and parts and tool['function']['name'] in ('task_journal','robot_step','select_target','set_robot_reference','identify_gripper'):
                messages[:]=messages[:2]+[{'role':'user','content':
                    'Latest user instruction: '+instruction+'. Older exchanges are in the task journal. '
                    'Use task_state, pending process evidence and latest observation below; do not repeat resolved reasoning.'}]+messages[tool_indexes[-2]:]
    raise ValueError('Maximum rounds reached; stopped between execution segments. Resume with the same task ID.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/deepseek.local.json'))
    parser.add_argument('--task-id', default=None)
    parser.add_argument('--instruction')
    parser.add_argument('--non-interactive',action='store_true',help='Disable console updates; requires --instruction')
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        if args.non_interactive and not args.instruction:
            raise ValueError('--non-interactive requires --instruction')
        instruction=args.instruction
        if not instruction:
            instruction=input('请输入任务指令（/quit 退出）：').strip()
            while not instruction:
                instruction=input('请输入任务指令：').strip()
            if instruction=='/quit': return 0
        task_id=args.task_id or 'task_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        print('Task ID: '+task_id,flush=True)
        inbox=None if args.non_interactive else ConsoleInput()
        if inbox:
            print('运行中可直接输入修订指令；/quit 在当前执行段结束后退出（不是急停）。',flush=True)
        run(config, instruction, task_id,inbox)
    except (ValueError, RuntimeError, OSError, KeyError, httpx.HTTPError, subprocess.SubprocessError) as error:
        print('Stopped: '+(type(error).__name__ if isinstance(error, httpx.HTTPError) else str(error)), file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
