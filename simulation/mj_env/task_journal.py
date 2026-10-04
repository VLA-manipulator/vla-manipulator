"""Single-task JSONL adapter. JSON request on stdin; fixed preview cache only."""
import json, sys, time, re, base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from mj_env.closed_loop import call, FIELDS, POSE_FIELDS, stationary
from mj_env.observation_bundle import render_bundle, render_decision_bundle
from mj_env.task_runtime import task_state, motion_report, validate_submission, preflight

SERVER='http://127.0.0.1:8765'
TASK_DIR=Path('.tmp/tasks')
CACHE=Path('.tmp/task_preview')
TERM=dict(joint_speed_max_rad_s=.02,tcp_speed_max_m_s=.002,stable_window_s=.3)
def lock_exclusive(handle):
    """Non-blocking exclusive lock held until the handle closes; OSError if already held."""
    handle.seek(0)
    if sys.platform=='win32':
        import msvcrt
        msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
def open_journal(request, directory=TASK_DIR):
    """Only explicit init may create a journal; all other calls resume it."""
    spec=request.get('task') or {}
    task_id=request.get('task_id')
    if not isinstance(task_id,str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',task_id):
        raise ValueError('task_id required: 1-64 lowercase ASCII letters, digits, _ or -')
    if task_id.split('_')[0].split('-')[0] in {'con','prn','aux','nul',*(f'com{i}' for i in range(10)),*(f'lpt{i}' for i in range(10))}:
        raise ValueError('reserved task_id')
    if spec.get('task_id',task_id)!=task_id:
        raise ValueError('conflicting task_id values')
    operation=request.get('operation','append')
    if operation not in ('init','append','resume'):
        raise ValueError('operation must be init, append or resume')
    if operation=='resume' and set(request)-{'operation','task_id'}:
        raise ValueError('resume only observes and restores state; actions require a separate request')
    path=directory/(task_id+'.jsonl')
    if operation=='init':
        if set(request)-{'operation','task_id','task'}:
            raise ValueError('init only establishes task; send observations/actions separately')
        if not isinstance(spec.get('instruction'),str) or not spec['instruction'].strip():
            raise ValueError('init requires task.instruction')
        criteria=spec.get('criteria')
        if not isinstance(criteria,list) or not criteria or any(not isinstance(c,str) or not c.strip() for c in criteria):
            raise ValueError('init requires nonempty task.criteria')
        budget=spec.get('retry_budget',3)
        if type(budget) is not int or budget<0:
            raise ValueError('retry_budget must be a nonnegative integer')
        directory.mkdir(parents=True,exist_ok=True)
        try:
            handle=path.open('x+b')
        except FileExistsError:
            raise ValueError('task already exists: reuse this task_id without init; do not generate a replacement ID') from None
    else:
        if spec:
            raise ValueError('task is only accepted at init; keep the existing task_id')
        try:
            handle=path.open('r+b')
        except FileNotFoundError:
            raise ValueError('unknown task_id: recover the original ID; ordinary requests never create logs') from None
    return task_id,path,handle

def main():
    request=json.load(sys.stdin)
    if sum(k in request for k in ('command','result','review','task_update','target_selection','robot_reference_selection'))>1:
        raise ValueError('one operation per request: command, result or historical review')
    task_id,log,handle=open_journal(request)
    with handle as lock, ExitStack() as resources:
        lock_exclusive(lock)
        records=[]
        lock.seek(0)
        content=lock.read().decode('utf-8')
        if content and not content.endswith('\n'):
            raise ValueError('incomplete journal tail; preserve history and report')
        for line in content.splitlines():
            if line.strip(): records.append(json.loads(line))
        initializing=request.get('operation')=='init'
        if not initializing and (not records or records[0]['type']!='task' or
                                 any(e['task_id']!=task_id for e in records)):
            raise ValueError('invalid task journal; preserve it and report, do not create a replacement')
        print(json.dumps(dict(task_id=task_id,journal_path=str(log.resolve()))))
        cache=CACHE/task_id
        cache.mkdir(parents=True,exist_ok=True)
        counter=len(records)
        cycle=1+max((e['cycle'] for e in records),default=0)
        def emit(kind,data):
            nonlocal counter
            counter+=1
            e=dict(schema_version='2.0',event_id=counter,task_id=task_id,cycle=cycle,type=kind,recorded_at=time.time(),data=data)
            lock.seek(0,2); lock.write((json.dumps(e,ensure_ascii=False)+'\n').encode('utf-8')); lock.flush()
            records.append(e)
            return counter
        if initializing:
            emit('task',request['task'])
            print(json.dumps(dict(status='initialized',task_id=task_id,event_id=1)))
            return
        def print_state():
            print(json.dumps(dict(type='task_state', **task_state(records))))
        def reject(error):
            data=dict(status='blocked', reason=str(error), motion_submitted=False,
                      next_required_action='observe_or_resume_and_replan')
            emit('submission_rejected',data)
            print(json.dumps(data)); print_state()
        if 'robot_reference_selection' in request:
            selection=request['robot_reference_selection']
            try:
                last_bundle=next(r for r in reversed(records) if r['type']=='observation_bundle')
                if selection['bundle_ref']!=last_bundle['event_id'] or last_bundle['data'].get('role')!='decision':
                    raise ValueError('Reference must come from latest decision bundle')
                point=selection['pixel_xy']
                if not isinstance(point,list) or len(point)!=2 or any(type(v) not in (int,float) for v in point) or not (16<=point[0]<624 and 16<=point[1]<464):
                    raise ValueError('Reference must be a visible interior pixel in original global image')
                ref=last_bundle['data']['current_observation_ref']
                sample=next(r['data'] for r in records if r['event_id']==ref)
                rotation=sample['tcp_rotation_matrix']
                emit('robot_reference_selection',dict(status='selected',pixel_xy=point,observation_ref=ref,
                    initial_rotation=rotation,anchors=[dict(observation_ref=ref,tcp_position_m=sample['tcp_position_m'],pixel_xy=point)],
                    identification='visually selected fixed fingertip TCP reference; not simulator projection'))
            except (StopIteration,KeyError,ValueError,TypeError) as error:
                reject(error);return
        if 'target_selection' in request:
            selection=request['target_selection']
            last_bundle=next((r for r in reversed(records) if r['type']=='observation_bundle'),None)
            try:
                if not last_bundle or last_bundle['data'].get('role')!='decision' or selection['bundle_ref']!=last_bundle['event_id']:
                    raise ValueError('Select only from the latest decision bundle')
                label=selection['label']
                if not isinstance(label,str) or not label.strip(): raise ValueError('Visual target label required')
                tracks={}
                for camera,index in selection['region_indexes'].items():
                    field={'global':'current_image_regions','wrist':'wrist_image_regions'}[camera]
                    regions=last_bundle['data'][field]['regions']
                    if type(index) is not int or not 0<=index<len(regions): raise ValueError('Invalid region index')
                    tracks[camera]=dict(label=label,status='selected',region=regions[index],
                                        observation_ref=last_bundle['data']['current_observation_ref'],
                                        session_id=last_bundle['data']['current_session_id'])
                if not tracks: raise ValueError('Select at least one visible camera region')
            except (KeyError,ValueError,TypeError) as error:
                reject(error); return
            emit('target_selection',dict(tracks=tracks,source_bundle_ref=selection['bundle_ref']))
        if 'task_update' in request:
            update=request['task_update']
            if not isinstance(update,dict) or set(update)-{'instruction','criteria','source'}:
                reject('task_update only accepts instruction, criteria, source'); return
            if not isinstance(update.get('instruction'),str) or not update['instruction'].strip():
                reject('task_update requires instruction'); return
            criteria=update.get('criteria')
            if criteria is not None and (not isinstance(criteria,list) or not criteria or any(not isinstance(c,str) or not c.strip() for c in criteria)):
                reject('criteria must be a nonempty string list'); return
            update=dict(update,criteria_pending=criteria is None)
            emit('task_update',update)
            print_state(); return
        if request.get('review') is not None:
            review=request['review']
            selected=[e for e in records if e['type']=='observation' and
                      (e['event_id'] in review['event_ids'] if 'event_ids' in review else e['cycle']==review['cycle'])]
            if not selected: raise ValueError('no observations in requested cycle')
            bundle=render_bundle([e['data'] for e in selected],[e['event_id'] for e in selected],cache/'history')
            bundle['role']='history'
            bundle['latest_role']='last_recorded_observation; not a new live observation'
            bid=emit('observation_bundle',bundle)
            print(json.dumps(dict(bundle_ref=bid,**bundle))); return
        if request.get('command') or request.get('result'):
            try:
                bundle=next((e['data'] for e in reversed(records) if e['type']=='observation_bundle'),{})
                verified=validate_submission(request,records,bundle)
            except ValueError as error:
                reject(error); return
            review=dict(request['visual_review'],verified_execution_refs=verified)
            emit('visual_review',review)
        if request.get('result'):
            emit('task_result',request['result']); print('task result appended'); print_state(); return
        if request.get('command'):
            # Different task journals must not become simultaneous robot senders.
            try:
                sender=resources.enter_context((CACHE/'controller.lock').open('a+b'))
                lock_exclusive(sender)
            except OSError:
                reject('another journal sender holds the robot; observe later'); return
        samples=[]; refs=[]
        def take():
            raw=call(SERVER,'observe')
            s={k:raw[k] for k in FIELDS}; s['received_monotonic_s']=time.monotonic()
            s.update({k:raw[k] for k in POSE_FIELDS if k in raw})
            s['images_png_base64']=raw['images_png_base64']
            s['image_encoding']='png;base64';s['image_size']=[640,480]
            if not samples or s['frame_id']!=samples[-1]['frame_id']:
                refs.append(emit('observation',s)); samples.append(s)
            if s['snapshot_age_s']>.2: raise RuntimeError('stale observation')
            if not isinstance(s['images_png_base64'],dict) or any(not s['images_png_base64'].get(c) for c in ('global','wrist')):
                raise RuntimeError('both camera images required')
            if len(samples)>1:
                if s['session_id']!=samples[0]['session_id']:raise RuntimeError('session changed')
                gap=samples[-1]['captured_at_unix_s']-samples[-2]['captured_at_unix_s']
                if not 0<gap<=.3:raise RuntimeError('invalid sample gap')
            return s
        try:
            first=take()
        except (RuntimeError, OSError) as error:
            reject(error); return
        if request.get('command'):
            try:
                expected_start=next(e['data'] for e in records if e['event_id']==request['start_observation_ref'] and e['type']=='observation')
                info=call(SERVER,'info')
                # Metadata round-trip can outlive the freshness budget. Refresh
                # independently; it is not part of the continuous motion sample.
                samples.clear(); refs.clear()
                first=take()
                state=task_state(records)
                candidate=request.get('decision')
                if isinstance(candidate,dict) and candidate.get('failure'):
                    # Count a newly reported visual failure before granting a retry.
                    state=task_state(records+[dict(event_id=counter+1,type='decision',data=candidate)])
                preflight(request,expected_start,first,info,state)
                # Check a measured stationary window before sending, not one pose.
                deadline=time.monotonic()+1
                while not stationary(samples,TERM) and time.monotonic()<deadline:
                    time.sleep(.035)
                    first=take()
                if not stationary(samples,TERM):
                    raise ValueError('robot not stationary before submission')
                checked=preflight(request,expected_start,first,info,state)
            except (ValueError, KeyError, StopIteration, RuntimeError, OSError) as error:
                reject(error); return
            command=request['command'];args=request['args'];duration=args['duration_s']
            checked['start_observation_ref']=refs[-1]
            emit('preflight',checked)
            decision=dict(request['decision']); decision.update(type='decision',task_id=task_id,cycle=cycle,
                input_refs=list(dict.fromkeys([request['start_observation_ref'],refs[-1]])),retry_count=state['failure_count'])
            did=emit('decision',decision)
            plan=dict(type='plan',decision_ref=did,subtask_id=decision['current_subtask'],objective=request['objective'],start_observation_ref=refs[-1],reviewed_start_observation_ref=request['start_observation_ref'],stage=request['stage'],execute_allowed=True,
              timing=dict(motion_s=duration,settle_budget_s=2,max_segment_s=duration+2),sampling=dict(requested_hz=dict(global_camera=20,wrist=20,robot=20),max_age_s=.2,max_gap_s=.3,max_sensor_skew_s=0),
              trajectory=dict(frame='robot_base; m; rad; table_z=0',reference_point='aperture centre',representation='controller joint smooth interpolation',waypoints=[dict(command=command,args=args)],endpoint_velocity=0),
              gripper_actions=[args] if command=='gripper' else [],path_check=request['path'],expected_observations=request['expected'],stop_conditions=['stale images','session change','gap > .3s'],stop_response='cancel then reobserve; do not open gripper',terminal=TERM)
            pid=emit('plan',plan)
            start=time.monotonic(); errors=[]; status='blocked'; response=None; future=None
            pool=ThreadPoolExecutor(max_workers=1)
            try:
                if first['action']['status']=='running':raise RuntimeError('existing motion')
                emit('command',dict(command=command,args=args))
                future=pool.submit(call,SERVER,command,args)
                while time.monotonic()-start<duration+2:
                    s=take()
                    if future.done() and response is None:
                        raw=future.result();response={k:raw[k] for k in ('action_id','status') if k in raw}
                        emit('command_feedback',response)
                    complete=response is not None and (command=='wait' or (s['action'].get('action_id')==response.get('action_id') and s['action']['status']=='completed'))
                    if complete and stationary(samples,TERM):status='completed';break
                    time.sleep(.035)
                else:raise RuntimeError('terminal not stationary within budget')
            except Exception as error:
                errors.append(str(error));status='aborted' if future else 'blocked'
                emit('anomaly',dict(error=str(error)))
                if future:call(SERVER,'stop')
            finally:
                pool.shutdown(wait=True)
                if status!='completed' and future:
                    call(SERVER,'stop')
            times=[s['captured_at_unix_s'] for s in samples]
            output=dict(type='execution',plan_ref=pid,status=status,observation_refs=refs,actual_duration_s=time.monotonic()-start,
              sampling_result=dict(unique_samples=len(samples),actual_hz=(len(samples)-1)/(times[-1]-times[0]) if len(times)>1 else None,max_gap_s=max((b-a for a,b in zip(times,times[1:])),default=0),quality_passed=not errors),
              terminal_observation_ref=refs[-1] if status=='completed' else None,robot_stationary=stationary(samples,TERM) if status=='completed' else None,errors=errors,requires_decision=True)
            output['motion_report']=motion_report(samples,refs,output)
            emit('execution',output);print(json.dumps(output))
        latest=samples[-1]
        bundle=render_decision_bundle(records,refs[-1],cache)
        if request.get('command') and output['status']!='completed':
            bundle['latest_role']='last_available_observation; final state unknown'
            bundle['role']='recovery_required'
        reference=bundle.get('robot_visual_reference')
        current_image_path=Path(bundle['required_images'][-1]['path']) if reference and reference.get('status')=='tracked' else None
        if request.get('command')=='gripper' and output['status']=='completed':
            from mj_env.gripper_identification import identify,encoded_pair,verify_return
            identification=identify(samples)
            previous_probe=next((e for e in reversed(records) if e['type']=='gripper_identification'),None)
            evidence_samples=samples
            evidence_refs=refs
            if previous_probe:
                previous_refs=previous_probe['data']['observation_refs']
                previous_samples=[next(e['data'] for e in records if e['event_id']==ref) for ref in (previous_refs[0],previous_refs[-1])]
                if identification['state']!='gripper_unknown' and previous_probe['data'].get('arm_joint_drift_rad',float('inf'))<=.003:
                    identification=verify_return(*previous_samples,samples[-1],identification)
                    if identification.get('return_validation')=='checked':
                        evidence_samples=[previous_samples[0],samples[-1]]
                        evidence_refs=[previous_refs[0],refs[-1]]
            encoded=encoded_pair(evidence_samples,identification)
            identification_ref=emit('gripper_identification',dict(identification,
                observation_refs=refs,evidence_refs=evidence_refs,annotated_pair_png_base64=encoded))
            directory=cache/'gripper_identification';directory.mkdir(exist_ok=True)
            path=directory/'current_pair.png';path.write_bytes(base64.b64decode(encoded))
            bundle['gripper_identification']=dict(identification,event_ref=identification_ref)
            bundle['required_images'].append(dict(path=str(path.resolve()),event_ids=[evidence_refs[0],evidence_refs[-1]]))
        from mj_env.visual_servo import visual_relation
        bundle['visual_relation']=visual_relation(reference,bundle.get('target_tracks',{}).get('global',{}))
        if reference and reference.get('status')=='tracked':
            from PIL import Image,ImageDraw
            path=current_image_path
            with Image.open(path) as source:
                annotated=source.convert('RGB')
            draw=ImageDraw.Draw(annotated)
            x,y=reference['pixel_xy'];y+=48
            draw.ellipse((x-6,y-6,x+6,y+6),outline='white',width=2)
            draw.text((max(0,x-100),y+8),'TRACKED TCP REFERENCE',fill='white',stroke_width=1,stroke_fill='black')
            annotated.save(path)
        bid=emit('observation_bundle',bundle)
        print(json.dumps(dict(bundle_ref=bid,**bundle)))
        print(json.dumps({k:v for k,v in latest.items() if k!='images_png_base64'}));print('event_id',refs[-1])
        print_state()

if __name__=='__main__':main()
