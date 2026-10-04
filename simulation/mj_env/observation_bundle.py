"""Deterministic dual-camera review pages from allowed recorded observations."""
import base64
import hashlib
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def render_bundle(samples, refs, cache):
    if not samples or len(samples) != len(refs):
        raise ValueError('matching nonempty observations and references required')
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    # Never join different sessions into an apparently continuous trajectory.
    if len({s['session_id'] for s in samples}) != 1:
        raise ValueError('mixed sessions: review each session separately')
    ordered = sorted(zip(refs, samples), key=lambda x: x[1]['captured_at_unix_s'])
    unique = []
    seen = set()
    for ref, sample in ordered:
        if sample['frame_id'] not in seen:
            unique.append((ref, sample))
            seen.add(sample['frame_id'])
    origin = unique[0][1]['captured_at_unix_s']

    def page(rows, name, width):
        height = width * 3 // 4
        canvas = Image.new('RGB', (2*width, len(rows)*(height+48)), '#eeeeee')
        draw = ImageDraw.Draw(canvas)
        for row, (ref, sample) in enumerate(rows):
            y = row*(height+48)
            dt = sample['captured_at_unix_s']-origin
            draw.text((8,y+3), f'event={ref} frame={sample["frame_id"]} t=+{dt:.3f}s', fill='black')
            for col, camera in enumerate(('global','wrist')):
                draw.text((col*width+8,y+23), camera.upper(), fill='black')
                encoded = sample.get('images_png_base64', {}).get(camera)
                if not encoded:
                    draw.text((col*width+12,y+70), 'MISSING IMAGE - UNKNOWN', fill='red')
                    continue
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    tile = ImageOps.pad(image.convert('RGB'), (width,height), color='black')
                    canvas.paste(tile, (col*width,y+48))
        path = cache/name
        canvas.save(path)
        return dict(path=str(path.resolve()), event_ids=[r for r,_ in rows])

    pages = [page(unique[i:i+4], f'timeline_{i//4+1:03d}.png', 480)
             for i in range(0,len(unique),4)]
    latest = page(unique[-1:], 'latest_pair.png', 640)
    endpoints = page([unique[0], unique[-1]] if len(unique)>1 else unique, 'endpoints.png', 480)
    indices = sorted(set(round(i*(len(unique)-1)/3) for i in range(4)))
    summary = page([unique[i] for i in indices], 'summary_pair.png', 480)
    identity = hashlib.sha256(str([(r,s['frame_id']) for r,s in unique]).encode()).hexdigest()[:16]
    times = [s['captured_at_unix_s'] for _,s in unique]
    return dict(format='dual_camera_timeline_v1', bundle_key=identity,
                layout='left=global; right=wrist; top_to_bottom=time; no frames skipped',
                observation_count=len(unique), event_ids=[r for r,_ in unique],
                max_gap_s=max((b-a for a,b in zip(times,times[1:])), default=0),
                missing_images=[dict(event_id=r,camera=c) for r,s in unique
                                for c in ('global','wrist') if not s.get('images_png_base64',{}).get(c)],
                selection='uniform samples including first and last; NOT anomaly detection',
                summary_is_sampled=len(indices)<len(unique),
                summary_omitted_count=len(unique)-len(indices),
                required_event_ids=summary['event_ids'],
                required_images=[summary,latest], summary=summary,
                timeline_pages=pages, latest_pair=latest, endpoints=endpoints,
                review_required=True,
                instruction='Open summary and latest pair. Summary is sampled, not continuous proof. Expand timeline for ambiguity, suspected events or continuous-state verification. Paths alone are not visual review.')


def validate_review(records, review):
    bundles = [r for r in records if r['type']=='observation_bundle']
    if not bundles:
        raise ValueError('observation bundle required: observe or review first')
    latest = bundles[-1]
    if not isinstance(review,dict) or review.get('bundle_ref') != latest['event_id']:
        raise ValueError('visual_review must reference the latest observation_bundle event')
    expected = set(latest['data'].get('required_event_ids', latest['data']['event_ids']))
    if not expected.issubset(set(review.get('reviewed_event_ids',[]))):
        raise ValueError('review must cover all bundle observation events')
    if any(not isinstance(review.get(k),str) or not review[k].strip()
           for k in ('global_findings','wrist_findings','process_findings')):
        raise ValueError('global, wrist and process visual findings required')
    return latest['event_id']


def render_decision_bundle(records, current_ref, cache):
    """Keep unreviewed execution evidence even after a fresh single observation."""
    from mj_env.task_runtime import pending_executions, motion_report
    from mj_env.image_regions import color_regions, track_region
    from mj_env.visual_servo import update_reference
    by_id = {r['event_id']: r for r in records}
    current = by_id[current_ref]['data']
    measured = {camera: color_regions(current, camera) for camera in ('global', 'wrist')}
    target_tracks = {}
    for record in reversed(records):
        if record['type'] == 'task_update':
            break
        if record['type'] == 'target_selection':
            target_tracks = record['data']['tracks']
            break
        if record['type'] == 'observation_bundle' and record['data'].get('target_tracks'):
            target_tracks = record['data']['target_tracks']
            break
    target_tracks = {camera: dict(track_region(track, measured[camera]),
                                  previous_observation_ref=track.get('observation_ref'), observation_ref=current_ref)
                     if track.get('session_id') == current['session_id'] else
                     dict(track, status='lost', reason='Robot/camera session changed; reselect target.')
                     for camera, track in target_tracks.items()}
    live = render_bundle([current], [current_ref], Path(cache)/'current')
    pending = pending_executions(records)
    last = next((r for r in reversed(records) if r['type']=='execution'), None)
    executions = list(pending)
    if last and last not in executions:
        executions.append(last)
    processes, images, required, missing = [], [], [], list(live['missing_images'])
    for index, execution in enumerate(executions):
        refs = execution['data']['observation_refs']
        samples = [by_id[ref]['data'] for ref in refs]
        # Keep discontinuous sessions in separate pages, never plot as continuous.
        groups = []
        for ref, sample in zip(refs, samples):
            if not groups or groups[-1][1][-1]['session_id'] != sample['session_id']:
                groups.append(([], []))
            groups[-1][0].append(ref)
            groups[-1][1].append(sample)
        parts = []
        for part, (part_refs, part_samples) in enumerate(groups):
            rendered = render_bundle(part_samples, part_refs, Path(cache)/f'process_{index:03d}'/f'session_{part:03d}')
            parts.append(rendered)
            # Summary already contains the same first/last pixels as endpoints.
            # Keep endpoints available in the manifest without sending duplicate
            # image tokens on every model request.
            images.append(rendered['summary'])
            required.extend(rendered['required_event_ids'])
            missing.extend(rendered['missing_images'])
        processes.append(dict(execution_ref=execution['event_id'], status=execution['data']['status'],
                              pending=execution in pending, parts=parts,
                              motion_report=motion_report(samples, refs, execution['data'])))
    images.append(live['latest_pair'])
    required.append(current_ref)
    return dict(format='decision_bundle_v2', role='decision',
                current_image_regions=measured['global'],
                wrist_image_regions=measured['wrist'],
                target_tracks=target_tracks,
                robot_visual_reference=update_reference(records,current_ref),
                current_observation_ref=current_ref, latest_role='current_observation',
                current_session_id=current['session_id'], current_captured_at_unix_s=current['captured_at_unix_s'],
                execution_refs=[e['event_id'] for e in executions],
                pending_execution_refs=[e['event_id'] for e in pending],
                processes=processes, current=live, missing_images=missing,
                required_event_ids=list(dict.fromkeys(required)),
                event_ids=list(dict.fromkeys(required)), required_images=images,
                instruction='Review process summaries and endpoints, then current pair. Fresh images never clear pending execution review. Measurements describe robot only.')
