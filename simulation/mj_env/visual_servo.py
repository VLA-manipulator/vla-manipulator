"""Local image Jacobian from a visually selected robot point and measured poses.

No camera extrinsics, object coordinates, renderer projections, or contact flags.
The selected pixel must represent the fixed fingertip TCP reference. Its visual
identification remains a human/model responsibility; the fit cannot certify it.
"""
import base64
import cv2
import numpy as np


def visual_relation(reference, target):
    result=dict(camera='global',units='pixels',
        interpretation='The global camera is fixed. A stationary ungrasped target should not move when the robot moves. Target pixel constancy says nothing about robot motion direction. Compare the SAME robot reference relative to the target.',
        status='unavailable')
    if not reference or reference.get('status')!='tracked' or target.get('status')!='tracked':
        result['reason']='Need both a tracked robot reference and tracked target'
        return result
    result.update(status='measured',target_minus_reference_px=(
        np.asarray(target['region']['center_px'])-reference['pixel_xy']).tolist(),
        limitation='2D separation only; not height, aperture alignment, or grasp evidence')
    return result


def gray(sample):
    encoded=sample.get('images_png_base64',{}).get('global')
    if not encoded:
        raise ValueError('Global image required')
    image=cv2.imdecode(np.frombuffer(base64.b64decode(encoded),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError('Invalid global image')
    return image


def track_pixel(before, after, point):
    previous,current=gray(before),gray(after)
    p=np.asarray(point,dtype=np.float32).reshape(1,1,2)
    options=dict(winSize=(31,31),maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q,status,_=cv2.calcOpticalFlowPyrLK(previous,current,p,None,**options)
    if q is None or not status[0,0]:
        raise ValueError('Robot feature lost')
    back,status,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None,**options)
    if back is None or not status[0,0] or np.linalg.norm(back-p)>1.:
        raise ValueError('Robot feature forward/backward tracking disagrees')
    x,y=map(float,q[0,0])
    if not (16<=x<current.shape[1]-16 and 16<=y<current.shape[0]-16):
        raise ValueError('Robot feature near/outside image boundary')
    return [x,y]


def fit_mapping(anchors):
    if len(anchors)<5:
        return dict(status='calibration_needed',reason='Need independent x, y, z probes plus an additional validation position (five anchors minimum)')
    xyz=np.asarray([a['tcp_position_m'] for a in anchors],dtype=float)
    pixels=np.asarray([a['pixel_xy'] for a in anchors],dtype=float)
    dx=xyz-xyz[0]; du=pixels-pixels[0]
    singular=np.linalg.svd(dx,compute_uv=False)
    if singular[-1]<.003 or singular[0]/singular[-1]>30:
        return dict(status='calibration_needed',reason='Probe displacements do not span three independent axes')
    coefficients=np.linalg.lstsq(dx,du,rcond=None)[0]
    residual=float(np.max(np.linalg.norm(dx@coefficients-du,axis=1)))
    jacobian=coefficients.T
    if residual>2 or np.linalg.cond(jacobian[:,:2])>15:
        return dict(status='invalid',reason='Inconsistent feature motion or ill-conditioned XY mapping',max_residual_px=residual)
    # A just-determined fit can exactly explain a mistracked pixel. Require a
    # held-out observation predictable from independent remaining probes.
    validation_errors=[]
    for index in range(1,len(anchors)):
        keep=np.arange(len(anchors))!=index
        remaining=dx[keep]
        values=np.linalg.svd(remaining,compute_uv=False)
        if values[-1]<.003 or values[0]/values[-1]>30:
            continue
        fitted=np.linalg.lstsq(remaining,du[keep],rcond=None)[0]
        validation_errors.append(float(np.linalg.norm(dx[index]@fitted-du[index])))
    if not validation_errors:
        return dict(status='calibration_needed',reason='Additional independent validation position required')
    validation_error=max(validation_errors)
    if validation_error>2:
        return dict(status='invalid',reason='Held-out feature motion is inconsistent',max_validation_error_px=validation_error)
    return dict(status='ready',jacobian_px_per_m=jacobian.tolist(),max_residual_px=residual,
                max_validation_error_px=validation_error,
                anchor_count=len(anchors),
                limitation='Local approximation; feature identity and depth still require visual review. Fit alone does not establish grasp.')


def update_reference(records, current_ref):
    by_id={r['event_id']:r for r in records}
    reference=None
    for record in reversed(records):
        if record['type']=='task_update': break
        if record['type']=='robot_reference_selection':
            reference=record['data'];break
        if record['type']=='observation_bundle' and record['data'].get('robot_visual_reference'):
            reference=record['data']['robot_visual_reference'];break
    if not reference: return None
    if reference.get('status')=='lost': return reference
    current=by_id[current_ref]['data']
    previous=by_id[reference['observation_ref']]['data']
    try:
        if current['session_id']!=previous['session_id']:
            raise ValueError('Session changed; reference must be selected again')
        # Reject orientation changes that can invalidate point-to-TCP association.
        r0=np.asarray(reference['initial_rotation'],dtype=float)
        r1=np.asarray(current['tcp_rotation_matrix'],dtype=float)
        angle=float(np.arccos(np.clip((np.trace(r0.T@r1)-1)/2,-1,1)))
        if angle>.10:
            raise ValueError('Robot orientation changed over 0.10rad; recalibrate local mapping')
        if np.linalg.norm(np.asarray(current['tcp_position_m'])-reference['anchors'][0]['tcp_position_m'])>.12:
            raise ValueError('Outside 0.12m local calibration neighborhood')
        point=track_pixel(previous,current,reference['pixel_xy'])
        anchors=list(reference['anchors'])
        if min(np.linalg.norm(np.asarray(current['tcp_position_m'])-a['tcp_position_m']) for a in anchors)>.003:
            anchors.append(dict(observation_ref=current_ref,tcp_position_m=current['tcp_position_m'],pixel_xy=point))
        return dict(reference,status='tracked',observation_ref=current_ref,pixel_xy=point,
                    anchors=anchors,mapping=fit_mapping(anchors))
    except (ValueError,KeyError,cv2.error) as error:
        return dict(reference,status='lost',reason=str(error))


def alignment_delta(reference, target, robot, width, max_step):
    if reference.get('status')!='tracked' or reference.get('mapping',{}).get('status')!='ready':
        raise ValueError('Tracked robot reference and validated local mapping required')
    if target.get('status')!='tracked': raise ValueError('Visible tracked target required')
    sweep=sorted(robot.get('free_space_aperture_sweep',[]),key=lambda row:row['width_m'])
    if not sweep or not sweep[0]['width_m']<=width<=sweep[-1]['width_m']:
        raise ValueError('Requested alignment width outside robot-provided aperture sweep')
    widths=[row['width_m'] for row in sweep]
    aperture=np.array([np.interp(width,widths,[row['position_m'][axis] for row in sweep]) for axis in range(3)])
    jacobian=np.asarray(reference['mapping']['jacobian_px_per_m'])
    projected=np.asarray(reference['pixel_xy'])+jacobian@(aperture-np.asarray(robot['tcp_position_m']))
    error=np.asarray(target['region']['center_px'])-projected
    delta=np.linalg.solve(jacobian[:,:2],error)
    norm=float(np.linalg.norm(delta))
    if norm>max_step: delta*=max_step/norm
    return dict(delta_m=[float(delta[0]),float(delta[1]),0.],error_px=error.tolist(),
                projected_aperture_px=projected.tolist(),alignment_only=True,
                aligned_in_projection=bool(np.linalg.norm(error)<3),
                limitation='Projection alignment is not 3D alignment or contact; descend separately with visual review.')
