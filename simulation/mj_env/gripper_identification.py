"""Active jaw-motion proposals from camera images and measured robot state only.

These proposals are not TCP/contact points. No camera or scene truth is used.
"""
import base64
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw


def pixels(sample, camera):
    data=base64.b64decode(sample['images_png_base64'][camera],validate=True)
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Invalid camera image')
    return image


def motion_candidates(before, after):
    old=cv2.cvtColor(before,cv2.COLOR_BGR2GRAY)
    new=cv2.cvtColor(after,cv2.COLOR_BGR2GRAY)
    points=cv2.goodFeaturesToTrack(old,maxCorners=600,qualityLevel=.015,minDistance=5)
    if points is None: return []
    opts=dict(winSize=(21,21),maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    moved,valid,_=cv2.calcOpticalFlowPyrLK(old,new,points,None,**opts)
    if moved is None: return []
    back,reverse,_=cv2.calcOpticalFlowPyrLK(new,old,moved,None,**opts)
    if back is None: return []
    displacement=np.linalg.norm(moved-points,axis=2).ravel()
    consistent=np.linalg.norm(back-points,axis=2).ravel()<1
    keep=(valid.ravel()>0)&(reverse.ravel()>0)&consistent&(displacement>1.5)&(displacement<100)
    selected=moved.reshape(-1,2)[keep]
    previous=points.reshape(-1,2)[keep]
    if not len(selected): return []
    # Cluster moving tracked pixels, not the full difference mask: disoccluded
    # background and shadows otherwise masquerade as the moving finger.
    mask=np.zeros(old.shape,np.uint8)
    for x,y in selected:
        cv2.circle(mask,(round(float(x)),round(float(y))),15,255,-1)
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    proposals=[]
    for index in range(1,count):
        members=np.array([labels[min(max(round(float(y)),0),old.shape[0]-1),
                                    min(max(round(float(x)),0),old.shape[1]-1)]==index for x,y in selected])
        if members.sum()<3: continue
        xy=selected[members];p=previous[members]
        x,y,w,h=map(int,stats[index,:4])
        proposals.append(dict(bbox_xywh=[x,y,w,h],tracked_points=int(members.sum()),
            median_displacement_px=np.median(xy-p,axis=0).round(2).tolist(),
            current_points_px=xy.round(1).tolist(),previous_points_px=p.round(1).tolist(),
            meaning='moving visual features only; identity and contact geometry unverified'))
    return sorted(proposals,key=lambda item:-item['tracked_points'])


def identify(samples):
    result=dict(state='gripper_unknown',reference_valid=False,alignment_motion_allowed=False,
                next_action='review_motion_candidates',cameras={},
                limitation='Motion identifies candidates, not fixed fingertip TCP or aperture center. No alignment or grasp conclusion.')
    if len(samples)<2:
        return dict(result,reason='Need before and after observations')
    first,last=samples[0],samples[-1]
    if any(s['session_id']!=first['session_id'] for s in samples):
        return dict(result,reason='Session changed')
    arm=np.asarray([s['joint_positions_rad'][:5] for s in samples])
    drift=float(np.max(np.abs(arm-arm[0])))
    width=last['gripper_width_m']-first['gripper_width_m']
    result.update(arm_joint_drift_rad=drift,measured_width_change_m=width,
                  start_width_m=first['gripper_width_m'],end_width_m=last['gripper_width_m'])
    if drift>.003:
        return dict(result,reason='Arm moved during jaw probe; image differences are not isolated jaw motion')
    if abs(width)<.006:
        return dict(result,reason='Insufficient measured jaw motion')
    for camera in ('global','wrist'):
        candidates=motion_candidates(pixels(first,camera),pixels(last,camera))
        for index,candidate in enumerate(candidates):
            candidate['candidate_id']=index
        result['cameras'][camera]=dict(candidates=candidates,
            status='candidate_needs_review' if len(candidates)==1 else 'ambiguous' if candidates else 'not_detected')
    if any(v['candidates'] for v in result['cameras'].values()):
        result['state']='motion_candidates_unverified'
    return result


def verify_return(first, middle, last, result):
    """Reject incidental motion using a separate, reviewed reverse jaw segment."""
    if len({s['session_id'] for s in (first,middle,last)})!=1:
        return dict(result,return_validation='session_changed')
    if abs(first['gripper_width_m']-last['gripper_width_m'])>.002:
        return dict(result,return_validation='not_at_original_width')
    if abs(first['gripper_width_m']-middle['gripper_width_m'])<.006:
        return dict(result,return_validation='insufficient_initial_probe')
    arm=np.asarray([s['joint_positions_rad'][:5] for s in (first,middle,last)])
    if np.max(np.abs(arm-arm[0]))>.003:
        return dict(result,return_validation='arm_pose_changed')
    cameras={}
    for camera in ('global','wrist'):
        proposals=motion_candidates(pixels(first,camera),pixels(middle,camera))
        gray_middle=cv2.cvtColor(pixels(middle,camera),cv2.COLOR_BGR2GRAY)
        gray_last=cv2.cvtColor(pixels(last,camera),cv2.COLOR_BGR2GRAY)
        verified=[]
        for proposal in proposals:
            p=np.asarray(proposal['current_points_px'],np.float32).reshape(-1,1,2)
            q,status,_=cv2.calcOpticalFlowPyrLK(gray_middle,gray_last,p,None,winSize=(21,21),maxLevel=3)
            if q is None: continue
            back,reverse,_=cv2.calcOpticalFlowPyrLK(gray_last,gray_middle,q,None,winSize=(21,21),maxLevel=3)
            if back is None: continue
            origins=np.asarray(proposal['previous_points_px'])
            errors=np.linalg.norm(q.reshape(-1,2)-origins,axis=1)
            keep=(status.ravel()>0)&(reverse.ravel()>0)&(errors<2)&(np.linalg.norm(back-p,axis=2).ravel()<1)
            if keep.sum()<3 or keep.mean()<.75: continue
            current=q.reshape(-1,2)[keep]
            low=np.floor(current.min(axis=0)-10).astype(int);high=np.ceil(current.max(axis=0)+10).astype(int)
            verified.append(dict(proposal,current_points_px=current.tolist(),previous_points_px=origins[keep].tolist(),
                bbox_xywh=[int(low[0]),int(low[1]),int(high[0]-low[0]),int(high[1]-low[1])],
                tracked_points=int(keep.sum()),return_error_max_px=float(errors[keep].max()),
                meaning='repeatable motion region; may be jaw OR its shadow, NOT a calibrated contact point'))
        for index,candidate in enumerate(verified):
            candidate['candidate_id']=index
        cameras[camera]=dict(candidates=verified,status='repeatable_candidate' if len(verified)==1 else 'ambiguous' if verified else 'not_detected')
    return dict(result,cameras=cameras,return_validation='checked',
        state='repeatable_motion_candidate' if any(v['candidates'] for v in cameras.values()) else 'gripper_unknown',
        next_action='review_candidate_identity_then_calibrate_geometry')


def annotated_pair(samples,result):
    candidates=[(camera,item) for camera in ('global','wrist') for item in result['cameras'].get(camera,{}).get('candidates',[])]
    canvas=Image.new('RGB',(1280,1056+280*((len(candidates)+3)//4)),'#eeeeee')
    draw=ImageDraw.Draw(canvas)
    for row,sample in enumerate((samples[0],samples[-1])):
        for col,camera in enumerate(('global','wrist')):
            xoff,yoff=col*640,row*528
            image=Image.fromarray(cv2.cvtColor(pixels(sample,camera),cv2.COLOR_BGR2RGB))
            canvas.paste(image,(xoff,yoff+48))
            draw.text((xoff+8,yoff+4),f'{"BEFORE" if row==0 else "AFTER"} | {camera.upper()} | width={sample["gripper_width_m"]:.4f}m',fill='black')
            draw.text((xoff+8,yoff+24),'CONTACT REFERENCE UNKNOWN | cyan: motion, may include shadows',fill='black')
            for i,item in enumerate(result['cameras'].get(camera,{}).get('candidates',[])):
                points=item['previous_points_px' if row==0 else 'current_points_px']
                for x,y in points:
                    draw.ellipse((xoff+x-3,yoff+48+y-3,xoff+x+3,yoff+48+y+3),outline='cyan',width=2)
                if row==1:
                    x,y,w,h=item['bbox_xywh']
                    draw.rectangle((xoff+x,yoff+48+y,xoff+x+w,yoff+48+y+h),outline='cyan',width=2)
                    draw.text((xoff+x,yoff+48+y),f'{camera[0].upper()}{i}',fill='cyan',stroke_width=1,stroke_fill='black')
    for index,(camera,item) in enumerate(candidates):
        x,y,w,h=item['bbox_xywh']
        source=Image.fromarray(cv2.cvtColor(pixels(samples[-1],camera),cv2.COLOR_BGR2RGB))
        crop=source.crop((max(0,x-20),max(0,y-20),min(640,x+w+20),min(480,y+h+20)))
        from PIL import ImageOps
        tile=ImageOps.contain(crop,(304,230))
        xoff=(index%4)*320;yoff=1056+(index//4)*280
        canvas.paste(tile,(xoff+8,yoff+40))
        draw.text((xoff+8,yoff+4),f'{camera[0].upper()}{item.get("candidate_id",index)} | {camera.upper()} | AFTER crop',fill='black')
        draw.text((xoff+8,yoff+22),f'original bbox={item["bbox_xywh"]}',fill='black')
    return canvas


def encoded_pair(samples,result):
    stream=io.BytesIO();annotated_pair(samples,result).save(stream,format='PNG')
    return base64.b64encode(stream.getvalue()).decode('ascii')
