"""Pixel-only color regions; no world geometry or simulated object identity."""
import base64
import cv2
import numpy as np


def track_region(reference, measured, max_distance_px=80):
    """Conservative 2D association, never an object identity/3D assertion."""
    if reference.get('status') == 'lost':
        return dict(reference, reason='Reselect from current images; lost tracks never auto-reacquire.')
    region = reference['region']
    candidates = []
    for candidate in measured.get('regions', []):
        if candidate['color'] != region['color']:
            continue
        distance = float(np.linalg.norm(np.array(candidate['center_px']) - region['center_px']))
        ratio = candidate['area_px'] / max(region['area_px'], 1)
        if distance <= max_distance_px and .3 <= ratio <= 3:
            candidates.append((distance, candidate))
    candidates.sort(key=lambda pair: pair[0])
    if not candidates or (len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 15):
        return dict(reference, status='lost', reason='Missing, occluded, or ambiguous color component; inspect and reselect.')
    candidate = candidates[0][1]
    return dict(reference, status='tracked', region=candidate,
                delta_px=[round(b-a, 1) for a,b in zip(region['center_px'], candidate['center_px'])],
                area_ratio=round(candidate['area_px']/max(region['area_px'], 1), 3),
                limitation='Image association only. Constant global pixels do not prove constant height; constant wrist pixels alone do not prove grasp.')


def color_regions(sample, camera='global'):
    if camera not in ('global', 'wrist'):
        raise ValueError('camera must be global or wrist')
    encoded=sample.get('images_png_base64',{}).get(camera)
    if not encoded: return {'camera':camera, 'status':'missing image'}
    pixels=cv2.imdecode(np.frombuffer(base64.b64decode(encoded),dtype=np.uint8),cv2.IMREAD_COLOR)
    if pixels is None: return {'status':'invalid image'}
    hsv=cv2.cvtColor(pixels,cv2.COLOR_BGR2HSV)
    output=[]
    for color,intervals in {'blue':[(95,130)],'red':[(0,8),(170,179)],'green':[(40,85)],
                            'purple':[(131,165)],'orange':[(9,24)]}.items():
        mask=np.zeros(hsv.shape[:2],dtype=np.uint8)
        for lo,hi in intervals:
            mask |= cv2.inRange(hsv,np.array([lo,100,55]),np.array([hi,255,255]))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:6]:
            area=cv2.contourArea(contour)
            if area<40: continue
            x,y,w,h=cv2.boundingRect(contour)
            moments=cv2.moments(contour)
            perimeter=cv2.arcLength(contour,True)
            output.append(dict(color=color,center_px=[round(moments['m10']/area,1),round(moments['m01']/area,1)],
                               bbox_xywh=[x,y,w,h],area_px=round(area),
                               circularity=round(4*np.pi*area/(perimeter*perimeter),3) if perimeter else None))
    for index, region in enumerate(output):
        region['region_index'] = index
    return dict(camera=camera,pixel_frame='original single camera; origin top-left; x right, y down',
                image_size=[pixels.shape[1],pixels.shape[0]],regions=output,
                limitation='Color components only; identity and depth unknown. Shadows/occlusion can split or merge regions. No 3D coordinates.')
