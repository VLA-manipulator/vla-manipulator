import base64
import unittest
import cv2
import numpy as np
from mj_env.gripper_identification import identify,verify_return,annotated_pair


class IdentificationTests(unittest.TestCase):
    def sample(self,x,width):
        rng=np.random.default_rng(5)
        image=np.full((480,640,3),180,np.uint8)
        image[200:250,x:x+50]=rng.integers(0,255,(50,50,3),dtype=np.uint8)
        encoded=base64.b64encode(cv2.imencode('.png',image)[1]).decode()
        return dict(session_id='s',joint_positions_rad=[0]*6,gripper_width_m=width,
                    images_png_base64={'global':encoded,'wrist':encoded})

    def test_candidate_is_never_contact_reference(self):
        a,b=self.sample(100,.06),self.sample(110,.045)
        result=identify([a,b])
        self.assertEqual(result['state'],'motion_candidates_unverified')
        self.assertFalse(result['reference_valid'])
        self.assertFalse(result['alignment_motion_allowed'])
        self.assertGreater(len(result['cameras']['wrist']['candidates']),0)
        self.assertGreater(annotated_pair([a,b],result).height,1056)

    def test_arm_motion_and_session_change_reject(self):
        a,b=self.sample(100,.06),self.sample(110,.045)
        b['joint_positions_rad'][1]=.01
        self.assertEqual(identify([a,b])['state'],'gripper_unknown')
        b['session_id']='other'
        self.assertEqual(identify([a,b])['reason'],'Session changed')

    def test_stationary_images_do_not_establish_jaw(self):
        result=identify([self.sample(100,.06),self.sample(100,.045)])
        self.assertEqual(result['state'],'gripper_unknown')

    def test_return_requires_repeatable_features(self):
        a,b,c=self.sample(100,.06),self.sample(110,.045),self.sample(100,.06)
        result=verify_return(a,b,c,identify([b,c]))
        self.assertEqual(result['state'],'repeatable_motion_candidate')
        self.assertFalse(result['reference_valid'])
        self.assertFalse(result['alignment_motion_allowed'])
        d=self.sample(130,.06)
        self.assertEqual(verify_return(a,b,d,identify([b,d]))['state'],'gripper_unknown')


if __name__=='__main__':unittest.main()
