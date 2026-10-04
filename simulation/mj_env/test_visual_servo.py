import unittest
import numpy as np
from mj_env.visual_servo import fit_mapping,alignment_delta,visual_relation


class VisualServoTests(unittest.TestCase):
    def test_relation_never_infers_robot_motion_from_target_alone(self):
        target=dict(status='tracked',region=dict(center_px=[100,120]))
        self.assertEqual(visual_relation(None,target)['status'],'unavailable')
        result=visual_relation(dict(status='tracked',pixel_xy=[80,90]),target)
        self.assertEqual(result['target_minus_reference_px'],[20,30])
        self.assertNotIn('delta_m',result)

    def anchors(self):
        matrix=np.array([[1000,200,50],[100,1200,-80]])
        xyz=np.array([[0,0,0],[.02,0,0],[.02,.02,0],[.02,.02,.02],[0,.01,.01]])
        return [dict(tcp_position_m=p.tolist(),pixel_xy=(np.array([200,200])+matrix@p).tolist()) for p in xyz]

    def test_mapping_uses_observed_three_axis_displacements(self):
        result=fit_mapping(self.anchors())
        self.assertEqual(result['status'],'ready')
        np.testing.assert_allclose(result['jacobian_px_per_m'],[[1000,200,50],[100,1200,-80]],atol=1e-6)
        self.assertEqual(fit_mapping(self.anchors()[:3])['status'],'calibration_needed')
        self.assertEqual(fit_mapping(self.anchors()[:4])['status'],'calibration_needed')
        self.assertLess(result['max_validation_error_px'],1e-6)

    def test_inconsistent_feature_is_rejected(self):
        anchors=self.anchors();anchors[-1]['pixel_xy'][0]+=25
        self.assertEqual(fit_mapping(anchors)['status'],'invalid')

    def test_step_is_bounded_and_does_not_infer_depth(self):
        reference=dict(status='tracked',pixel_xy=[200,200],mapping=fit_mapping(self.anchors()))
        target=dict(status='tracked',region=dict(center_px=[240,220]))
        robot=dict(tcp_position_m=[0,0,0],free_space_aperture_sweep=[
            dict(width_m=.01,position_m=[0,0,0]),dict(width_m=.06,position_m=[0,0,0])])
        result=alignment_delta(reference,target,robot,.03,.008)
        self.assertAlmostEqual(np.linalg.norm(result['delta_m']),.008)
        self.assertEqual(result['delta_m'][2],0)
        self.assertTrue(result['alignment_only'])
        with self.assertRaises(ValueError):alignment_delta(reference,dict(status='lost'),robot,.03,.008)
        with self.assertRaises(ValueError):alignment_delta(reference,target,robot,.08,.008)
