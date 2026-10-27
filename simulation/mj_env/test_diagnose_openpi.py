"""Unit checks for the conservative symptom labels in diagnose_openpi."""

from __future__ import annotations

import unittest

import numpy as np

from mj_env.scripts.diagnose_openpi import diagnose, summarize_chunk_seams, summarize_seams_near_event


def record(**overrides):
    result = {
        "success": False,
        "min_tcp_target_distance_m": 0.02,
        "ever_grasped": False,
        "max_lift_height_m": 0.0,
        "wrong_object_two_pad_contact_steps": 0,
        "clipped_action_steps": 0,
        "mean_joint_tracking_error_rad": 0.01,
    }
    result.update(overrides)
    return result


class DiagnoseLabelsTest(unittest.TestCase):
    def test_close_without_grasp(self):
        self.assertEqual(diagnose(record()), ["approached_but_no_two_pad_grasp"])

    def test_brief_grasp_without_lift_and_clipping(self):
        labels = diagnose(record(ever_grasped=True, max_lift_height_m=0.02, clipped_action_steps=7))
        self.assertEqual(labels, ["grasped_but_not_lifted_6cm", "action_outside_motor_range"])

    def test_success_is_not_mislabeled(self):
        self.assertEqual(diagnose(record(success=True, clipped_action_steps=7)), ["success"])

    def test_center_distance_is_only_proxy(self):
        self.assertEqual(
            diagnose(record(min_tcp_target_distance_m=0.08)),
            ["tcp_not_within_5cm_of_object_center"],
        )

    def test_chunk_seam_metrics(self):
        first = np.zeros((10, 6), dtype=np.float32)
        second = np.zeros((10, 6), dtype=np.float32)
        first[:, 0] = np.arange(10)
        second[:, 0] = np.arange(10) + 7
        metrics = summarize_chunk_seams([first, second], replan_steps=5)
        self.assertEqual(metrics["within_chunk_target_step_l2_rad"], {"count": 8, "mean": 1.0, "p95": 1.0})
        self.assertEqual(metrics["boundary_target_step_l2_rad"], {"count": 1, "mean": 3.0, "p95": 3.0})
        self.assertEqual(metrics["overlap_prediction_disagreement_l2_rad"], {"count": 1, "mean": 2.0, "p95": 2.0})

    def test_single_chunk_has_no_boundary(self):
        metrics = summarize_chunk_seams([np.zeros((10, 6), dtype=np.float32)], replan_steps=5)
        self.assertIsNone(metrics["boundary_target_step_l2_rad"]["mean"])

    def test_seams_are_split_around_contact(self):
        events = [
            {"step": step, "target_jump_l2_rad": float(step),
             "observed_step_l2_rad": float(step) / 2,
             "overlap_disagreement_l2_rad": float(step) / 4}
            for step in (5, 30, 40, 60, 65)
        ]
        windows = summarize_seams_near_event(events, anchor_step=50)
        self.assertEqual([windows[key]["count"] for key in
                          ("earlier", "before_20_steps", "after_20_steps", "later")], [1, 2, 2, 0])
        self.assertEqual(windows["before_20_steps"]["mean_target_jump_l2_rad"], 35.0)


if __name__ == "__main__":
    unittest.main()
