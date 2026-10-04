"""Sensor privacy and terminal-condition tests; no robot motion."""
import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from mj_env.closed_loop import check_start, observe, read_sample, stationary


TERMINAL = dict(joint_speed_max_rad_s=.02, tcp_speed_max_m_s=.002, stable_window_s=.3)


def sample(index):
    return dict(session_id='test', frame_id=index, captured_at_unix_s=index*.05,
                snapshot_age_s=.01, joint_positions_rad=[0.]*6, joint_targets_rad=[0.]*6,
                tcp_position_m=[0., 0., .1], aperture_position_m=[0., 0., .1],
                gripper_width_m=.05, action={'status': 'completed'})


class ClosedLoopTests(unittest.TestCase):
    def test_sensor_whitelist_and_dual_images(self):
        raw = sample(1)
        raw.update(ground_truth={'secret': True}, success=True, grasped=True, lift_height_m=999)
        buffer = io.BytesIO()
        Image.new('RGB', (4, 4)).save(buffer, format='PNG')
        raw['images_png_base64'] = {k: base64.b64encode(buffer.getvalue()).decode() for k in ('global', 'wrist')}
        with tempfile.TemporaryDirectory() as directory, patch('mj_env.closed_loop.call', return_value=raw):
            result = observe('unused', Path(directory))
            for field in ('ground_truth', 'success', 'grasped', 'lift_height_m'):
                self.assertNotIn(field, result)
            for path in result['images'].values():
                self.assertTrue(Path(path).is_file())

    def test_static_window(self):
        self.assertTrue(stationary([sample(i) for i in range(8)], TERMINAL))
        self.assertFalse(stationary([sample(i) for i in range(3)], TERMINAL))

    def test_running_moving_stale_duplicate_session_and_gap_are_rejected(self):
        cases = []
        for field, value in [('action', {'status':'running'}), ('joint_positions_rad', [1.]*6),
                             ('tcp_position_m', [1.,0.,.1]), ('snapshot_age_s', 1),
                             ('frame_id', 6), ('session_id', 'changed'), ('captured_at_unix_s', 2)]:
            data = [sample(i) for i in range(8)]
            data[-1][field] = value
            cases.append(data)
        for data in cases:
            self.assertFalse(stationary(data, TERMINAL))

    def test_start_mismatch(self):
        check_start(sample(0), sample(1))
        for field, value in [('session_id','other'), ('joint_positions_rad',[.1]*6), ('tcp_position_m',[1.,0.,0.])]:
            actual = sample(1)
            actual[field] = value
            with self.assertRaises(ValueError):
                check_start(sample(0), actual)

    def test_exact_sample_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'samples.jsonl'
            path.write_text('\n'.join(json.dumps(sample(i)) for i in range(2)), encoding='utf-8')
            self.assertEqual(read_sample(str(path)+'#1')['frame_id'], 1)


if __name__ == '__main__':
    unittest.main()
