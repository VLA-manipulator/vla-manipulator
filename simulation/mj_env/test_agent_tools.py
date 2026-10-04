"""Exercise concurrent HTTP control and observations against real MuJoCo."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
import numpy as np


class AgentToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.url = f"http://127.0.0.1:{port}/call"
        cls.process = subprocess.Popen(
            [sys.executable, "-m", "mj_env.agent_server", "--headless", "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(150):
            try:
                cls.call("state")
                return
            except OSError:
                time.sleep(.1)
        cls.process.terminate()
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        try:
            cls.call("shutdown")
            cls.process.wait(timeout=10)
        finally:
            if cls.process.poll() is None:
                cls.process.kill()
                cls.process.wait()

    @classmethod
    def call(cls, command, args=None):
        request = urllib.request.Request(cls.url, data=json.dumps(
            {"command": command, "args": args or {}}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    def setUp(self):
        self.call("reset", {"seed": 3, "target": "cube_red"})
        time.sleep(.15)

    def test_observe_during_motion_and_concurrent_wait(self):
        before = self.call("observe")
        begin = time.monotonic()
        action = self.call("gripper", {"width_m": .015, "duration_s": 3})
        self.assertLess(time.monotonic() - begin, 1)
        self.assertEqual(action["status"], "running")
        with ThreadPoolExecutor() as pool:
            waiting = pool.submit(self.call, "wait", {"duration_s": 2})
            time.sleep(.4)
            mid = self.call("observe")
            self.assertFalse(waiting.done())
            self.assertEqual(mid["action"]["action_id"], action["action_id"])
            self.assertEqual(mid["action"]["status"], "running")
            self.assertGreater(mid["frame_id"], before["frame_id"])
            self.assertGreater(mid["simulation_time_s"], before["simulation_time_s"])
            self.assertLess(mid["snapshot_age_s"], 1)
            self.assertNotEqual(mid["joint_positions_rad"], before["joint_positions_rad"])
            self.assertNotEqual(mid["images_png_base64"]["wrist"], before["images_png_base64"]["wrist"])
            for pixels in mid["images_png_base64"].values():
                self.assertTrue(base64.b64decode(pixels).startswith(bytes.fromhex("89504e47")))
            self.assertNotIn("ground_truth", mid)
            self.assertEqual(len(self.call("state", {"ground_truth": True})["ground_truth"]), 8)
            waiting.result()
        self.call("wait", {"duration_s": 1.2})
        self.assertEqual(self.call("state")["action"]["status"], "completed")

    def test_stop_and_busy(self):
        self.call("gripper", {"width_m": .015, "duration_s": 3})
        with self.assertRaises(urllib.error.HTTPError):
            self.call("gripper", {"width_m": .07})
        self.assertEqual(self.call("stop")["status"], "cancelled")
        self.assertEqual(self.call("gripper", {"width_m": .07})["status"], "running")
        self.call("stop")

    def test_invalid_and_ik(self):
        for command, args in [("joints", {"positions_rad": [20]*6}),
                              ("wait", {"duration_s": -1}),
                              ("state", {"ground_truth": "false"})]:
            with self.assertRaises(urllib.error.HTTPError):
                self.call(command, args)
        result = self.call("move", {"position_m": [.2, 0, .16], "duration_s": 1})
        self.assertEqual(result["status"], "running")
        self.call("wait", {"duration_s": 1.5})
        self.assertEqual(self.call("state")["action"]["status"], "completed")

    def test_robot_pose_and_free_space_closure_geometry(self):
        state=self.call('state')
        rotation=np.asarray(state['tcp_rotation_matrix'])
        np.testing.assert_allclose(rotation.T @ rotation,np.eye(3),atol=1e-6)
        np.testing.assert_allclose(state['jaw_approach_axis_robot'],rotation[:,0])
        sweep=state['free_space_aperture_sweep']
        self.assertGreater(len(sweep),1)
        self.assertTrue(all(set(row)=={'width_m','position_m'} for row in sweep))
        self.assertGreater(np.linalg.norm(np.asarray(sweep[0]['position_m'])-sweep[-1]['position_m']),.01)


if __name__ == "__main__":
    unittest.main()
