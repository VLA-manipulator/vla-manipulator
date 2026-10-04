"""Persistent simulation tools for coding agents; run with --help."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import queue
import threading
import time
import uuid

import mujoco
import numpy as np
from PIL import Image

from mj_env.env import JOINT_NAMES, SO101TabletopEnv
from mj_env.objects import NAMES
from mj_env.scripts.scripted_pick import GraspSolver, ik_seed


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class SimulationTools:
    """Physics owns live MjData; the camera worker renders independent copies."""

    def __init__(self, *, headless=False, seed=3, target="cube_red"):
        self.env = SO101TabletopEnv(
            render_mode=None if headless else "human", image_size=(480, 640),
            max_episode_steps=100_000, target_object=target,
        )
        self.env.reset(seed=seed)
        self.solver = GraspSolver(self.env.model)
        self.planner_lock = threading.Lock()
        self.closed = False
        self.action = None
        self.snapshot_lock = threading.Lock()
        self.snapshot = None
        self.frame_id = 0
        self.session_id = uuid.uuid4().hex
        self.frames = queue.Queue(maxsize=1)
        self.sensor_stop = threading.Event()
        self.sensor_error = None
        self.sensor = threading.Thread(target=self.capture_loop, daemon=True)
        self.sensor.start()
        self.publish()
        timeout = time.monotonic() + 15
        while self.snapshot is None:
            if self.sensor_error is not None:
                raise RuntimeError(self.sensor_error)
            if time.monotonic() > timeout:
                raise RuntimeError("camera initialization timed out")
            time.sleep(.01)
        self.deadline = time.monotonic() + 1 / self.env.control_hz

    def advance(self):
        """Advance wall-clock physics while holding the last actuator targets."""
        while time.monotonic() >= self.deadline and not self.closed:
            if self.env.render_mode == "human" and not self.env.viewer_running:
                self.closed = True
                break
            target = self.env.data.ctrl[self.env._actuator_ids].copy()
            if self.action is not None and self.action["status"] == "running":
                action = self.action
                action["executed_steps"] += 1
                fraction = action["executed_steps"] / action["total_steps"]
                blend = fraction * fraction * (3 - 2 * fraction)
                target = action["start"] + (action["target"] - action["start"]) * blend
                if action["executed_steps"] >= action["total_steps"]:
                    action["status"] = "completed"
            self.env.step(target, capture_observation=False)
            self.publish()
            self.deadline += 1 / self.env.control_hz

    def state(self, ground_truth=False):
        env = self.env
        info = env._info()
        q = env.data.qpos[env._qpos_ids].copy()
        calibration = self.solver.calibration
        width = float(np.interp(q[5], calibration.angles, calibration.openings))
        rotation = env.data.site_xmat[env._tcp_site_id].reshape(3, 3)
        result = {
            "task": env.task, "target_object": env.target_object,
            "joint_names": JOINT_NAMES, "joint_positions_rad": q,
            "joint_targets_rad": env.data.ctrl[env._actuator_ids].copy(),
            "tcp_position_m": info["tcp_position"],
            "tcp_rotation_matrix": rotation.copy(),
            "jaw_approach_axis_robot": rotation[:, 0].copy(),
            "free_space_aperture_sweep": [
                {"width_m": w, "position_m": env.data.site_xpos[env._tcp_site_id]
                 + rotation @ calibration.offset_for(w)}
                for w in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
                if calibration.min_opening <= w <= calibration.max_opening
            ],
            "aperture_position_m": env.data.site_xpos[env._tcp_site_id]
            + rotation @ calibration.offset_for(width),
            "gripper_width_m": width, "simulation_time_s": float(env.data.time),
            "steps": info["steps"], "grasped": bool(info["grasped"]),
            "success": info["success"], "lift_height_m": info["lift_height"],
        }
        if ground_truth:
            result["ground_truth"] = {
                name: {"position_m": env.data.xpos[env._objects[name].body_id].copy(),
                       "quaternion_wxyz": env.data.xquat[env._objects[name].body_id].copy()}
                for name in env.objects_subset
            }
        return result

    @staticmethod
    def number(value, name, low, high):
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"{name} must be a number")
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite and in [{low}, {high}]")
        return float(value)

    def width(self, value):
        return self.number(value, "width_m", 0, self.solver.calibration.max_opening)

    def motion(self, target, duration):
        duration = self.number(duration, "duration_s", 0.05, 10)
        target = np.asarray(target, dtype=float)
        env = self.env
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("positions_rad must contain six finite numbers")
        if np.any(target < env.action_space.low) or np.any(target > env.action_space.high):
            raise ValueError("joint target outside limits; call info for allowed ranges")
        if self.action is not None and self.action["status"] == "running":
            raise ValueError("an action is running; use stop before submitting another motion")
        self.action = {
            "action_id": uuid.uuid4().hex, "status": "running", "executed_steps": 0,
            "total_steps": max(1, round(duration * env.control_hz)),
            "start": env.data.ctrl[env._actuator_ids].copy(), "target": target.copy(),
        }
        self.publish()
        return self.action_status()

    def action_status(self):
        if self.action is None:
            return {"status": "idle"}
        return {k: v for k, v in self.action.items() if k not in ("start", "target")}

    def publish(self):
        # Publish only immutable copies. HTTP encoding/network I/O never owns MjData.
        self.frame_id += 1
        captured = time.time()
        state = self.state(True)
        state.update(frame_id=self.frame_id, captured_at_unix_s=captured,
                     action=self.action_status())
        state["session_id"] = self.session_id
        data = mujoco.MjData(self.env.model)
        mujoco.mj_copyData(data, self.env.model, self.env.data)
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait((state, data))

    def capture_loop(self):
        renderer = None
        try:
            renderer = mujoco.Renderer(self.env.model, height=480, width=640)
            while not self.sensor_stop.is_set():
                try:
                    state, data = self.frames.get(timeout=.1)
                except queue.Empty:
                    continue
                images = []
                for camera in ("global_camera", "wrist_camera"):
                    renderer.update_scene(data, camera=camera)
                    images.append(renderer.render().copy())
                with self.snapshot_lock:
                    self.snapshot = (state, tuple(images))
        except Exception as error:
            self.sensor_error = str(error)
        finally:
            if renderer is not None:
                renderer.close()

    def close(self):
        self.sensor_stop.set()
        self.sensor.join(timeout=5)
        self.env.close()

    def observe(self, command, args):
        if not isinstance(args, dict) or set(args) - {"ground_truth"}:
            raise ValueError("only ground_truth is accepted")
        if not isinstance(args.get("ground_truth", False), bool):
            raise ValueError("ground_truth must be a boolean")
        if self.sensor_error is not None:
            raise RuntimeError(f"camera worker failed: {self.sensor_error}")
        with self.snapshot_lock:
            state, images = self.snapshot
        result = dict(state)
        if not args.get("ground_truth", False):
            result.pop("ground_truth", None)
        if command == "observe":
            result["images_png_base64"] = {}
            for name, pixels in zip(("global", "wrist"), images):
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer, format="PNG")
                result["images_png_base64"][name] = base64.b64encode(buffer.getvalue()).decode("ascii")
        result["snapshot_age_s"] = max(0, time.time() - result["captured_at_unix_s"])
        return result

    def call(self, command, args):
        allowed = {
            "info": set(), "state": {"ground_truth"},
            "observe": {"ground_truth"}, "reset": {"seed", "target", "task"},
            "joints": {"positions_rad", "duration_s"},
            "gripper": {"width_m", "duration_s"},
            "move": {"position_m", "hinge_yaw_rad", "aperture_width_m", "duration_s"},
            "wait": {"duration_s"}, "stop": set(), "shutdown": set(),
        }
        if command not in allowed:
            raise ValueError(f"unknown command {command!r}")
        if not isinstance(args, dict) or set(args) - allowed[command]:
            raise ValueError(f"allowed arguments for {command}: {sorted(allowed[command])}")
        if "ground_truth" in args and not isinstance(args["ground_truth"], bool):
            raise ValueError("ground_truth must be a boolean")
        if command == "info":
            return {
                "protocol_version": 2, "commands": {k: sorted(v) for k, v in allowed.items()},
                "objects": NAMES, "joint_names": JOINT_NAMES,
                "joint_limits_rad": np.column_stack((self.env.action_space.low, self.env.action_space.high)),
                "max_gripper_width_m": self.solver.calibration.max_opening,
                "control_hz": self.env.control_hz,
                "frame": "robot base origin; +x forward, +z up; table surface z=0; metres/radians",
                "idle_physics": "continuous wall-clock physics; last actuator targets held while agent thinks",
                "move": "jaw aperture XYZ with soft top-down orientation; inspect measured jaw_approach_axis_robot; preserves gripper command",
                "observation_pose_hint": {
                    "aperture_position_m": [0.20, 0.0, 0.16],
                    "source": "robot-specific nominal viewing posture, not a detected object position",
                    "collision_checked": False,
                    "instruction": "Approach in separately reviewed small segments only if images show a clear path. This is preparation, not target alignment."
                },
            }
        if command in ("state", "observe"):
            return self.observe(command, args)
        if command == "stop":
            if self.action is not None and self.action["status"] == "running":
                self.action["status"] = "cancelled"
            self.publish()
            return self.action_status()
        if command == "reset":
            target = args.get("target", self.env.target_object)
            if target not in NAMES:
                raise ValueError(f"target must be one of {NAMES}")
            seed = args.get("seed", 3)
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            options = {"target_object": target}
            if "task" in args:
                if not isinstance(args["task"], str) or not args["task"].strip():
                    raise ValueError("task must be a nonempty string")
                options["task"] = args["task"]
            self.action = None
            self.env.reset(seed=seed, options=options)
            self.publish()
            self.deadline = time.monotonic() + 1 / self.env.control_hz
            return self.state()
        if command == "shutdown":
            self.closed = True
            return {"shutdown": True}
        target = self.env.data.ctrl[self.env._actuator_ids].copy()
        if command == "joints":
            target = args["positions_rad"]
        elif command == "gripper":
            target[5] = self.solver.calibration.angle_for(self.width(args["width_m"]))
        elif command == "move":
            with self.planner_lock:
                target = self.plan_move(args)
        return self.motion(target, args.get("duration_s", 1.0))

    def plan_move(self, args):
        allowed = {"position_m", "hinge_yaw_rad", "aperture_width_m", "duration_s"}
        if not isinstance(args, dict) or set(args) - allowed:
            raise ValueError("invalid move arguments")
        self.number(args.get("duration_s", 1), "duration_s", .05, 10)
        snapshot = self.observe("state", {})
        target = np.asarray(snapshot["joint_targets_rad"]).copy()
        point = np.asarray(args["position_m"], dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("position_m must contain three finite numbers")
        yaw = self.number(args.get("hinge_yaw_rad", 0), "hinge_yaw_rad", -np.pi, np.pi)
        width = self.width(args.get("aperture_width_m", snapshot["gripper_width_m"]))
        offset = self.solver.calibration.offset_for(width)
        base = -np.pi / 2 - yaw
        solutions = []
        current = np.asarray(snapshot["joint_positions_rad"][:5]).copy()
        for angle in (base, base + np.pi):
            psi = (angle + np.pi) % (2 * np.pi) - np.pi
            # Current-pose seed avoids unnecessarily switching IK branches on
            # small moves. Analytic seed remains a fallback near singularities.
            for seed in (current, ik_seed(point, psi)):
                q = self.solver.solve_aperture(point, psi, offset, seed)
                if q is not None:
                    actual = self.solver.arm.fkine(q).t + self.solver.site_rotation(q) @ offset
                    if np.linalg.norm(actual - point) <= 0.003:
                        solutions.append(q)
        if not solutions:
            raise ValueError("unreachable aperture position/orientation; no motion executed")
        target[:5] = np.clip(
            min(solutions, key=lambda q: np.linalg.norm(q - current)),
            self.env.action_space.low[:5], self.env.action_space.high[:5],
        )
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--target", choices=NAMES, default="cube_red")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    pending = queue.Queue()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                if self.path != "/call":
                    raise ValueError("use POST /call")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("request must be 1..65536 bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or set(payload) - {"command", "args"}:
                    raise ValueError("expected {command, args}")
                command, values = payload["command"], payload.get("args", {})
                if command in ("observe", "state"):
                    value = sim.observe(command, values)
                elif command == "move":
                    # IK uses the solver's private MjData, never live simulation data.
                    with sim.planner_lock:
                        target = sim.plan_move(values)
                    future = Future()
                    pending.put(("joints", {"positions_rad": target,
                                 "duration_s": values.get("duration_s", 1)}, future))
                    value = future.result(timeout=30)
                elif command == "wait":
                    if not isinstance(values, dict) or set(values) - {"duration_s"}:
                        raise ValueError("wait accepts only duration_s")
                    duration = sim.number(values.get("duration_s", 1), "duration_s", .05, 10)
                    time.sleep(duration)
                    value = sim.observe("state", {})
                else:
                    future = Future()
                    pending.put((command, values, future))
                    value = future.result(timeout=30)
                status = 200
            except (ValueError, KeyError, TypeError) as error:
                status, value = 400, {"error": str(error)}
            except Exception as error:
                status, value = 500, {"error": str(error)}
            body = json.dumps(value, default=serializable, allow_nan=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    # Bind before opening a window, so a duplicate launch fails cleanly.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    sim = None
    worker = None
    try:
        sim = SimulationTools(headless=args.headless, seed=args.seed, target=args.target)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        print(f"Simulation tools ready: http://127.0.0.1:{args.port}/call", flush=True)
        while not sim.closed:
            sim.advance()
            try:
                command, values, future = pending.get(timeout=0.02)
            except queue.Empty:
                if not args.headless:
                    if not sim.env.viewer_running:
                        break
                    with sim.snapshot_lock:
                        sim.env._latest_observation_images = sim.snapshot[1]
                    sim.env.render()
                continue
            try:
                future.set_result(sim.call(command, values))
            except Exception as error:
                future.set_exception(error)
    finally:
        if worker is not None:
            server.shutdown()
        server.server_close()
        if sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
