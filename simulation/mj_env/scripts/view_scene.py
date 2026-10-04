"""Open the SO101 MuJoCo scene without any policy or network service."""

from __future__ import annotations

import argparse
import time

from mj_env import objects
from mj_env.env import SO101TabletopEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", choices=objects.NAMES, default="cube_red")
    args = parser.parse_args()

    env = SO101TabletopEnv(render_mode="human", target_object=args.target)
    try:
        env.reset(seed=args.seed)
        env.render()
        while env.viewer_running:
            env.render()
            time.sleep(1.0 / 60.0)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
