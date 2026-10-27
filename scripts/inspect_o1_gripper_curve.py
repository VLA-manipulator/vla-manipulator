"""Print sparse O1 gripper actions from saved staging episodes."""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    with np.load(args.episode, allow_pickle=False) as data:
        actions = data["actions"]
        state = data["state"]
    print("frames", len(actions), "gripper_minmax", actions[:, 5].min(), actions[:, 5].max())
    for index in range(0, len(actions), 10):
        print(index, round(float(state[index, 5]), 2), round(float(actions[index, 5]), 2))


if __name__ == "__main__":
    main()
