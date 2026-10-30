"""Serve an SO101 Pi0.5 checkpoint with optional guided RTC requests.

Run from an OpenPI checkout with this repository on PYTHONPATH::

    PYTHONPATH=/path/to/vla-manipulator uv run /path/to/vla-manipulator/openpi_ext/serve_rtc.py \
      --config pi05_so101_lora_pick8_o1_slow240_5k --checkpoint /path/to/29999
"""

from __future__ import annotations

import argparse
import logging

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as train_config

from openpi_ext.rtc_policy import RTCPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=5)
    args = parser.parse_args()
    base = policy_config.create_trained_policy(train_config.get_config(args.config), args.checkpoint)
    base._sample_kwargs = {**base._sample_kwargs, "num_steps": args.num_steps}
    policy = RTCPolicy(base)
    logging.info("RTC policy ready on port %s", args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
