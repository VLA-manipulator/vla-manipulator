"""Call the persistent simulation; save camera images for an agent to view."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("info", "state", "observe", "reset", "joints", "gripper", "move", "wait", "stop", "shutdown"))
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--args", default="{}", help="JSON object")
    group.add_argument("--args-file", type=Path, help="UTF-8 JSON file (avoids shell quoting)")
    parser.add_argument("--output", type=Path, default=Path(".tmp/agent_observation"))
    args = parser.parse_args()
    try:
        values = json.loads(args.args_file.read_text(encoding="utf-8-sig") if args.args_file else args.args)
        body = json.dumps({"command": args.command, "args": values}).encode("utf-8")
        request = urllib.request.Request(args.server.rstrip("/") + "/call", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        images = result.pop("images_png_base64", {})
        if images:
            args.output = args.output / result.get("session_id", "session") / str(result.get("frame_id", "frame"))
            args.output.mkdir(parents=True, exist_ok=True)
            result["images"] = {}
            for name in ("global", "wrist"):
                if name in images:
                    path = args.output / f"{name}.png"
                    path.write_bytes(base64.b64decode(images[name], validate=True))
                    result["images"][name] = str(path.resolve())
            (args.output / "state.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8"), file=sys.stderr)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
