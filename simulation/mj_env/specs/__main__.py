"""``python -m mj_env.specs`` prints the level ladder."""

from __future__ import annotations

from . import LEVELS, describe_all


def main() -> int:
    print(describe_all())
    print()
    for task_id, spec in LEVELS.items():
        print(f"{task_id}")
        print(f"  场景     : {spec.scene}")
        print(f"  判据     : {spec.criterion_text}  (hold {spec.hold_s}s)")
        print(f"  指令池   : {len(spec.instructions)} 训练 + {len(spec.holdout)} hold-out")
        print(f"  spec_hash: {spec.hash()}")
        if spec.notes:
            print(f"  备注     : {spec.notes}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())