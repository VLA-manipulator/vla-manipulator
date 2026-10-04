"""Joint ordering shared by hardware and kinematics, without model imports."""
ARM_JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL_JOINT_NAMES = (*ARM_JOINT_NAMES, "gripper")
ACTION_NAMES = tuple(f"{name}.pos" for name in ALL_JOINT_NAMES)
