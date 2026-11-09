# G1 whole-body + 6-DoF hand modality config for GR00T N1.7 (EmbodimentTag.NEW_EMBODIMENT).
# Body layout follows IsaacLab G1_INSPIRE_FTP_CFG (29 body + 12 hand DoF).
# State split: robot_q_current[36] = root(7) + legs(12) + waist(3) + arms(14); hand_state[12] = 6+6
# Action split: robot_q_desired[36] same layout; hand_cmd[12] same layout
# Drops base_height_command/navigate_command (dataset has no teleop.* features)

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

g1_pillow_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "waist",
            "left_leg",
            "right_leg",
        ],
        action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_arm
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_arm
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_hand (binary-gripper-like)
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_hand
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # waist
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # left_leg
            ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),  # right_leg
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(g1_pillow_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
