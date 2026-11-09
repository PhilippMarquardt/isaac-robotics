"""Reusable Isaac Lab InteractiveScene config for an imported game-engine scene.

Drop the USD into ../assets/usd/imported_scene.usd then use:

    from scripts.scene_cfg import ImportedGameSceneCfg
    scene = InteractiveScene(ImportedGameSceneCfg(num_envs=1, env_spacing=0.0))
"""
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils


USD_DEFAULT = "scene_pipeline/assets/usd/imported_scene.usd"


@configclass
class ImportedGameSceneCfg(InteractiveSceneCfg):
    """Loads a user-provided USD as the world geometry + a fallback dome light."""

    imported_scene: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ImportedScene",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_DEFAULT),
    )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.9)),
    )
