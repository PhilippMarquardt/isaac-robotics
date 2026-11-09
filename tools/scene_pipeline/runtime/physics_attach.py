"""Attach physics to a lamp prop at runtime, after it has been spawned into the
live stage.

USD reference composition wasn't passing our RigidBodyAPI overrides through to
PhysX for joint resolution, so we author physics directly on the live composed
prims instead. This is the same thing the converter tries to do at asset-build
time, just executed against the spawned scene rather than the source USD.
"""
from __future__ import annotations


def attach_lamp_switch_physics(
    lamp_root_path: str,             # e.g. "/World/Lamp"
    body_in_visual: str,             # e.g. "/industrial_pipe_lamp/industrial_pipe_lamp"
    switch_in_visual: str,           # e.g. "/industrial_pipe_lamp/industrial_pipe_lamp_switch"
    switch_world_center: tuple,
    joint_travel: float = 0.003,
    spring_stiffness: float = 2000.0,
    spring_damping: float = 8.0,
    switch_mass: float = 0.01,
) -> dict:
    """Apply rigid body + collision APIs and a prismatic joint that lets the
    switch depress along world Z. Returns paths the runtime needs.
    """
    import omni.usd
    from pxr import UsdPhysics, UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()

    body_path = f"{lamp_root_path}/Visual{body_in_visual}"
    switch_path = f"{lamp_root_path}/Visual{switch_in_visual}"
    joint_path = f"{lamp_root_path}/SwitchJoint"

    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"body prim not found: {body_path}")
    switch_prim = stage.GetPrimAtPath(switch_path)
    if not switch_prim.IsValid():
        raise RuntimeError(f"switch prim not found: {switch_path}")

    # Lamp body: kinematic rigid, convex-decomposition collider.
    rb = UsdPhysics.RigidBodyAPI.Apply(body_prim)
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(body_prim)
    mca_body = UsdPhysics.MeshCollisionAPI.Apply(body_prim)
    mca_body.CreateApproximationAttr().Set("convexDecomposition")

    # Switch: dynamic rigid, convex hull.
    UsdPhysics.RigidBodyAPI.Apply(switch_prim)
    UsdPhysics.MassAPI.Apply(switch_prim).CreateMassAttr(switch_mass)
    UsdPhysics.CollisionAPI.Apply(switch_prim)
    mca_sw = UsdPhysics.MeshCollisionAPI.Apply(switch_prim)
    mca_sw.CreateApproximationAttr().Set("convexHull")

    # Prismatic joint, Z axis, joint_travel of inward travel.
    joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-joint_travel)
    joint.CreateUpperLimitAttr(0.0)
    joint.CreateBody0Rel().SetTargets([body_path])
    joint.CreateBody1Rel().SetTargets([switch_path])
    # joint frame: world position of the switch, both bodies use same frame.
    joint.CreateLocalPos0Attr(Gf.Vec3f(*switch_world_center))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr().Set("force")
    drive.CreateMaxForceAttr(50.0)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateStiffnessAttr(spring_stiffness)
    drive.CreateDampingAttr(spring_damping)

    return {
        "body_path": body_path,
        "switch_path": switch_path,
        "joint_path": joint_path,
    }
