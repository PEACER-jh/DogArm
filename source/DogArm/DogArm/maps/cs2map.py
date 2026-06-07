# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObjectCfg

_current_dir = os.path.dirname(os.path.abspath(__file__))
DUST2_USD_PATH = os.path.join(_current_dir, "usd", "dust2", "dust2.usdc")
_WALKABLE_AREA_PATH = os.path.join(_current_dir, "usd", "dust2", "walkable_area.usdc")  # unused, kept for reference

DUST2_SPAWN_CFG = sim_utils.UsdFileCfg(
    usd_path=DUST2_USD_PATH,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,  # static map
        disable_gravity=True,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
    ),
)
"""Spawn config for the de_dust2 CS2 map (static, shared across envs)."""


def load_walkable_vertices(device: str = "cuda") -> torch.Tensor:
    """Read world-space vertices of all mesh prims under any parent named 'walkable*'.

    The unified USD file has both the map and walkable_area in the same coordinate
    space, so vertices are directly in Isaac Sim world coordinates.
    """
    import omni.usd
    from pxr import Gf, Usd, UsdGeom
    import numpy as np
    import torch

    stage = omni.usd.get_context().get_stage()
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points_list = []

    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        # Only process prims whose parent has 'walkable' in the name
        parent = prim.GetParent()
        if parent and "walkable" in parent.GetName().lower():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
                world_xf = xf_cache.GetLocalToWorldTransform(prim)
                for i in range(len(pts)):
                    p = world_xf.Transform(Gf.Vec3f(float(pts[i][0]), float(pts[i][1]), float(pts[i][2])))
                    pts[i] = (float(p[0]), float(p[1]), float(p[2]))
                points_list.append(pts)

    if not points_list:
        raise RuntimeError(
            "No 'walkable*' parent found in stage. "
            "Ensure the Blender Collection containing walkable area copies is named 'walkable_area' or similar."
        )
    all_verts = np.concatenate(points_list, axis=0)
    return torch.tensor(all_verts, dtype=torch.float32, device=device)
