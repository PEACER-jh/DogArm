# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(os.path.abspath(__file__))

GO2_ARX5_USD = os.path.join(current_dir, "go2_arx5.usd")

##
# Configuration
##


GO2_ARX5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_ARX5_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled=True,
        #     contact_offset=0.02,
        #     rest_offset=0.005 ,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            # leg
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
            # arm
            "joint.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=30.0,
            damping=0.6,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        # DO NOT MOVE IF POSSIBLE
        "joint1": DelayedPDActuatorCfg(
            joint_names_expr=["joint1"],
            # effort_limit=20.0,
            # effort_limit_sim=20.0,
            stiffness=50.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint2": DelayedPDActuatorCfg(
            joint_names_expr=["joint2"],
            # effort_limit=20.0,
            # effort_limit_sim=20.0,
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint3": DelayedPDActuatorCfg(
            joint_names_expr=["joint3"],
            # effort_limit=15.0,
            # effort_limit_sim=15.0,
            stiffness=80.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint4": DelayedPDActuatorCfg(
            joint_names_expr=["joint4"],
            # effort_limit=7.0,
            # effort_limit_sim=7.0,
            stiffness=30.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint5": DelayedPDActuatorCfg(
            joint_names_expr=["joint5"],
            # effort_limit=5.0,
            # effort_limit_sim=5.0,
            stiffness=30.0,
            damping=2.5,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint6": DelayedPDActuatorCfg(
            joint_names_expr=["joint6"],
            # effort_limit=5.0,
            # effort_limit_sim=5.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
    },
)
