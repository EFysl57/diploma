import asyncio
import importlib
import os
import random
import sys
import traceback
from typing import Optional, Tuple

sys.dont_write_bytecode = True

import isaacsim.cortex.framework.math_util as math_util
import numpy as np
import omni
import omni.client
import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.api.objects.capsule import VisualCapsule
from isaacsim.core.api.objects.sphere import VisualSphere
from isaacsim.core.api.tasks.base_task import BaseTask
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.cortex.behaviors.ur10 import bin_stacking_behavior as behavior
from isaacsim.cortex.framework.cortex_rigid_prim import CortexRigidPrim
from isaacsim.cortex.framework.cortex_utils import get_assets_root_path
from isaacsim.cortex.framework.robot import CortexUr10
from isaacsim.examples.interactive.cortex.cortex_base import CortexBase
from pxr import Gf, UsdGeom, UsdLux

_candidate_dirs = []
if "__file__" in globals():
    _candidate_dirs.append(os.path.dirname(os.path.abspath(__file__)))
_candidate_dirs.extend([
    r"D:\diploma\scripts",
    r"C:\diploma\scripts",
    r"D:\diploma\codes\scripts",
    os.getcwd(),
])

SCRIPTS_DIR = next(
    (p for p in _candidate_dirs if os.path.exists(os.path.join(p, "agv_manager.py"))),
    r"D:\diploma\scripts",
)

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import agv_manager
import safety_module

importlib.reload(safety_module)
importlib.reload(agv_manager)

from agv_manager import AGVFleetManager, AGVManager
from safety_module import SafetyDecisionModule


class Ur10Assets:
    def __init__(self):
        self.assets_root_path = get_assets_root_path()
        self.ur10_table_usd = (
            self.assets_root_path + "/Isaac/Samples/Leonardo/Stage/ur10_bin_stacking_short_suction.usd"
        )
        self.small_klt_usd = self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        self.background_usd = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        self.iw_hub_usd = self.assets_root_path + "/Isaac/Robots/Idealworks/iwhub/iw_hub_static.usd"


def random_bin_spawn_transform() -> Tuple[np.ndarray, np.ndarray]:
    x = random.uniform(-0.10, 0.10)
    y = 1.50
    z = -0.15
    position = np.array([x, y, z], dtype=float)

    z_rot = random.random() * 0.02 - 0.01
    w_rot = random.random() * 0.02 - 0.01
    norm = np.sqrt(z_rot**2 + w_rot**2)
    quat = math_util.Quaternion([w_rot / norm, 0.0, 0.0, z_rot / norm])
    if random.random() > 0.5:
        quat = quat * math_util.Quaternion([0.0, 0.0, 1.0, 0.0])

    return position, quat.vals


def setup_scene_lighting(stage) -> None:
    lights_root = "/World/Lights"
    if not stage.GetPrimAtPath(lights_root).IsValid():
        UsdGeom.Xform.Define(stage, lights_root)
    dome = UsdLux.DomeLight.Define(stage, f"{lights_root}/WarehouseDomeLight")
    dome.CreateIntensityAttr(1000.0)
    sun = UsdLux.DistantLight.Define(stage, f"{lights_root}/WarehouseSunLight")
    sun.CreateIntensityAttr(2200.0)
    xf = UsdGeom.Xformable(sun.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(35.0, 25.0, 0.0))


def build_safety_zones_and_personnel(stage, safety_inst: SafetyDecisionModule) -> None:
    ox, oy = safety_inst.robot_origin
    floor_z = -1.175

    zones_spec = [
        ("RedZone_UR10", getattr(safety_inst, "red_radius", 1.35), floor_z + 0.005, (0.95, 0.12, 0.12), 0.35),
        ("YellowZone_Buffer", getattr(safety_inst, "yellow_radius", 2.55), floor_z + 0.003, (0.98, 0.78, 0.08), 0.25),
        ("GreenZone_Logistics", getattr(safety_inst, "green_radius", 3.80), floor_z + 0.001, (0.12, 0.85, 0.28), 0.16),
    ]

    zones_root = "/World/SafetyZones"
    if stage.GetPrimAtPath(zones_root).IsValid():
        stage.RemovePrim(zones_root)
    UsdGeom.Xform.Define(stage, zones_root)

    for zone_name, radius, z_pos, rgb, opacity in zones_spec:
        prim_path = f"{zones_root}/{zone_name}"
        cyl = UsdGeom.Cylinder.Define(stage, prim_path)
        cyl.CreateRadiusAttr(float(radius))
        cyl.CreateHeightAttr(0.002)
        cyl.CreateAxisAttr("Z")
        cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        cyl.CreateDisplayOpacityAttr([float(opacity)])

        xf = UsdGeom.Xformable(cyl.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(ox), float(oy), float(z_pos)))

    if not stage.GetPrimAtPath("/World/Personnel").IsValid():
        UsdGeom.Xform.Define(stage, "/World/Personnel")


class BinStackingTask(BaseTask):
    def __init__(self, env_path: str, assets: Ur10Assets, target_bins_per_pallet: int = 4) -> None:
        super().__init__("bin_stacking")
        self.assets = assets
        self.env_path = env_path
        self.bins: list[CortexRigidPrim] = []
        self._bin_pool: list[CortexRigidPrim] = []
        self.on_conveyor: Optional[CortexRigidPrim] = None
        self.target_bins_per_pallet = target_bins_per_pallet
        self.pallet_full = False
        self._placing_wait_steps = 0

    def _spawn_bin(self, rigid_bin: CortexRigidPrim) -> None:
        x, q = random_bin_spawn_transform()
        rigid_bin.set_world_pose(position=x, orientation=q)
        rigid_bin.set_linear_velocity(np.array([0.0, -0.30, 0.0]))
        rigid_bin.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        rigid_bin.set_visibility(True)

    def _stash_bin(self, idx: int, rigid_bin: CortexRigidPrim) -> None:
        rigid_bin.set_visibility(False)
        rigid_bin.set_linear_velocity(np.zeros(3))
        rigid_bin.set_angular_velocity(np.zeros(3))
        rigid_bin.set_world_pose(
            position=np.array([-10.0 - idx * 0.6, 10.0, -4.0], dtype=float),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

    def hide_current_bins(self) -> None:
        for idx, rigid_bin in enumerate(self._bin_pool):
            self._stash_bin(idx, rigid_bin)

    def reset_pallet_cycle(self) -> None:
        self.hide_current_bins()
        self.bins.clear()
        self.on_conveyor = None
        self.pallet_full = False
        self._placing_wait_steps = 0
        print("[LOGISTICS] Pallet cleared: UR10 resumed for next 4 bins.")

    def clear_all_bins(self) -> None:
        stage = omni.usd.get_context().get_stage()
        for rigid_bin in list(self._bin_pool):
            try:
                if self.scene and self.scene.object_exists(rigid_bin.name):
                    self.scene.remove_object(rigid_bin.name)
            except Exception:
                pass
        try:
            bins_root = f"{self.env_path}/bins"
            if stage and stage.GetPrimAtPath(bins_root).IsValid():
                stage.RemovePrim(bins_root)
        except Exception:
            pass
        self._bin_pool.clear()
        self.bins.clear()
        self.on_conveyor = None
        self.pallet_full = False
        self._placing_wait_steps = 0

    def post_reset(self) -> None:
        pass

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        if self.pallet_full:
            return

        spawn_new = False
        if self.on_conveyor is None:
            spawn_new = True
        else:
            (x, y, z), _ = self.on_conveyor.get_world_pose()
            is_on_conveyor = y > 0.0 and -0.4 < x < 0.4
            if not is_on_conveyor:
                if len(self.bins) >= self.target_bins_per_pallet:
                    lin_vel = self.on_conveyor.get_linear_velocity()
                    speed = float(np.linalg.norm(lin_vel)) if lin_vel is not None else 0.0
                    is_placed_on_pallet = (x > 0.35) and (z < -0.55) and (speed < 0.04)

                    if is_placed_on_pallet:
                        self._placing_wait_steps += 1
                    else:
                        self._placing_wait_steps = 0

                    if self._placing_wait_steps >= 110:
                        self.pallet_full = True
                        print(f"[COUNTER] All {len(self.bins)} bins placed and UR10 paused in Idle! Pallet is full.")
                    return
                spawn_new = True

        if spawn_new and len(self.bins) < self.target_bins_per_pallet:
            slot_idx = len(self.bins)
            if slot_idx < len(self._bin_pool):
                rigid_bin = self._bin_pool[slot_idx]
            else:
                name = f"bin_{slot_idx}"
                prim_path = f"{self.env_path}/bins/{name}"
                add_reference_to_stage(usd_path=self.assets.small_klt_usd, prim_path=prim_path)
                rigid_bin = self.scene.add(CortexRigidPrim(name=name, prim_path=prim_path))
                self._bin_pool.append(rigid_bin)

            self.on_conveyor = rigid_bin
            self._spawn_bin(self.on_conveyor)
            self.bins.append(self.on_conveyor)
            self._placing_wait_steps = 0
            print(f"[SPAWNER] Spawned bin #{len(self.bins)} ({rigid_bin.name})")

    def world_cleanup(self) -> None:
        self.clear_all_bins()


class SmartPalletizing5(CortexBase):
    def __init__(self, monitor_fn=None):
        super().__init__()
        self._monitor_fn = monitor_fn
        self.robot = None
        self.task = None
        self.decider_network = None

        self.safety = SafetyDecisionModule()

        rovers_cfg = [
            ("/World/iw_hub_01", "AGV-01", "192.168.10.101", 1, True,  (-3.40,  2.40, -1.18), 1),
            ("/World/iw_hub_02", "AGV-02", "192.168.10.102", 2, True,  (-3.40,  0.80, -1.18), 2),
            ("/World/iw_hub_03", "AGV-03", "192.168.10.103", 3, True,  (-3.40, -0.80, -1.18), 3),
            ("/World/iw_hub_04", "AGV-04", "192.168.10.199", 4, False, (-3.40, -2.60, -1.18), 4),
        ]
        self.rovers = [
            AGVManager(
                agv_prim_path=prim_path,
                robot_id=rid,
                ip_address=ip_addr,
                marker_id=mid,
                authorized=auth,
                standby_pos=pos,
                dropoff_pos=(-3.40, -3.60, -1.18),
                priority=prio,
            )
            for prim_path, rid, ip_addr, mid, auth, pos, prio in rovers_cfg
        ]
        self.fleet = AGVFleetManager(self.rovers)

        self._frame_counter = 0
        self._agv_dispatched_for_current_pallet = False

    @staticmethod
    def purge_stale_bins_from_stage(env_path: str = "/World/Ur10Table") -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        bins_root = stage.GetPrimAtPath(f"{env_path}/bins")
        if bins_root.IsValid():
            stage.RemovePrim(f"{env_path}/bins")
        for prim in list(stage.Traverse()):
            path_str = str(prim.GetPath())
            if "/bins/bin_" in path_str or prim.GetName().startswith("bin_"):
                if stage.GetPrimAtPath(path_str).IsValid():
                    stage.RemovePrim(path_str)

    def setup_scene(self) -> None:
        world = self.get_world()
        env_path = "/World/Ur10Table"
        stage = omni.usd.get_context().get_stage()

        self.purge_stale_bins_from_stage(env_path)
        for legacy in ("/agv", "/World/iw_hub", "/World/GroundPlane"):
            if stage.GetPrimAtPath(legacy).IsValid():
                stage.RemovePrim(legacy)

        ur10_assets = Ur10Assets()
        if not stage.GetPrimAtPath(env_path).IsValid():
            add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=env_path)

        if not stage.GetPrimAtPath("/World/Background").IsValid():
            add_reference_to_stage(usd_path=ur10_assets.background_usd, prim_path="/World/Background")
            SingleXFormPrim(
                "/World/Background",
                position=[10.00, 2.00, -1.18180],
                orientation=[0.7071, 0.0, 0.0, 0.7071],
            )

        iwhub_dir = ur10_assets.assets_root_path + "/Isaac/Robots/Idealworks/iwhub"
        resolved_iwhub_usd = ur10_assets.iw_hub_usd
        res_sub, sub_entries = omni.client.list(iwhub_dir)
        if res_sub == omni.client.Result.OK:
            names = [e.relative_path for e in sub_entries]
            if "iw_hub_static.usd" in names:
                resolved_iwhub_usd = f"{iwhub_dir}/iw_hub_static.usd"

        for rover in self.rovers:
            if not stage.GetPrimAtPath(rover.prim_path).IsValid():
                add_reference_to_stage(usd_path=resolved_iwhub_usd, prim_path=rover.prim_path)
            prim = stage.GetPrimAtPath(rover.prim_path)
            if prim.IsValid():
                prim.Load()
            SingleXFormPrim(rover.prim_path, position=rover.wait_pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))

        build_safety_zones_and_personnel(stage, self.safety)

        if world.scene.object_exists("robot"):
            world.scene.remove_object("robot", registry_only=True)
        self.robot = world.add_robot(CortexUr10(name="robot", prim_path=f"{env_path}/ur10"))

        obstacles = [
            VisualSphere(
                "/World/Ur10Table/Obstacles/FlipStationSphere",
                name="flip_station_sphere",
                position=np.array([0.73, 0.76, -0.13]),
                radius=0.2,
                visible=False,
            ),
            VisualSphere(
                "/World/Ur10Table/Obstacles/NavigationDome",
                name="navigation_dome_obs",
                position=np.array([-0.031, -0.018, -1.086]),
                radius=1.1,
                visible=False,
            ),
            VisualCapsule(
                "/World/Ur10Table/Obstacles/NavigationFlipStation",
                name="navigation_flip_station_obs",
                position=np.array([0.766, 0.755, -0.5]),
                radius=0.5,
                height=0.5,
                visible=False,
            ),
        ]
        az = np.array([1.0, 0.0, -0.3])
        ax = np.array([0.0, 1.0, 0.0])
        ay = np.cross(az, ax)
        quat = math_util.matrix_to_quat(math_util.pack_R(ax, ay, az))
        obstacles.append(
            VisualCapsule(
                "/World/Ur10Table/Obstacles/NavigationBarrier",
                name="navigation_barrier_obs",
                position=np.array([0.471, 0.276, -0.563]),
                orientation=quat,
                radius=0.5,
                height=0.9,
                visible=False,
            )
        )
        for obs in obstacles:
            if not world.scene.object_exists(obs.name):
                world.scene.add(obs)
            self.robot.register_obstacle(obs)

        setup_scene_lighting(stage)

    def _unload_bins_at_dropoff(self) -> None:
        if self.task:
            self.task.hide_current_bins()

    def _rebuild_behavior_without_world_reset(self) -> None:
        world = self.get_world()
        if world is None or self.robot is None:
            return
        if hasattr(self.robot, "suction_gripper") and self.robot.suction_gripper is not None:
            try:
                self.robot.suction_gripper.open()
            except Exception:
                pass
        try:
            if hasattr(world, "_behaviors"):
                world._behaviors.clear()
            if hasattr(world, "_logical_state_monitors"):
                world._logical_state_monitors.clear()
        except Exception:
            pass

        self.decider_network = behavior.make_decider_network(self.robot, self._on_monitor_update)
        world.add_decider_network(self.decider_network)
        if hasattr(self.decider_network, "reset"):
            try:
                self.decider_network.reset()
            except Exception:
                pass

    def _resume_ur10_after_pallet_return(self) -> None:
        if self.rovers:
            self.rovers[0].restore_pallet_nominal_pose()
        if self.task:
            self.task.reset_pallet_cycle()
        self._agv_dispatched_for_current_pallet = False
        self._rebuild_behavior_without_world_reset()
        if self.task:
            self.task.pre_step(self._frame_counter, 0.0)
        print("[LOGISTICS] Cycle complete: empty pallet seated and UR10 unpaused for next 4 bins.")

    async def setup_post_load(self) -> None:
        world = self.get_world()
        env_path = "/World/Ur10Table"
        ur10_assets = Ur10Assets()

        if not self.robot:
            self.robot = world._robots["robot"]
            world._current_tasks.clear()
            world._behaviors.clear()
            world._logical_state_monitors.clear()

        self.task = BinStackingTask(env_path, ur10_assets, target_bins_per_pallet=4)
        self.task.set_up_scene(world.scene)
        world.add_task(self.task)

        self.decider_network = behavior.make_decider_network(self.robot, self._on_monitor_update)
        world.add_decider_network(self.decider_network)

        self.fleet.setup(world=world)

    def _on_monitor_update(self, diagnostics) -> None:
        if self._monitor_fn:
            stack = "\n".join(
                f"{'  ' * i}{el}" for i, el in enumerate(str(x) for x in self.decider_network._decider_state.stack)
            )
            self._monitor_fn(diagnostics, stack)

    def _on_physics_step(self, step_size: float) -> None:
        world = self.get_world()
        self._frame_counter += 1

        # 1. Step all AGVs in the fleet
        self.fleet.update(
            step_size,
            spawner_reset_callback=self._resume_ur10_after_pallet_return,
            unload_bins_callback=self._unload_bins_at_dropoff,
        )

        # 2. Dispatch strictly ONE AGV per full pallet
        if (
            self.task
            and self.task.pallet_full
            and not self._agv_dispatched_for_current_pallet
            and not self.fleet.has_active_service()
        ):
            dispatched = self.fleet.dispatch_next_available(stacked_bins=self.task.bins)
            if dispatched:
                self._agv_dispatched_for_current_pallet = True
                print(f"[LOGISTICS] UR10 paused in Idle. Dispatched {dispatched.robot_id} with {len(self.task.bins)} bins.")

        # 3. PAUSE UR10 Cortex while pallet is full or AGV is inside the cell
        if (self.task and self.task.pallet_full) or self.fleet.has_active_service():
            return

        world.step(False, False)

    async def on_event_async(self) -> None:
        world = self.get_world()
        await omni.kit.app.get_app().next_update_async()
        world.reset_cortex()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")
        world.add_physics_callback("sim_step", self._on_physics_step)
        await world.play_async()

    async def setup_pre_reset(self) -> None:
        world = self.get_world()
        if world and world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")

    def world_cleanup(self) -> None:
        if self.task:
            self.task.world_cleanup()


APP_INSTANCE: Optional[SmartPalletizing5] = None


async def run_standalone() -> None:
    global APP_INSTANCE
    try:
        from isaacsim.cortex.framework.cortex_world import CortexWorld

        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        if CortexWorld.instance():
            CortexWorld.instance().clear_instance()

        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()

        world = CortexWorld(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()

        app = SmartPalletizing5()
        app._world = world
        APP_INSTANCE = app

        app.setup_scene()
        await world.reset_async()

        await app.setup_post_load()
        await app.on_event_async()

        print("[SYSTEM] SmartPalletizing5 (Pause/Resume Cortex + 4-AGV Fleet) started.")
    except Exception:
        print("[ERROR] Failed to launch SmartPalletizing5:")
        traceback.print_exc()


asyncio.ensure_future(run_standalone())