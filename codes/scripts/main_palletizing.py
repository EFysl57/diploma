import asyncio
import importlib
import importlib.util
import os
import random
import shutil
import sys
import traceback

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
from pxr import Gf, Sdf, UsdGeom, UsdLux

# ==============================================================================
# 1. FORCE FRESH IMPORT OF LOCAL MODULES
# ==============================================================================
SCRIPTS_DIR = os.path.abspath(r"D:\diploma\codes\scripts")
if os.path.exists(SCRIPTS_DIR) and SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

pycache_dir = os.path.join(SCRIPTS_DIR, "__pycache__")
if os.path.exists(pycache_dir):
    shutil.rmtree(pycache_dir, ignore_errors=True)

for _mod_name in ("safety_module", "agv_manager", "camera_sensor"):
    sys.modules.pop(_mod_name, None)
importlib.invalidate_caches()

safety_module = importlib.import_module("safety_module")
agv_manager = importlib.import_module("agv_manager")
camera_sensor = importlib.import_module("camera_sensor")

SafetyDecisionModule = safety_module.SafetyDecisionModule
AGVManager = agv_manager.AGVManager
AGVFleetManager = getattr(agv_manager, "AGVFleetManager", None)
DualContourCameraSystem = getattr(camera_sensor, "DualContourCameraSystem", None)

if AGVFleetManager is None or DualContourCameraSystem is None:
    raise RuntimeError(
        f"Missing AGVFleetManager={AGVFleetManager is None}, "
        f"Missing DualContourCameraSystem={DualContourCameraSystem is None}"
    )


class Ur10Assets:
    def __init__(self):
        self.assets_root_path = get_assets_root_path()
        self.ur10_table_usd = (
            self.assets_root_path + "/Isaac/Samples/Leonardo/Stage/ur10_bin_stacking_short_suction.usd"
        )
        self.small_klt_usd = self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        self.background_usd = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        self.iw_hub_usd = self.assets_root_path + "/Isaac/Robots/Idealworks/iwhub/iwhub.usd"


def random_bin_spawn_transform():
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
    """Create persistent warehouse DomeLight + DistantLight so viewport never goes dark on reset."""
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
    """Create 3 floor safety zones (Red, Yellow, Green) around the UR10 palletizing cell."""
    ox, oy = safety_inst.robot_origin
    floor_z = -1.175

    zones_spec = [
        ("RedZone_UR10", safety_inst.red_radius, floor_z + 0.004, (0.95, 0.12, 0.12), 0.35),
        ("YellowZone_Buffer", safety_inst.yellow_radius, floor_z + 0.002, (0.98, 0.78, 0.08), 0.25),
        ("GreenZone_Logistics", 3.60, floor_z + 0.001, (0.12, 0.85, 0.28), 0.18),
    ]

    zones_root = "/World/SafetyZones"
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

    UsdGeom.Xform.Define(stage, "/World/Personnel")


# ==============================================================================
# 2. BIN STACKING TASK (POOL OF 4 REUSABLE BINS FOR RELIABLE MULTI-CYCLE CORTEX)
# ==============================================================================
class BinStackingTask(BaseTask):
    def __init__(self, env_path: str, assets: Ur10Assets, target_bins_per_pallet: int = 4) -> None:
        super().__init__("bin_stacking")
        self.assets = assets
        self.env_path = env_path
        self.bins = []
        self._bin_pool: list[CortexRigidPrim] = []
        self.on_conveyor = None
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
        rigid_bin.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        rigid_bin.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        rigid_bin.set_world_pose(
            position=np.array([-8.0 - idx * 0.6, 8.0, -3.5], dtype=float),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

    def reset_pallet_cycle(self) -> None:
        """Stash all 4 pooled bins off-stage and reset counters without destroying CortexRigidPrims."""
        for i, rigid_bin in enumerate(self._bin_pool):
            self._stash_bin(i, rigid_bin)
        self.bins.clear()
        self.on_conveyor = None
        self.pallet_full = False
        self._placing_wait_steps = 0
        print("[LOGISTICS] Pallet cleared and 4-bin pool stashed for next cycle.")

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
        self.reset_pallet_cycle()

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
                    self._placing_wait_steps += 1
                    if self._placing_wait_steps >= 210:
                        self.pallet_full = True
                        print(f"[COUNTER] {len(self.bins)} bins stacked! Pallet is full.")
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


# ==============================================================================
# 3. MAIN APPLICATION WITH MULTI-AGV FLEET, DUAL CAMERAS, AND 3-ZONE SAFETY
# ==============================================================================
class SmartPalletizing5(CortexBase):
    def __init__(self, monitor_fn=None):
        super().__init__()
        self._monitor_fn = monitor_fn
        self.robot = None
        self.task = None
        self.decider_network = None

        self.safety = SafetyDecisionModule()
        self.cameras = DualContourCameraSystem()

        self.agv_01 = AGVManager(
            agv_prim_path="/World/iw_hub_01",
            robot_id="AGV-01",
            authorized=True,
            standby_pos=(-3.20, 2.20, -1.175),
            dropoff_pos=(-3.20, -1.80, -1.175),
            priority=1,
        )
        self.agv_02 = AGVManager(
            agv_prim_path="/World/iw_hub_02",
            robot_id="AGV-02",
            authorized=True,
            standby_pos=(-3.20, -2.80, -1.175),
            dropoff_pos=(-3.20, -1.80, -1.175),
            priority=2,
        )
        self.fleet = AGVFleetManager([self.agv_01, self.agv_02])

        self._frame_counter = 0
        self._current_safety_state = "NORMAL"

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

        # Full Simple_Warehouse environment
        if not stage.GetPrimAtPath("/World/Background").IsValid():
            add_reference_to_stage(usd_path=ur10_assets.background_usd, prim_path="/World/Background")
            SingleXFormPrim(
                "/World/Background",
                position=[10.00, 2.00, -1.18180],
                orientation=[0.7071, 0.0, 0.0, 0.7071],
            )

        # Inspect idealworks/iwhub folder contents in local Isaac Sim assets
        iwhub_dir = ur10_assets.assets_root_path + "/Isaac/Robots/Idealworks/iwhub"
        res_sub, sub_entries = omni.client.list(iwhub_dir)
        resolved_iwhub_usd = ur10_assets.iw_hub_usd
        if res_sub == omni.client.Result.OK:
            names = [e.relative_path for e in sub_entries]
            print(f"[ASSET] iwhub folder contents: {names}")
            for fname in names:
                if fname.endswith(".usd") and "base" not in fname.lower():
                    resolved_iwhub_usd = f"{iwhub_dir}/{fname}"
                    break

        # Spawn AGV-01 and AGV-02 idealworks iw.hub rovers and force-load USD payloads
        for prim_path, pos in [
            ("/World/iw_hub_01", np.array([-3.20, 2.20, -1.175])),
            ("/World/iw_hub_02", np.array([-3.20, -2.80, -1.175])),
        ]:
            if not stage.GetPrimAtPath(prim_path).IsValid():
                add_reference_to_stage(usd_path=resolved_iwhub_usd, prim_path=prim_path)
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                prim.Load()
            SingleXFormPrim(prim_path, position=pos, orientation=np.array([1.0, 0.0, 0.0, 0.0]))

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

    def _reset_bin_stacking_context(self) -> None:
        world = self.get_world()
        if world is None or self.robot is None:
            return

        world._behaviors.clear()
        world._logical_state_monitors.clear()
        if hasattr(self.robot, "_decider_networks"):
            self.robot._decider_networks.clear()

        if hasattr(self.robot, "post_reset"):
            self.robot.post_reset()

        self.decider_network = behavior.make_decider_network(self.robot, self._on_monitor_update)
        world.add_decider_network(self.decider_network)
        world.reset_cortex()

    def _clear_pallet_only(self) -> None:
        self.agv_01.restore_pallet_nominal_pose()
        if self.task:
            self.task.reset_pallet_cycle()
        self._reset_bin_stacking_context()
        print("[LOGISTICS] Pallet seated at nominal pose and UR10 behavior rebuilt (0/4 slots).")

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
        self.fleet.update(step_size, spawner_reset_callback=self._clear_pallet_only)

        # 2. Dual-contour safety & CV-vs-Network verification every 20 frames
        if self._frame_counter % 20 == 0:
            detections = self.cameras.read_visual_detections_from_stage()
            telemetry = self.fleet.get_all_telemetry()

            new_state, log_msg = self.safety.evaluate_detected_objects(detections, agv_telemetry=telemetry)
            self.fleet.apply_security_halts(self.safety.blocked_agv_ids)

            if new_state != self._current_safety_state:
                self._current_safety_state = new_state
                print(f"[SAFETY MODULE] Status: {self._current_safety_state} -> {log_msg}")

        # 3. Dispatch next available authorized AGV (AGV-01 -> AGV-02 -> ...) when pallet is full
        if self.task and self.task.pallet_full and not self.fleet.has_active_service():
            dispatched = self.fleet.dispatch_next_available()
            if dispatched:
                print(f"[LOGISTICS] Pallet full! Dispatched {dispatched.robot_id} for swap.")

        # 4. Gate UR10 Cortex execution by safety decision and AGV service status
        if self._current_safety_state == "STOP" or self.fleet.has_active_service():
            return

        if self._current_safety_state == "REDUCED_SPEED":
            if self._frame_counter % 3 != 0:
                return

        try:
            world.step(False, False)
        except AttributeError as exc:
            if "grasp_T" in str(exc):
                self._reset_bin_stacking_context()
            else:
                raise

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


# ==============================================================================
# 4. STANDALONE LAUNCHER FOR ISAAC SIM SCRIPT EDITOR
# ==============================================================================
async def run_standalone() -> None:
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

        app.setup_scene()
        await world.reset_async()

        await app.setup_post_load()
        await app.on_event_async()

        print("[SYSTEM] SmartPalletizing5 (Multi-AGV A* + 3-Zone Safety + Dual Cameras) started.")
    except Exception:
        print("[ERROR] Failed to launch SmartPalletizing5:")
        traceback.print_exc()


asyncio.ensure_future(run_standalone())