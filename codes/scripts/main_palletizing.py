import asyncio
import importlib
import os
import random
import sys
import traceback

import isaacsim.cortex.framework.math_util as math_util
import numpy as np
import omni
import omni.usd
from isaacsim.core.api.objects import VisualCuboid
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

# ==============================================================================
# 1. IMPORT LOCAL MODULES (safety_module, agv_manager, camera_sensor)
# ==============================================================================
SCRIPTS_DIR = os.path.abspath(r"D:\дипломка\codes\scripts")
if os.path.exists(SCRIPTS_DIR) and SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

safety_module = importlib.import_module("safety_module")
agv_manager = importlib.import_module("agv_manager")
camera_sensor = importlib.import_module("camera_sensor")

importlib.reload(safety_module)
importlib.reload(agv_manager)
importlib.reload(camera_sensor)

SafetyDecisionModule = safety_module.SafetyDecisionModule
AGVManager = agv_manager.AGVManager
SafetyCamera = camera_sensor.SafetyCamera


class Ur10Assets:
    def __init__(self):
        self.assets_root_path = get_assets_root_path()
        self.ur10_table_usd = (
            self.assets_root_path + "/Isaac/Samples/Leonardo/Stage/ur10_bin_stacking_short_suction.usd"
        )
        self.small_klt_usd = self.assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT.usd"
        self.background_usd = self.assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"


def random_bin_spawn_transform():
    x = random.uniform(-0.15, 0.15)
    y = 1.5
    z = -0.15
    position = np.array([x, y, z])

    z_rot = random.random() * 0.02 - 0.01
    w_rot = random.random() * 0.02 - 0.01
    norm = np.sqrt(z_rot**2 + w_rot**2)
    quat = math_util.Quaternion([w_rot / norm, 0, 0, z_rot / norm])
    if random.random() > 0.5:
        quat = quat * math_util.Quaternion([0, 0, 1, 0])

    return position, quat.vals


# ==============================================================================
# 2. BIN STACKING TASK WITH PALLET COUNTING AND FULL BIN CLEANUP
# ==============================================================================
class BinStackingTask(BaseTask):
    def __init__(self, env_path, assets, target_bins_per_pallet=4) -> None:
        super().__init__("bin_stacking")
        self.assets = assets
        self.env_path = env_path
        self.bins = []
        self.stashed_bins = []
        self.on_conveyor = None
        self.target_bins_per_pallet = target_bins_per_pallet
        self.pallet_full = False
        self._placing_wait_steps = 0
# Внутри BinStackingTask.__init__ добавь общий счетчик за всю сессию:
        self._total_spawned_count = 0


    def _spawn_bin(self, rigid_bin):
        x, q = random_bin_spawn_transform()
        rigid_bin.set_world_pose(position=x, orientation=q)
        rigid_bin.set_linear_velocity(np.array([0, -0.30, 0]))
        rigid_bin.set_visibility(True)

    def clear_all_bins(self) -> None:
        """Remove all spawned bins from both Isaac Sim scene registry and USD stage."""
        stage = omni.usd.get_context().get_stage()

        for rigid_bin in self.bins:
            try:
                if self.scene and self.scene.object_exists(rigid_bin.name):
                    self.scene.remove_object(rigid_bin.name)
            except Exception:
                pass

            try:
                prim_path = f"{self.env_path}/bins/{rigid_bin.name}"
                if stage and stage.GetPrimAtPath(prim_path).IsValid():
                    stage.RemovePrim(prim_path)
            except Exception:
                pass

        # Clean up parent /bins container prim to guarantee zero leftover meshes
        try:
            bins_root = f"{self.env_path}/bins"
            if stage and stage.GetPrimAtPath(bins_root).IsValid():
                stage.RemovePrim(bins_root)
        except Exception:
            pass

        self.bins.clear()
        self.stashed_bins.clear()
        self.on_conveyor = None
        self.pallet_full = False
        self._placing_wait_steps = 0
        print("[LOGISTICS] All bins removed from stage. Pallet is empty and ready.")

    def post_reset(self) -> None:
        """Called during pallet swap or simulation reset."""
        self.clear_all_bins()

    def pre_step(self, time_step_index, simulation_time) -> None:
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
                    # Wait ~3.5 seconds (210 steps at 60 Hz) for the 4th bin to be placed on the pallet
                    self._placing_wait_steps += 1
                    if self._placing_wait_steps >= 210:
                        self.pallet_full = True
                        print(f"[COUNTER] {len(self.bins)} bins stacked! Pallet is full.")
                    return
                spawn_new = True

        if spawn_new:
            name = f"bin_{self._total_spawned_count}"
            self._total_spawned_count += 1
            prim_path = f"{self.env_path}/bins/{name}"
            add_reference_to_stage(usd_path=self.assets.small_klt_usd, prim_path=prim_path)
            self.on_conveyor = self.scene.add(CortexRigidPrim(name=name, prim_path=prim_path))

            self._spawn_bin(self.on_conveyor)
            self.bins.append(self.on_conveyor)
            self._placing_wait_steps = 0
            print(f"[SPAWNER] Spawned bin #{len(self.bins)}")

    def world_cleanup(self):
        self.clear_all_bins()


# ==============================================================================
# 3. MAIN CLASS WITH SAFETY, CAMERA, AND AGV INTEGRATION
# ==============================================================================
class SmartPalletizing5(CortexBase):
    def __init__(self, monitor_fn=None):
        super().__init__()
        self._monitor_fn = monitor_fn
        self.robot = None
        self.task = None

        # Custom modules
        self.safety = SafetyDecisionModule()
        self.agv = AGVManager(agv_prim_path="/agv")
        self.camera = SafetyCamera()

        self._frame_counter = 0
        self._current_safety_state = "NORMAL"

    def setup_scene(self):
        world = self.get_world()
        env_path = "/World/Ur10Table"
        stage = omni.usd.get_context().get_stage()

        # 1. Удаляем только старые коробки из текущей открытой сцены
        bins_root = stage.GetPrimAtPath(f"{env_path}/bins")
        if bins_root.IsValid():
            stage.RemovePrim(f"{env_path}/bins")

        # Страховка: если при перезапуске остались примы bin_* в сцене
        for prim in list(stage.Traverse()):
            path_str = str(prim.GetPath())
            if "/bins/bin_" in path_str:
                stage.RemovePrim(path_str)

        # 2. Если стол UR10 и фон уже есть в открытом .usd файле — не дублируем их
        ur10_assets = Ur10Assets()
        if not stage.GetPrimAtPath(env_path).IsValid():
            add_reference_to_stage(usd_path=ur10_assets.ur10_table_usd, prim_path=env_path)
        if not stage.GetPrimAtPath("/World/Background").IsValid():
            add_reference_to_stage(usd_path=ur10_assets.background_usd, prim_path="/World/Background")
            SingleXFormPrim(
                "/World/Background", position=[10.00, 2.00, -1.18180], orientation=[0.7071, 0, 0, 0.7071]
            )

        self.robot = world.add_robot(CortexUr10(name="robot", prim_path=f"{env_path}/ur10"))

        # Planning obstacles for Cortex motion planner
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
                position=[-0.031, -0.018, -1.086],
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
                position=[0.471, 0.276, -0.563],
                orientation=quat,
                radius=0.5,
                height=0.9,
                visible=False,
            )
        )
        for obs in obstacles:
            world.scene.add(obs)
            self.robot.register_obstacle(obs)

        # Ensure /agv prim exists on stage for AGVManager
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath("/agv").IsValid():
            world.scene.add(
                VisualCuboid(
                    prim_path="/agv",
                    name="agv_rover",
                    position=np.array([-2.5, -0.5, -0.45]),
                    scale=np.array([0.8, 0.5, 0.3]),
                    color=np.array([1.0, 0.6, 0.0]),
                )
            )
    def _clear_pallet_only(self):
        """Clear physical bins and cleanly reset BinStackingContext without touching the robot arm."""
        if hasattr(self, "decider_network") and self.decider_network is not None:
            ctx = getattr(self.decider_network, "context", None)
            if ctx is not None:
                # 1. Пробуем вызвать штатный метод reset() самого контекста bin_stacking_behavior
                if hasattr(ctx, "reset") and callable(ctx.reset):
                    ctx.reset()
                else:
                    # 2. Если reset() нет, сбрасываем все поля состояния активной коробки и слотов
                    for attr in ("active_bin", "in_flip_station", "grasp_bin"):
                        if hasattr(ctx, attr):
                            setattr(ctx, attr, None)
                    for bool_attr in ("has_active_bin", "is_bin_grasped", "is_pallet_full"):
                        if hasattr(ctx, bool_attr):
                            setattr(ctx, bool_attr, False)
                    for list_attr in ("bins", "stacked_bins", "registered_bins"):
                        if hasattr(ctx, list_attr):
                            getattr(ctx, list_attr).clear()

        # 3. Удаляем физические коробки со сцены и разрешаем спавн новых
        if self.task:
            self.task.post_reset()

        print("[LOGISTICS] Pallet and behavior context cleared. Ready for new bins.")

        # 2. Удаляем старые коробки со сцены и разрешаем спавн новых
        if self.task:
            self.task.post_reset()

        print("[LOGISTICS] Pallet cleared. Robot arm remains active and ready for new bins.")
    async def setup_post_load(self):
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

        # Initialize overhead safety camera
        self.camera.setup()

    def _on_monitor_update(self, diagnostics):
        if self._monitor_fn:
            stack = "\n".join(
                f"{'  ' * i}{el}" for i, el in enumerate(str(x) for x in self.decider_network._decider_state.stack)
            )
            self._monitor_fn(diagnostics, stack)

    def _on_physics_step(self, step_size):
        world = self.get_world()
        self._frame_counter += 1

        # 1. Update AGV logistics
        self.agv.update(step_size, spawner_reset_callback=self._clear_pallet_only)

        # 2. Evaluate safety zones every 30 frames
        if self._frame_counter % 30 == 0:
            detected = []
            if self.agv.state != "IDLE":
                detected.append({"class": "agv", "pos": (self.agv.current_pos[0], self.agv.current_pos[1])})

            new_state, log_msg = self.safety.evaluate_detected_objects(detected)
            if new_state != self._current_safety_state:
                self._current_safety_state = new_state
                print(f"[SAFETY MODULE] Status: {self._current_safety_state} -> {log_msg}")

        # 3. Trigger AGV call when pallet is full
        if self.task and self.task.pallet_full and self.agv.state == "IDLE":
            print("[LOGISTICS] Pallet is full! Calling AGV for swap...")
            self.agv.trigger_pallet_swap()

        # 4. Step Cortex / UR10 execution based on safety status and AGV state
        if self._current_safety_state == "STOP" or self.agv.state != "IDLE":
            # Freeze robot arm during safety stop or active AGV service
            return

        if self._current_safety_state == "REDUCED_SPEED":
            # Skip 2 out of 3 Cortex ticks for smooth 3x speed reduction
            if self._frame_counter % 3 != 0:
                return

        world.step(False, False)

    async def on_event_async(self):
        world = self.get_world()
        await omni.kit.app.get_app().next_update_async()
        world.reset_cortex()
        world.add_physics_callback("sim_step", self._on_physics_step)
        await world.play_async()

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")

    def world_cleanup(self):
        if self.task:
            self.task.world_cleanup()


# ==============================================================================
# 4. STANDALONE LAUNCHER FOR ISAAC SIM SCRIPT EDITOR
# ==============================================================================
# ==============================================================================
# 4. STANDALONE LAUNCHER FOR ISAAC SIM SCRIPT EDITOR
# ==============================================================================
async def run_standalone():
    try:
        from isaacsim.cortex.framework.cortex_world import CortexWorld

        # Останавливаем таймлайн перед зачисткой старых коробок
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        await omni.kit.app.get_app().next_update_async()

        if CortexWorld.instance():
            CortexWorld.instance().clear_instance()

        world = CortexWorld(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()

        app = SmartPalletizing5()
        app._world = world

        app.setup_scene()
        await world.reset_async()

        await app.setup_post_load()
        await app.on_event_async()

        print("[SYSTEM] Started on existing stage with old bins cleared!")
    except Exception:
        print("[ERROR] Failed to launch SmartPalletizing5:")
        traceback.print_exc()


asyncio.ensure_future(run_standalone())

