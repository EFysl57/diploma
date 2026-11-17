import asyncio
import importlib
import math
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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

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
import camera_sensor
import safety_module

importlib.reload(safety_module)
importlib.reload(camera_sensor)
importlib.reload(agv_manager)

from agv_manager import AGVFleetManager, AGVManager
from camera_sensor import DualContourCameraSystem
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
        self.people_dir = self.assets_root_path + "/Isaac/People/Characters"
        self.idle_anim_usd = self.assets_root_path + "/Isaac/People/Animations/idle_loop.usd"


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


def _create_smooth_floor_disc(
    stage,
    prim_path: str,
    center_xy: Tuple[float, float],
    z_height: float,
    radius: float,
    rgb: Tuple[float, float, float],
    opacity: float,
    segments: int = 128,
) -> None:
    """Create a 128-segment completely matte translucent floor disc with zero Fresnel glare."""
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    points = [Gf.Vec3f(0.0, 0.0, 0.0)]
    for i in range(segments):
        theta = (2.0 * math.pi * i) / segments
        points.append(Gf.Vec3f(float(radius * math.cos(theta)), float(radius * math.sin(theta)), 0.0))

    face_vertex_counts = [3] * segments
    face_vertex_indices = []
    normals = [Gf.Vec3f(0.0, 0.0, 1.0)] * (segments + 1)
    for i in range(segments):
        v1 = i + 1
        v2 = 1 if (i + 1 == segments) else (i + 2)
        face_vertex_indices.extend([0, v1, v2])

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.render)

    xf = UsdGeom.Xformable(mesh.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(z_height)))

    mat_path = f"{prim_path}/ZoneMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("emissiveScale", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def _resolve_character_usd_paths(assets: Ur10Assets) -> list:
    """Resolve 4 distinct 3D character USD files from Isaac/People/Characters/."""
    fallback = [
        f"{assets.people_dir}/F_Business_02/F_Business_02.usd",
        f"{assets.people_dir}/M_Medical_01/M_Medical_01.usd",
        f"{assets.people_dir}/F_Medical_01/F_Medical_01.usd",
        f"{assets.people_dir}/male_adult_construction_03/male_adult_construction_03.usd",
    ]
    res, entries = omni.client.list(assets.people_dir)
    if res != omni.client.Result.OK:
        return fallback

    discovered = []
    for entry in entries:
        rel = entry.relative_path
        if rel.lower().endswith(".usd"):
            discovered.append(f"{assets.people_dir}/{rel}")
        else:
            sub_dir = f"{assets.people_dir}/{rel}"
            sub_res, sub_entries = omni.client.list(sub_dir)
            if sub_res == omni.client.Result.OK:
                for sub_e in sub_entries:
                    if sub_e.relative_path.lower().endswith(".usd") and "anim" not in sub_e.relative_path.lower():
                        discovered.append(f"{sub_dir}/{sub_e.relative_path}")
                        break

    if len(discovered) >= 4:
        discovered.sort(key=lambda p: (0 if ("construction" in p.lower() or "worker" in p.lower()) else 1, p))
        return discovered[:4]
    return (discovered + fallback)[:4]


def _set_custom_str_or_bool(prim, name: str, value) -> None:
    if isinstance(value, bool):
        attr = prim.GetAttribute(name)
        if not attr:
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool, custom=True)
        attr.Set(bool(value))
    else:
        attr = prim.GetAttribute(name)
        if not attr:
            attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.String, custom=True)
        attr.Set(str(value))


def _attach_idle_animation(stage, person_prim, idle_anim_usd: str) -> None:
    """Keep clean default bind pose without custom joint overrides."""
    pass


def _recolor_character_outfit(
    stage,
    person_prim,
    top_rgb: Tuple[float, float, float],
    pants_rgb: Tuple[float, float, float] = (0.18, 0.20, 0.24),
    helmet_rgb: Tuple[float, float, float] = (0.95, 0.95, 0.95),
    show_helmet: bool = True,
) -> None:
    """Tint native OmniPBR fabric materials, unify helmet color, and toggle helmet visibility."""
    person_path = str(person_prim.GetPath())
    helmet_keywords = ("helmet", "hardhat", "hat", "cap")

    # 1. Shared uniform helmet material for meshes that share materials with the shirt
    helmet_mat_path = f"{person_path}/Looks/UniformHelmetMat"
    helmet_mat = UsdShade.Material.Define(stage, helmet_mat_path)
    helmet_sh = UsdShade.Shader.Define(stage, f"{helmet_mat_path}/PreviewSurface")
    helmet_sh.CreateIdAttr("UsdPreviewSurface")
    helmet_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*helmet_rgb))
    helmet_sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    helmet_mat.CreateSurfaceOutput().ConnectToSource(helmet_sh.ConnectableAPI(), "surface")

    # 2. Toggle helmet visibility & bind uniform helmet color to helmet meshes
    for desc in Usd.PrimRange(person_prim):
        name_low = desc.GetName().lower()
        if any(hw in name_low for hw in helmet_keywords):
            if desc.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(desc).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.inherited if show_helmet else UsdGeom.Tokens.invisible
                )
            if show_helmet and desc.IsA(UsdGeom.Mesh):
                UsdShade.MaterialBindingAPI.Apply(desc).Bind(helmet_mat)

    # 3. Tint native cloth & helmet materials while preserving fabric textures and creases
    skip_words = (
        "skin", "face", "head", "hair", "eye", "lash", "brow",
        "shoe", "boot", "glove", "hand",
    )
    pants_words = ("pant", "trouser", "jean", "lower", "leg", "bottom", "skirt")

    for desc in Usd.PrimRange(person_prim):
        if not desc.IsA(UsdShade.Material):
            continue
        mat_name = desc.GetName().lower()
        if any(sw in mat_name for sw in skip_words):
            continue

        if any(hw in mat_name for hw in helmet_keywords):
            target_rgb = helmet_rgb
        elif any(pw in mat_name for pw in pants_words):
            target_rgb = pants_rgb
        else:
            target_rgb = top_rgb

        for child in desc.GetChildren():
            shader = UsdShade.Shader(child)
            if not shader:
                continue
            for inp_name in ("diffuse_tint", "albedo_tint", "diffuseColor"):
                inp = shader.GetInput(inp_name)
                if inp:
                    inp.Set(Gf.Vec3f(*target_rgb))


def build_safety_zones_and_personnel(stage, safety_inst: SafetyDecisionModule, assets: Ur10Assets) -> None:
    ox, oy = safety_inst.robot_origin
    floor_z = -1.180

    # 1. Completely matte translucent floor discs (Red & Yellow; outside = Green)
    zones_root = "/World/SafetyZones"
    if stage.GetPrimAtPath(zones_root).IsValid():
        stage.RemovePrim(zones_root)
    UsdGeom.Xform.Define(stage, zones_root)

    _create_smooth_floor_disc(
        stage=stage,
        prim_path=f"{zones_root}/RedZone_UR10",
        center_xy=(ox, oy),
        z_height=floor_z + 0.004,
        radius=float(safety_inst.red_radius),
        rgb=(0.82, 0.14, 0.14),
        opacity=0.22,
        segments=128,
    )
    _create_smooth_floor_disc(
        stage=stage,
        prim_path=f"{zones_root}/YellowZone_Buffer",
        center_xy=(ox, oy),
        z_height=floor_z + 0.002,
        radius=float(safety_inst.yellow_radius),
        rgb=(0.86, 0.68, 0.12),
        opacity=0.14,
        segments=128,
    )

    # 2. Four 3D personnel characters: Синий, Зеленый, Красный, Желтый + единая белая каска
    personnel_root = "/World/Personnel"
    if stage.GetPrimAtPath(personnel_root).IsValid():
        stage.RemovePrim(personnel_root)
    UsdGeom.Xform.Define(stage, personnel_root)

    char_usds = _resolve_character_usd_paths(assets)
    personnel_spec = [
        (
            "Person_RoboticsEngineer",
            "robotics_engineer",
            True,
            (4.70, 2.20, floor_z),
            char_usds[0],
            (0.15, 0.38, 0.88),  # Синий — Робототехник
            (0.18, 0.22, 0.35),
        ),
        (
            "Person_TechStaff",
            "tech_staff",
            True,
            (4.70, 0.80, floor_z),
            char_usds[1],
            (0.18, 0.68, 0.30),  # Зеленый — Тех. персонал
            (0.20, 0.26, 0.22),
        ),
        (
            "Person_WarehouseWorker",
            "warehouse_worker",
            False,
            (4.70, -0.60, floor_z),
            char_usds[2],
            (0.88, 0.20, 0.20),  # Красный — Работник склада (без каски)
            (0.24, 0.22, 0.22),
        ),
        (
            "Person_Electrician",
            "electrician",
            True,
            (4.70, -2.00, floor_z),
            char_usds[3],
            (0.95, 0.80, 0.15),  # Желтый — Электрик
            (0.22, 0.22, 0.25),
        ),
    ]

    for prim_name, role_name, has_helmet, pos_xyz, char_usd, top_rgb, pants_rgb in personnel_spec:
        person_path = f"{personnel_root}/{prim_name}"
        add_reference_to_stage(usd_path=char_usd, prim_path=person_path)
        person_prim = stage.GetPrimAtPath(person_path)
        if person_prim.IsValid():
            person_prim.Load()

            xf = UsdGeom.Xformable(person_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos_xyz))
            xf.AddRotateZOp().Set(180.0)

            _attach_idle_animation(stage, person_prim, assets.idle_anim_usd)
            _recolor_character_outfit(
                stage=stage,
                person_prim=person_prim,
                top_rgb=top_rgb,
                pants_rgb=pants_rgb,
                helmet_rgb=(0.95, 0.95, 0.95),
                show_helmet=has_helmet,
            )

            _set_custom_str_or_bool(person_prim, "safety:role", role_name)
            _set_custom_str_or_bool(person_prim, "safety:has_helmet", has_helmet)


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

        self.safety = SafetyDecisionModule(
            robot_origin=(0.35, 0.45),
            red_radius=2.55,
            yellow_radius=4.20,
        )
        self.camera_system = DualContourCameraSystem()

        # 4 AGVs parked on the left at X = -4.25 (outside Yellow zone radius 4.20 m from X = 0.35)
        rovers_cfg = [
            ("/World/iw_hub_01", "AGV-01", "192.168.10.101", 1, True,  (-4.25,  2.40, -1.18), 1),
            ("/World/iw_hub_02", "AGV-02", "192.168.10.102", 2, True,  (-4.25,  0.80, -1.18), 2),
            ("/World/iw_hub_03", "AGV-03", "192.168.10.103", 3, True,  (-4.25, -0.80, -1.18), 3),
            ("/World/iw_hub_04", "AGV-04", "192.168.10.199", 4, False, (-4.25, -2.60, -1.18), 4),
        ]
        self.rovers = [
            AGVManager(
                agv_prim_path=prim_path,
                robot_id=rid,
                ip_address=ip_addr,
                marker_id=mid,
                authorized=auth,
                standby_pos=pos,
                dropoff_pos=(-4.25, -3.60, -1.18),
                priority=prio,
            )
            for prim_path, rid, ip_addr, mid, auth, pos, prio in rovers_cfg
        ]
        self.fleet = AGVFleetManager(self.rovers)

        self._frame_counter = 0
        self._agv_dispatched_for_current_pallet = False
        self._timeline_sub = None

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

    def _on_timeline_event(self, event) -> None:
        """Reset blue dolly + pallet back to native pose whenever simulation stops (if moved)."""
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self.fleet.reset_all_carriers_to_nominal()

    def _register_timeline_reset_hook(self) -> None:
        timeline = omni.timeline.get_timeline_interface()
        stream = timeline.get_timeline_event_stream()
        self._timeline_sub = stream.create_subscription_to_pop(
            self._on_timeline_event,
            name="SmartPalletizing5_CarrierResetOnStop",
        )

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

        build_safety_zones_and_personnel(stage, self.safety, ur10_assets)
        self.camera_system.setup()

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
        self.fleet.reset_all_carriers_to_nominal()
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
        self._register_timeline_reset_hook()

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

        # 2. Dispatch strictly ONE authorized AGV per full pallet
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
        self.fleet.reset_all_carriers_to_nominal()

    def world_cleanup(self) -> None:
        self._timeline_sub = None
        self.fleet.reset_all_carriers_to_nominal()
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

        if APP_INSTANCE is not None:
            APP_INSTANCE.fleet.reset_all_carriers_to_nominal()
            APP_INSTANCE._timeline_sub = None

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

        print("[SYSTEM] SmartPalletizing5 started.")
    except Exception:
        print("[ERROR] Failed to launch SmartPalletizing5:")
        traceback.print_exc()


asyncio.ensure_future(run_standalone())