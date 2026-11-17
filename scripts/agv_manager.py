import heapq
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni.physx
import omni.usd
from isaacsim.core.prims import SingleXFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


ARUCO_4X4_PATTERNS = {
    1: [
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ],
    2: [
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 1],
    ],
    3: [
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 1],
    ],
    4: [
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ],
}


def _generate_number_decal_png(robot_id: str, marker_id: int, out_dir: str) -> Optional[str]:
    """Generate a high-contrast bright-red license plate PNG with centered bold white robot_id text."""
    if Image is None:
        return None
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{robot_id.lower().replace('-', '_')}_decal.png")

    w, h = 640, 180
    bg_color = (225, 22, 22, 255)
    img = Image.new("RGBA", (w, h), bg_color)
    draw = ImageDraw.Draw(img)

    draw.rectangle([6, 6, w - 7, h - 7], outline=(255, 255, 255, 255), width=7)

    try:
        font = ImageFont.truetype("arialbd.ttf", 112)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), robot_id, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (w - tw) / 2.0 - bbox[0]
    ty = (h - th) / 2.0 - bbox[1]
    draw.text((tx, ty), robot_id, fill=(255, 255, 255, 255), font=font)

    img.save(png_path)
    return png_path


class WarehousePathPlanner:
    """Builds a 2D occupancy grid from static USD scene obstacles and plans smoothed A* paths."""

    def __init__(
        self,
        bounds_min: Tuple[float, float] = (-5.5, -4.5),
        bounds_max: Tuple[float, float] = (3.8, 4.5),
        resolution: float = 0.20,
        robot_radius: float = 0.46,
    ):
        self.min_x, self.min_y = bounds_min
        self.max_x, self.max_y = bounds_max
        self.res = resolution
        self.robot_radius = robot_radius
        self.nx = int(math.ceil((self.max_x - self.min_x) / self.res))
        self.ny = int(math.ceil((self.max_y - self.min_y) / self.res))
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)
        self._built = False

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(np.clip(round((x - self.min_x) / self.res), 0, self.nx - 1))
        gy = int(np.clip(round((y - self.min_y) / self.res), 0, self.ny - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return self.min_x + gx * self.res, self.min_y + gy * self.res

    def build_from_stage(
        self,
        stage,
        pallet_xy: Tuple[float, float],
        approach_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.grid.fill(False)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        obstacle_keywords = ("ur10table", "conveyor", "table", "wall", "rack", "shelf", "pillar", "barrier")
        ignore_keywords = (
            "floor", "ground", "ceiling", "bin_", "iw_hub",
            "personnel", "safetyzones", "pallet", "cart", "trolley", "dolly",
        )

        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path_lower = str(prim.GetPath()).lower()
            if any(ig in path_lower for ig in ignore_keywords):
                continue
            if not any(kw in path_lower for kw in obstacle_keywords):
                continue

            bound = bbox_cache.ComputeWorldBound(prim)
            rng = bound.ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            bmin = rng.GetMin()
            bmax = rng.GetMax()

            if bmax[2] < -1.12 or bmin[2] > 0.35:
                continue
            if (bmax[0] - bmin[0]) > 7.5 and (bmax[1] - bmin[1]) > 7.5:
                continue

            self._mark_rect(bmin[0], bmin[1], bmax[0], bmax[1], blocked=True)

        # Keep UR10 table pedestal & conveyor blocked
        self._mark_rect(-0.55, -0.35, 0.22, 1.85, blocked=True)

        # Open longitudinal docking corridor along +X from open end into dolly
        px, py = pallet_xy
        if approach_xy is not None:
            ax, ay = approach_xy
            x0, x1 = min(ax, px) - 0.35, max(ax, px) + 0.35
            y0, y1 = min(ay, py) - 0.35, max(ay, py) + 0.35
            self._mark_rect(x0, y0, x1, y1, blocked=False)
        else:
            self._mark_rect(px - 0.35, py - 0.42, px + 1.85, py + 0.42, blocked=False)
        self._built = True

    def _mark_rect(self, x0: float, y0: float, x1: float, y1: float, blocked: bool) -> None:
        pad = self.robot_radius if blocked else 0.0
        gx0, gy0 = self.world_to_grid(x0 - pad, y0 - pad)
        gx1, gy1 = self.world_to_grid(x1 + pad, y1 + pad)
        self.grid[gx0 : gx1 + 1, gy0 : gy1 + 1] = blocked

    def _nearest_free_cell(self, gx: int, gy: int) -> Tuple[int, int]:
        if not self.grid[gx, gy]:
            return gx, gy
        for r in range(1, 15):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.nx and 0 <= ny < self.ny and not self.grid[nx, ny]:
                        return nx, ny
        return gx, gy

    def _has_line_of_sight(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        temp_grid: Optional[np.ndarray] = None,
    ) -> bool:
        occ = temp_grid if temp_grid is not None else self.grid
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        steps = max(2, int(math.ceil(dist / (self.res * 0.5))))
        for i in range(steps + 1):
            t = i / steps
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            gx, gy = self.world_to_grid(x, y)
            if occ[gx, gy]:
                return False
        return True

    def _smooth_waypoints(
        self,
        start_xy: Tuple[float, float],
        waypoints: List[np.ndarray],
        temp_grid: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        if len(waypoints) <= 1:
            return waypoints
        pts = [np.array([start_xy[0], start_xy[1], 0.0], dtype=float)] + waypoints
        smoothed: List[np.ndarray] = []
        anchor_idx = 0
        while anchor_idx < len(pts) - 1:
            furthest_idx = anchor_idx + 1
            for candidate_idx in range(len(pts) - 1, anchor_idx, -1):
                if self._has_line_of_sight(
                    (float(pts[anchor_idx][0]), float(pts[anchor_idx][1])),
                    (float(pts[candidate_idx][0]), float(pts[candidate_idx][1])),
                    temp_grid=temp_grid,
                ):
                    furthest_idx = candidate_idx
                    break
            smoothed.append(pts[furthest_idx])
            anchor_idx = furthest_idx
        return smoothed

    def plan_path(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        dynamic_obstacles: Optional[List[Tuple[float, float]]] = None,
    ) -> List[np.ndarray]:
        temp_grid = self.grid.copy()
        if dynamic_obstacles:
            for ox, oy in dynamic_obstacles:
                gx0, gy0 = self.world_to_grid(ox - 0.55, oy - 0.55)
                gx1, gy1 = self.world_to_grid(ox + 0.55, oy + 0.55)
                temp_grid[gx0 : gx1 + 1, gy0 : gy1 + 1] = True

        sx, sy = self._nearest_free_cell(*self.world_to_grid(*start_xy))
        gx, gy = self._nearest_free_cell(*self.world_to_grid(*goal_xy))
        temp_grid[sx, sy] = False
        temp_grid[gx, gy] = False

        open_heap = [(0.0, (sx, sy))]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]

        found = False
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == (gx, gy):
                found = True
                break

            cx, cy = current
            for dx, dy, step_cost in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.nx and 0 <= ny < self.ny):
                    continue
                if temp_grid[nx, ny]:
                    continue
                if dx != 0 and dy != 0 and (temp_grid[cx + dx, cy] or temp_grid[cx, cy + dy]):
                    continue

                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get((nx, ny), float("inf")):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    h = math.hypot(gx - nx, gy - ny)
                    heapq.heappush(open_heap, (tentative_g + h, (nx, ny)))

        if not found:
            return [np.array([goal_xy[0], goal_xy[1], 0.0], dtype=float)]

        cells = [(gx, gy)]
        cur = (gx, gy)
        while cur in came_from:
            cur = came_from[cur]
            cells.append(cur)
        cells.reverse()

        waypoints: List[np.ndarray] = []
        last_dir = None
        for i in range(1, len(cells)):
            d = (cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1])
            if d != last_dir:
                wx, wy = self.grid_to_world(*cells[i - 1])
                waypoints.append(np.array([wx, wy, 0.0], dtype=float))
                last_dir = d

        waypoints.append(np.array([goal_xy[0], goal_xy[1], 0.0], dtype=float))
        raw_wps = waypoints[1:] if len(waypoints) > 1 else waypoints
        return self._smooth_waypoints(start_xy, raw_wps, temp_grid=temp_grid)


class AGVManager:
    """Autonomous idealworks iw.hub controller:
    - Slides longitudinally under the blue dolly from its open +X end (caster wheels on rover sides)
    - Lifts & carries the entire blue dolly + pallet + stacked KLT bins together
    - Restores the dolly + pallet back to native pose after cycle or timeline Stop
    """

    CELL_BUSY_STATES = {
        "APPROACHING_PALLET",
        "ALIGNING_DOCK",
        "DOCKING",
        "LIFTING",
        "BACKING_OUT_WITH_LOAD",
        "RETURNING_WITH_EMPTY",
        "ALIGNING_DOCK_EMPTY",
        "DOCKING_EMPTY",
        "LOWERING",
        "UNDOCKING",
    }

    @classmethod
    def capture_or_restore_nominal_carrier(cls, stage) -> None:
        return None

    def __init__(
        self,
        agv_prim_path: str = "/World/iw_hub_01",
        pallet_prim_path: str = "/World/Ur10Table/pallet",
        cart_prim_path: Optional[str] = None,
        robot_id: str = "AGV-01",
        ip_address: str = "192.168.10.101",
        marker_id: int = 1,
        authorized: bool = True,
        standby_pos: Tuple[float, float, float] = (-4.25, 2.40, -1.18),
        dropoff_pos: Tuple[float, float, float] = (-4.25, -3.60, -1.18),
        priority: int = 1,
        wheel_radius: float = 0.085,
        wheel_base: float = 0.58,
    ):
        self.prim_path = agv_prim_path
        self.articulation_path = agv_prim_path
        self.pallet_prim_path = pallet_prim_path
        self.cart_prim_path = cart_prim_path
        self.robot_id = robot_id
        self.ip_address = ip_address
        self.marker_id = marker_id
        self.authorized = authorized
        self.priority = priority
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base

        self.floor_z = -1.180
        self.root_z_offset = 0.0
        self.wait_pos = np.array(standby_pos, dtype=float)
        self.pallet_pos = np.array([0.688, 0.575, self.floor_z], dtype=float)
        self.dropoff_pos = np.array(dropoff_pos, dtype=float)

        # Open longitudinal end of the blue dolly is on +X (away from UR10 pedestal at -X)
        # Rover drives from +X towards -X (yaw = 180 deg) so dolly caster wheels sit on rover left/right sides!
        self.dock_approach_offset = np.array([1.45, 0.0], dtype=float)
        self.approach_pos = np.array([0.688 + 1.45, 0.575, self.floor_z], dtype=float)

        self.current_pos = self.wait_pos.copy()
        self.current_yaw = 0.0
        self.carrier_base_yaw = math.pi

        self.state = "IDLE"
        self.task_flag = "STANDBY"
        self.cargo_flag = "EMPTY"
        self.max_linear_speed = 1.15
        self.dock_linear_speed = 0.30
        self.max_angular_speed = 1.65

        self.lift_height = 0.0
        self.max_lift_height = 0.085
        self.lift_speed = 0.08

        self.planner: Optional[WarehousePathPlanner] = None
        self._active_waypoints: List[np.ndarray] = []
        self._peer_xy_snapshot: List[Tuple[float, float]] = []

        self._wheel_prims: List[Tuple[Usd.Prim, float]] = []
        self._wheel_angles_deg: List[float] = []

        self.use_physx_lidar = True
        self.lidar_max_range = 3.5
        self.lidar_angles = np.linspace(-math.radians(50.0), math.radians(50.0), 35)
        self.last_lidar_scan: List[Tuple[float, float]] = []
        self.lidar_stop_distance = 0.65
        self.lidar_obstacle_detected = False
        self.min_front_obstacle_dist = 99.0

        self._pallet_attached = False
        self._carrier_moved = False
        self._native_pose_recorded = False
        self.pallet_init_pos = np.array([0.688, 0.575, -0.82], dtype=float)
        self.pallet_init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.cart_init_pos: Optional[np.ndarray] = None
        self.cart_init_quat: Optional[np.ndarray] = None
        self.cart_xy_offset = np.zeros(2, dtype=float)

        self._carried_bins: list = []
        self._bin_rel_offsets: List[np.ndarray] = []
        self._bin_init_quats: List[np.ndarray] = []

        self.wait_timer = 0.0
        self.wait_duration = 1.0
        self.external_safety_halt = False

    def setup(self, world=None, planner: Optional[WarehousePathPlanner] = None) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        agv_prim = stage.GetPrimAtPath(self.prim_path)
        if not agv_prim.IsValid():
            print(f"[AGV ERROR] Prim not found at {self.prim_path}")
            return

        self._ensure_meshes_visible(agv_prim)
        self._remove_custom_clutter(stage)
        self._calibrate_ground_stance_and_decals(stage, agv_prim)
        self._discover_wheel_prims(agv_prim)
        self._sync_visual_identity_attrs(agv_prim)
        self.articulation_path = self._find_articulation_root(agv_prim)
        self._discover_native_carrier_prims(stage)

        self.wait_pos[2] = self.floor_z + self.root_z_offset
        self.current_pos = self.wait_pos.copy()
        self.current_yaw = 0.0
        self._write_root_transform()

        self.pallet_pos[2] = self.floor_z
        self.dropoff_pos[2] = self.floor_z
        self.approach_pos = np.array(
            [
                float(self.pallet_pos[0] + self.dock_approach_offset[0]),
                float(self.pallet_pos[1] + self.dock_approach_offset[1]),
                self.floor_z,
            ],
            dtype=float,
        )

        if planner is not None:
            self.planner = planner
        else:
            self.planner = WarehousePathPlanner()
            self.planner.build_from_stage(
                stage,
                (float(self.pallet_pos[0]), float(self.pallet_pos[1])),
                approach_xy=(float(self.approach_pos[0]), float(self.approach_pos[1])),
            )

        self._set_task_flag("STANDBY")
        print(
            f"[AGV] {self.robot_id} ({self.ip_address}) ready at Z={self.current_pos[2]:.3f} "
            f"(cart={self.cart_prim_path}, dock_offset=({self.dock_approach_offset[0]:.2f}, {self.dock_approach_offset[1]:.2f}))."
        )

    def _calibrate_ground_stance_and_decals(self, stage, agv_prim) -> None:
        self.floor_z = -1.180

        xf_root = UsdGeom.Xformable(agv_prim)
        xf_root.ClearXformOpOrder()
        xf_root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xf_root.AddRotateZOp().Set(0.0)

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        min_z_vals = []
        max_z_vals = []
        max_y_vals = []
        x_centers = []

        for prim in Usd.PrimRange(agv_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            name_low = str(prim.GetPath()).lower()
            if any(skip in name_low for skip in ("numberplate", "coll", "proxy", "lidar", "sensor", "camera")):
                continue
            img = UsdGeom.Imageable(prim)
            if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue

            rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            bmin = rng.GetMin()
            bmax = rng.GetMax()
            if abs(float(bmin[2])) > 0.30 or abs(float(bmax[2])) > 0.90:
                continue
            if abs(float(bmax[1])) > 0.60 or abs(float(bmax[0])) > 1.00:
                continue

            min_z_vals.append(float(bmin[2]))
            max_z_vals.append(float(bmax[2]))
            max_y_vals.append(max(abs(float(bmin[1])), abs(float(bmax[1]))))
            x_centers.append(0.5 * (float(bmin[0]) + float(bmax[0])))

        if min_z_vals and max_z_vals:
            mesh_min_z = min(min_z_vals)
            mesh_max_z = max(max_z_vals)
            self.root_z_offset = float(np.clip(-mesh_min_z + 0.002, -0.02, 0.08))
            half_width_y = float(np.clip(max(max_y_vals), 0.25, 0.35))
            center_x = float(np.mean(x_centers)) if x_centers else 0.0
            chassis_z = mesh_min_z + 0.42 * (mesh_max_z - mesh_min_z)
        else:
            self.root_z_offset = 0.0
            half_width_y = 0.30
            center_x = 0.0
            chassis_z = 0.16

        if self.authorized:
            self._attach_side_number_decals(stage, half_width_y=half_width_y, center_x=center_x, plate_z=chassis_z)

    def _attach_side_number_decals(
        self,
        stage,
        half_width_y: float = 0.30,
        center_x: float = 0.0,
        plate_z: float = 0.16,
    ) -> None:
        decal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decals")
        png_path = _generate_number_decal_png(self.robot_id, self.marker_id, decal_dir)

        mat_path = f"{self.prim_path}/Looks/NumberDecalMat"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)

        if png_path and os.path.exists(png_path):
            tex = UsdShade.Shader.Define(stage, f"{mat_path}/DiffuseTex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(png_path.replace("\\", "/")))
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

            st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/STReader")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
            shader.CreateInput("emissiveScale", Sdf.ValueTypeNames.Float).Set(0.20)
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.88, 0.08, 0.08))

        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        hw, hh = 0.19, 0.046
        plate_x = -0.3494
        plate_y = 0.3258
        plate_z = 0.0777

        sides = [
            (
                "NumberPlate_Left",
                plate_y,
                [(hw, 0.0, -hh), (-hw, 0.0, -hh), (-hw, 0.0, hh), (hw, 0.0, hh)],
                [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)],
            ),
            (
                "NumberPlate_Right",
                -plate_y,
                [(-hw, 0.0, -hh), (hw, 0.0, -hh), (hw, 0.0, hh), (-hw, 0.0, hh)],
                [Gf.Vec2f(0.0, 0.0), Gf.Vec2f(1.0, 0.0), Gf.Vec2f(1.0, 1.0), Gf.Vec2f(0.0, 1.0)],
            ),
        ]

        for plate_name, y_offset, pts, uvs in sides:
            plate_path = f"{self.prim_path}/{plate_name}"
            if stage.GetPrimAtPath(plate_path).IsValid():
                stage.RemovePrim(plate_path)

            mesh = UsdGeom.Mesh.Define(stage, plate_path)
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh.CreatePurposeAttr(UsdGeom.Tokens.render)
            mesh.CreateDoubleSidedAttr(True)

            pv_api = UsdGeom.PrimvarsAPI(mesh)
            st = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(uvs)

            xf = UsdGeom.Xformable(mesh.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(plate_x, y_offset, plate_z))

            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def _discover_wheel_prims(self, root_prim) -> None:
        self._wheel_prims.clear()
        self._wheel_angles_deg.clear()
        for prim in Usd.PrimRange(root_prim):
            if prim == root_prim or not prim.IsA(UsdGeom.Xformable):
                continue
            name_low = prim.GetName().lower()
            if "wheel" in name_low or "tire" in name_low or "roller" in name_low:
                parent_low = prim.GetParent().GetName().lower() if prim.GetParent() else ""
                if "wheel" in parent_low or "tire" in parent_low:
                    continue
                side_sign = -1.0 if ("right" in name_low or "_r" in name_low) else 1.0
                self._wheel_prims.append((prim, side_sign))
                self._wheel_angles_deg.append(0.0)

    def _animate_wheels(self, linear_vel: float, angular_vel: float, dt: float) -> None:
        if not self._wheel_prims:
            return
        if abs(linear_vel) < 1e-4 and abs(angular_vel) < 1e-4:
            return

        for idx, (wprim, side_sign) in enumerate(self._wheel_prims):
            if not wprim.IsValid():
                continue
            v_wheel = linear_vel - side_sign * angular_vel * (0.5 * self.wheel_base)
            d_deg = math.degrees((v_wheel * dt) / max(0.03, self.wheel_radius))
            self._wheel_angles_deg[idx] = (self._wheel_angles_deg[idx] + d_deg) % 360.0

            xf = UsdGeom.Xformable(wprim)
            ops = xf.GetOrderedXformOps()
            rot_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateY:
                    rot_op = op
                    break
            if rot_op is None:
                rot_op = xf.AddRotateYOp()
            rot_op.Set(float(self._wheel_angles_deg[idx]))

    def _sync_visual_identity_attrs(self, agv_prim) -> None:
        attrs = {
            "agv:robot_id": self.robot_id,
            "agv:ip_address": self.ip_address,
            "agv:marker_id": str(self.marker_id),
            "agv:authorized": str(self.authorized),
            "agv:status_flag": self.task_flag,
            "agv:cargo_flag": self.cargo_flag,
        }
        for k, v in attrs.items():
            attr = agv_prim.GetAttribute(k)
            if not attr:
                attr = agv_prim.CreateAttribute(k, Sdf.ValueTypeNames.String, custom=True)
            attr.Set(v)

    def _ensure_meshes_visible(self, root_prim) -> None:
        root_prim.Load()
        if root_prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(root_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)

        for prim in Usd.PrimRange(root_prim):
            if prim.HasPayload():
                prim.Load()
            if not prim.IsA(UsdGeom.Imageable):
                continue

            img = UsdGeom.Imageable(prim)
            path_lower = str(prim.GetPath()).lower()
            purpose_attr = img.GetPurposeAttr()
            purpose = purpose_attr.Get() if purpose_attr else None

            is_collision_or_proxy = (
                purpose in (UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide)
                or "coll" in path_lower
                or "proxy" in path_lower
                or prim.HasAPI(UsdPhysics.CollisionAPI)
            )

            if is_collision_or_proxy:
                img.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            else:
                vis = img.GetVisibilityAttr()
                if vis and vis.Get() == UsdGeom.Tokens.invisible:
                    vis.Set(UsdGeom.Tokens.inherited)

    def _remove_custom_clutter(self, stage) -> None:
        clutter_parts = (
            "Chassis_Lower", "Chassis_UpperDeck", "Lift_Platform",
            "Front_Lidar_Hood", "Front_Lidar_Lens",
            "Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR",
            "Skirt_Center", "Skirt_Front", "Skirt_Rear",
            "Body_Center", "Body_Front", "Body_Rear",
            "Collar_Center", "Collar_Front", "Collar_Rear",
            "Lift_Table", "Lidar_Window_Front", "Lidar_Beam_Front", "Lidar_Window_Rear",
            "Wheel_Drive_L", "Wheel_Drive_R",
            "Wheel_Caster_FL", "Wheel_Caster_FR", "Wheel_Caster_RL", "Wheel_Caster_RR",
            "Badge_Front", "Badge_Rear", "Badge_Left", "Badge_Right",
            "StatusLED_Left", "StatusLED_Right",
        )
        for part in clutter_parts:
            p = f"{self.prim_path}/{part}"
            if stage.GetPrimAtPath(p).IsValid():
                stage.RemovePrim(p)

    def _set_task_flag(self, flag: str, cargo_flag: Optional[str] = None) -> None:
        self.task_flag = flag
        if cargo_flag is not None:
            self.cargo_flag = cargo_flag
        stage = omni.usd.get_context().get_stage()
        if stage:
            agv_prim = stage.GetPrimAtPath(self.prim_path)
            if agv_prim.IsValid():
                self._sync_visual_identity_attrs(agv_prim)

    def _find_articulation_root(self, root_prim) -> str:
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(prim.GetPath())
        return str(root_prim.GetPath())

    @staticmethod
    def _read_world_pose(prim) -> Tuple[np.ndarray, np.ndarray]:
        tf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        trans = tf.ExtractTranslation()
        rot = tf.ExtractRotationQuat()
        pos = np.array([float(trans[0]), float(trans[1]), float(trans[2])], dtype=float)
        quat = np.array(
            [
                float(rot.GetReal()),
                float(rot.GetImaginary()[0]),
                float(rot.GetImaginary()[1]),
                float(rot.GetImaginary()[2]),
            ],
            dtype=float,
        )
        return pos, quat

    @staticmethod
    def _yaw_to_quat_wxyz(yaw_rad: float) -> np.ndarray:
        half = 0.5 * yaw_rad
        return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=float)

    @staticmethod
    def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    def _discover_native_carrier_prims(self, stage) -> None:
        """Read pallet & blue dolly native world poses without moving them."""
        for path in (self.pallet_prim_path, "/World/Ur10Table/pallet", "/World/Ur10Table/Pallet", "/World/Pallet"):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                self.pallet_prim_path = path
                self.pallet_init_pos, self.pallet_init_quat = self._read_world_pose(prim)
                self.pallet_pos[0] = float(self.pallet_init_pos[0])
                self.pallet_pos[1] = float(self.pallet_init_pos[1])
                break

        found_cart = None
        if self.cart_prim_path:
            cprim = stage.GetPrimAtPath(self.cart_prim_path)
            if cprim.IsValid():
                found_cart = cprim

        if found_cart is None:
            table_root = stage.GetPrimAtPath("/World/Ur10Table")
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            cart_keywords = ("cart", "dolly", "trolley", "rack", "stand", "holder", "carrier")
            ignore_names = ("ur10", "pallet", "obstacles", "bins", "conveyor", "rail", "flip")

            if table_root.IsValid():
                # 1. Search by explicit cart/dolly/rack naming anywhere under Ur10Table
                for desc in Usd.PrimRange(table_root):
                    if desc == table_root or not desc.IsA(UsdGeom.Xformable):
                        continue
                    dname = desc.GetName().lower()
                    if any(ig in dname for ig in ignore_names):
                        continue
                    if any(kw in dname for kw in cart_keywords):
                        rng = bbox_cache.ComputeWorldBound(desc).ComputeAlignedRange()
                        if not rng.IsEmpty():
                            center = rng.GetMidpoint()
                            if math.hypot(
                                float(center[0]) - float(self.pallet_init_pos[0]),
                                float(center[1]) - float(self.pallet_init_pos[1]),
                            ) < 0.35:
                                found_cart = desc
                                break

                # 2. Fallback: find Xformable directly beneath pallet center
                if found_cart is None:
                    for child in table_root.GetChildren():
                        cname = child.GetName().lower()
                        if any(ig in cname for ig in ignore_names):
                            continue
                        rng = bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
                        if rng.IsEmpty():
                            continue
                        center = rng.GetMidpoint()
                        horiz_dist = math.hypot(
                            float(center[0]) - float(self.pallet_init_pos[0]),
                            float(center[1]) - float(self.pallet_init_pos[1]),
                        )
                        if horiz_dist < 0.25 and float(center[2]) <= float(self.pallet_init_pos[2]) + 0.05:
                            found_cart = child
                            break

        if found_cart is not None and found_cart.IsValid():
            self.cart_prim_path = str(found_cart.GetPath())
            self.cart_init_pos, self.cart_init_quat = self._read_world_pose(found_cart)
            self.cart_xy_offset = self.cart_init_pos[:2] - self.pallet_init_pos[:2]
            self.pallet_pos[0] = float(self.cart_init_pos[0])
            self.pallet_pos[1] = float(self.cart_init_pos[1])
        else:
            self.cart_prim_path = None
            self.cart_init_pos = None
            self.cart_init_quat = None
            self.cart_xy_offset = np.zeros(2, dtype=float)

        self._native_pose_recorded = True
        self._carrier_moved = False

    def _capture_native_pose_before_lift(self, stage) -> None:
        pprim = stage.GetPrimAtPath(self.pallet_prim_path)
        if pprim.IsValid():
            self.pallet_init_pos, self.pallet_init_quat = self._read_world_pose(pprim)
        if self.cart_prim_path:
            cprim = stage.GetPrimAtPath(self.cart_prim_path)
            if cprim.IsValid():
                self.cart_init_pos, self.cart_init_quat = self._read_world_pose(cprim)
                self.cart_xy_offset = self.cart_init_pos[:2] - self.pallet_init_pos[:2]
        self._native_pose_recorded = True

    def _prepare_carrier_kinematics(self, stage) -> None:
        for p in (self.pallet_prim_path, self.cart_prim_path):
            if not p:
                continue
            prim = stage.GetPrimAtPath(p)
            if not prim.IsValid():
                continue
            for desc in Usd.PrimRange(prim):
                if desc.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(desc).GetKinematicEnabledAttr().Set(True)

    def _set_bins_kinematic(self, kinematic: bool) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        for b in self._carried_bins:
            try:
                prim = stage.GetPrimAtPath(b.prim_path)
                if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(kinematic)
                if not kinematic:
                    b.set_linear_velocity(np.zeros(3))
                    b.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def restore_pallet_nominal_pose(self) -> None:
        self._pallet_attached = False
        self.lift_height = 0.0
        if not self._carrier_moved or not self._native_pose_recorded:
            return

        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        self._prepare_carrier_kinematics(stage)
        if stage.GetPrimAtPath(self.pallet_prim_path).IsValid():
            SingleXFormPrim(self.pallet_prim_path).set_world_pose(
                position=self.pallet_init_pos,
                orientation=self.pallet_init_quat,
            )
        if self.cart_prim_path and self.cart_init_pos is not None and stage.GetPrimAtPath(self.cart_prim_path).IsValid():
            SingleXFormPrim(self.cart_prim_path).set_world_pose(
                position=self.cart_init_pos,
                orientation=self.cart_init_quat,
            )
        self._carrier_moved = False

    def attach_stacked_bins(self, bins: list) -> None:
        self._carried_bins = list(bins) if bins else []
        self._bin_rel_offsets.clear()
        self._bin_init_quats.clear()
        for b in self._carried_bins:
            try:
                b_pos, b_quat = b.get_world_pose()
                self._bin_rel_offsets.append(np.array(b_pos, dtype=float) - self.pallet_init_pos)
                self._bin_init_quats.append(np.array(b_quat, dtype=float))
                b.set_linear_velocity(np.zeros(3))
                b.set_angular_velocity(np.zeros(3))
            except Exception:
                self._bin_rel_offsets.append(np.array([0.0, 0.0, 0.12], dtype=float))
                self._bin_init_quats.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

    def release_carried_bins(self, unload_callback=None) -> None:
        self._set_bins_kinematic(False)
        self._carried_bins.clear()
        self._bin_rel_offsets.clear()
        self._bin_init_quats.clear()
        self.cargo_flag = "PALLET_EMPTY"
        if unload_callback:
            unload_callback()

    def _poll_lidar_obstacles(self, peer_positions: Optional[List[Tuple[str, np.ndarray, int]]] = None) -> bool:
        blocked = False
        min_dist = 99.0
        self.last_lidar_scan.clear()

        if self.use_physx_lidar:
            try:
                sq = omni.physx.get_physx_scene_query_interface()
                sensor_z = float(self.floor_z) + 0.16
                forward_offset = 0.45

                for az in self.lidar_angles:
                    world_angle = self.current_yaw + float(az)
                    dx = math.cos(world_angle)
                    dy = math.sin(world_angle)
                    origin = (
                        float(self.current_pos[0]) + dx * forward_offset,
                        float(self.current_pos[1]) + dy * forward_offset,
                        sensor_z,
                    )
                    hit = sq.raycast_closest(origin, (dx, dy, 0.0), self.lidar_max_range)
                    if hit.get("hit", False):
                        hit_prim = str(hit.get("rigidBody", "") or hit.get("collision", "")).lower()
                        if (
                            self.prim_path.lower() in hit_prim
                            or "pallet" in hit_prim
                            or "cart" in hit_prim
                            or "dolly" in hit_prim
                            or "trolley" in hit_prim
                            or "bin_" in hit_prim
                            or "ur10table" in hit_prim
                        ):
                            continue
                        dist = float(hit["distance"])
                        self.last_lidar_scan.append((float(az), dist))
                        if abs(az) <= math.radians(22.0) and dist < min_dist:
                            min_dist = dist
                            if dist < self.lidar_stop_distance:
                                blocked = True
            except Exception:
                pass

        if peer_positions:
            heading = np.array([math.cos(self.current_yaw), math.sin(self.current_yaw)])
            for peer_id, peer_pos, peer_prio in peer_positions:
                if peer_id == self.robot_id:
                    continue
                rel = peer_pos[:2] - self.current_pos[:2]
                dist = float(np.linalg.norm(rel))
                if 1e-3 < dist < 1.15:
                    forward_proj = float(np.dot(rel / dist, heading))
                    if forward_proj > 0.55 and peer_prio <= self.priority:
                        blocked = True
                        min_dist = min(min_dist, dist)

        self.min_front_obstacle_dist = min_dist
        self.lidar_obstacle_detected = blocked
        return blocked

    def _attach_pallet_joint(self) -> None:
        if self._pallet_attached:
            return
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        self._prepare_carrier_kinematics(stage)
        self._set_bins_kinematic(True)
        self.carrier_base_yaw = float(self.current_yaw)
        self._pallet_attached = True
        self._carrier_moved = True
        self.cargo_flag = "PALLET_KLT_LOADED"
        self._sync_carried_pallet_pose()
        print(f"[AGV] {self.robot_id} lifted entire blue dolly + pallet (caster wheels on sides).")

    def _detach_pallet_joint(self) -> None:
        if not self._pallet_attached:
            return
        self._pallet_attached = False
        self._set_bins_kinematic(False)
        self.cargo_flag = "EMPTY"
        self.restore_pallet_nominal_pose()
        print(f"[AGV] {self.robot_id} lowered blue dolly + pallet at native pose.")

    def _apply_carrier_poses(self, stage, height_m: float) -> None:
        if height_m > 1e-5 or self._pallet_attached:
            self._carrier_moved = True

        delta_yaw = float(self.current_yaw - self.carrier_base_yaw)
        c, s = math.cos(delta_yaw), math.sin(delta_yaw)
        q_rot = self._yaw_to_quat_wxyz(delta_yaw)

        base_x = float(self.current_pos[0]) if self._pallet_attached else float(self.pallet_init_pos[0])
        base_y = float(self.current_pos[1]) if self._pallet_attached else float(self.pallet_init_pos[1])

        lifted_pallet_pos = np.array(
            [base_x, base_y, float(self.pallet_init_pos[2] + height_m)],
            dtype=float,
        )

        pallet_prim = stage.GetPrimAtPath(self.pallet_prim_path)
        if pallet_prim.IsValid():
            rot_pallet_quat = self._quat_mul_wxyz(q_rot, self.pallet_init_quat)
            SingleXFormPrim(self.pallet_prim_path).set_world_pose(
                position=lifted_pallet_pos,
                orientation=rot_pallet_quat,
            )

        if self.cart_prim_path and self.cart_init_pos is not None and self.cart_init_quat is not None:
            cart_prim = stage.GetPrimAtPath(self.cart_prim_path)
            if cart_prim.IsValid():
                ox, oy = float(self.cart_xy_offset[0]), float(self.cart_xy_offset[1])
                lifted_cart_pos = np.array(
                    [
                        float(base_x + (ox * c - oy * s)),
                        float(base_y + (ox * s + oy * c)),
                        float(self.cart_init_pos[2] + height_m),
                    ],
                    dtype=float,
                )
                rot_cart_quat = self._quat_mul_wxyz(q_rot, self.cart_init_quat)
                SingleXFormPrim(self.cart_prim_path).set_world_pose(
                    position=lifted_cart_pos,
                    orientation=rot_cart_quat,
                )

        for idx, rigid_bin in enumerate(self._carried_bins):
            if idx >= len(self._bin_rel_offsets):
                break
            rel = self._bin_rel_offsets[idx]
            rot_bx = float(rel[0]) * c - float(rel[1]) * s
            rot_by = float(rel[0]) * s + float(rel[1]) * c
            bin_pos = np.array(
                [
                    float(lifted_pallet_pos[0] + rot_bx),
                    float(lifted_pallet_pos[1] + rot_by),
                    float(lifted_pallet_pos[2] + float(rel[2])),
                ],
                dtype=float,
            )
            bin_quat = self._quat_mul_wxyz(q_rot, self._bin_init_quats[idx])
            try:
                rigid_bin.set_world_pose(position=bin_pos, orientation=bin_quat)
                rigid_bin.set_linear_velocity(np.zeros(3))
                rigid_bin.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _sync_carried_pallet_pose(self) -> None:
        if not self._pallet_attached:
            return
        stage = omni.usd.get_context().get_stage()
        if stage:
            self._apply_carrier_poses(stage, self.lift_height)

    def _plan_route_to(self, goal_pos: np.ndarray) -> None:
        if self.planner is not None:
            raw_wps = self.planner.plan_path(
                (float(self.current_pos[0]), float(self.current_pos[1])),
                (float(goal_pos[0]), float(goal_pos[1])),
                dynamic_obstacles=self._peer_xy_snapshot,
            )
            self._active_waypoints = [
                np.array([wp[0], wp[1], self.floor_z + self.root_z_offset], dtype=float) for wp in raw_wps
            ]
        else:
            self._active_waypoints = [goal_pos.copy()]

    def trigger_pallet_swap(self) -> None:
        if self.state == "IDLE":
            stage = omni.usd.get_context().get_stage()
            if stage:
                self._capture_native_pose_before_lift(stage)
            self.approach_pos = np.array(
                [
                    float(self.pallet_pos[0] + self.dock_approach_offset[0]),
                    float(self.pallet_pos[1] + self.dock_approach_offset[1]),
                    self.floor_z + self.root_z_offset,
                ],
                dtype=float,
            )
            self._plan_route_to(self.approach_pos)
            self.state = "APPROACHING_PALLET"
            self._set_task_flag("FETCH_FULL_PALLET", cargo_flag="EMPTY")
            print(
                f"[LOGISTICS] {self.robot_id} ({self.ip_address}) dispatched -> "
                f"longitudinal A* route ({len(self._active_waypoints)} wps) to +X open dolly end."
            )

    def trigger_patrol_intrusion(self, target_xy: Tuple[float, float]) -> None:
        goal = np.array([target_xy[0], target_xy[1], self.floor_z + self.root_z_offset], dtype=float)
        self._plan_route_to(goal)
        self.state = "TRANSIT_PATROL"
        self._set_task_flag("UNSCHEDULED_TRANSIT")

    def set_external_safety_halt(self, halt: bool) -> None:
        self.external_safety_halt = halt

    def get_network_telemetry(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "ip_address": self.ip_address,
            "marker_id": self.marker_id,
            "authorized": self.authorized,
            "task_flag": self.task_flag,
            "cargo_flag": self.cargo_flag,
            "state": self.state,
            "net_pos": (round(float(self.current_pos[0]), 3), round(float(self.current_pos[1]), 3)),
            "lidar_blocked": self.lidar_obstacle_detected,
            "lidar_min_dist": round(float(self.min_front_obstacle_dist), 2),
        }

    def _write_root_transform(self) -> None:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.prim_path) if stage else None
        if prim and prim.IsValid():
            self.current_pos[2] = self.floor_z + self.root_z_offset
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*self.current_pos.tolist()))
            xf.AddRotateZOp().Set(math.degrees(self.current_yaw))

    def _apply_wheel_velocities(self, linear_vel: float, angular_vel: float, dt: float = 1.0 / 60.0) -> None:
        self.current_yaw = math.atan2(
            math.sin(self.current_yaw + angular_vel * dt),
            math.cos(self.current_yaw + angular_vel * dt),
        )
        heading = np.array([math.cos(self.current_yaw), math.sin(self.current_yaw), 0.0])
        self.current_pos += heading * (linear_vel * dt)
        self._write_root_transform()
        self._animate_wheels(linear_vel, angular_vel, dt)

    def _align_to_yaw(self, target_yaw: float, tol_deg: float = 1.5, dt: float = 1.0 / 60.0) -> bool:
        yaw_error = math.atan2(
            math.sin(target_yaw - self.current_yaw),
            math.cos(target_yaw - self.current_yaw),
        )
        if abs(yaw_error) <= math.radians(tol_deg):
            self.current_yaw = target_yaw
            self._apply_wheel_velocities(0.0, 0.0, dt)
            return True

        ang_vel = float(np.clip(2.4 * yaw_error, -self.max_angular_speed, self.max_angular_speed))
        if abs(ang_vel) < 0.25:
            ang_vel = math.copysign(0.25, yaw_error)
        self._apply_wheel_velocities(0.0, ang_vel, dt)
        return False

    def _drive_towards(
        self,
        target: np.ndarray,
        cruise_speed: float,
        reverse: bool = False,
        arrival_tol: float = 0.08,
        dt: float = 1.0 / 60.0,
    ) -> bool:
        diff = target[:2] - self.current_pos[:2]
        distance = float(np.linalg.norm(diff))

        if distance <= arrival_tol:
            return True

        desired_yaw = math.atan2(float(diff[1]), float(diff[0]))
        if reverse:
            desired_yaw = math.atan2(math.sin(desired_yaw + math.pi), math.cos(desired_yaw + math.pi))

        yaw_error = math.atan2(
            math.sin(desired_yaw - self.current_yaw),
            math.cos(desired_yaw - self.current_yaw),
        )

        if abs(yaw_error) > math.radians(45.0):
            ang_vel = float(np.clip(1.8 * yaw_error, -self.max_angular_speed, self.max_angular_speed))
            if abs(ang_vel) < 0.25:
                ang_vel = math.copysign(0.25, yaw_error)
            self._apply_wheel_velocities(0.0, ang_vel, dt)
            return False

        turn_factor = max(0.25, math.cos(yaw_error))
        min_creep = 0.08 if arrival_tol < 0.04 else 0.18
        lin_vel = min(cruise_speed, max(min_creep, distance * 1.35)) * turn_factor
        if reverse:
            lin_vel = -lin_vel
        ang_vel = float(np.clip(1.5 * yaw_error, -self.max_angular_speed, self.max_angular_speed))
        self._apply_wheel_velocities(lin_vel, ang_vel, dt)
        return False

    def _follow_waypoints(self, cruise_speed: float, dt: float = 1.0 / 60.0) -> bool:
        if not self._active_waypoints:
            return True
        tol = 0.05 if len(self._active_waypoints) == 1 else 0.10
        if self._drive_towards(self._active_waypoints[0], cruise_speed, arrival_tol=tol, dt=dt):
            self._active_waypoints.pop(0)
        return len(self._active_waypoints) == 0

    def _set_lift_height(self, height_m: float) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        agv_prim = stage.GetPrimAtPath(self.prim_path)
        if agv_prim.IsValid():
            for desc in Usd.PrimRange(agv_prim):
                if "lift" in desc.GetName().lower():
                    drive = UsdPhysics.DriveAPI.Get(desc, "linear")
                    if drive:
                        drive.GetTargetPositionAttr().Set(float(height_m))

        self._apply_carrier_poses(stage, height_m)

    def update(
        self,
        dt: float,
        spawner_reset_callback=None,
        unload_bins_callback=None,
        peer_positions: Optional[List[Tuple[str, np.ndarray, int]]] = None,
    ) -> None:
        if peer_positions:
            self._peer_xy_snapshot = [
                (float(p[0]), float(p[1])) for pid, p, _ in peer_positions if pid != self.robot_id
            ]

        if self.state == "IDLE":
            return

        if self.external_safety_halt:
            return

        step_dt = float(dt) if (dt is not None and dt > 1e-5) else (1.0 / 60.0)

        obstacle_check_states = {
            "APPROACHING_PALLET",
            "TRANSPORT_TO_DROPOFF",
            "RETURNING_WITH_EMPTY",
            "RETURNING_STANDBY",
            "TRANSIT_PATROL",
        }
        if self.state in obstacle_check_states and self._poll_lidar_obstacles(peer_positions):
            return

        if self._pallet_attached:
            self._sync_carried_pallet_pose()

        if self.state == "TRANSIT_PATROL":
            if self._follow_waypoints(self.max_linear_speed, dt=step_dt):
                self.state = "IDLE"
                self._set_task_flag("STANDBY")

        elif self.state == "APPROACHING_PALLET":
            if self._follow_waypoints(self.max_linear_speed, dt=step_dt):
                self.state = "ALIGNING_DOCK"
                self._set_task_flag("DOCKING_UNDER_PALLET")

        elif self.state == "ALIGNING_DOCK":
            dx = float(self.pallet_pos[0] - self.current_pos[0])
            dy = float(self.pallet_pos[1] - self.current_pos[1])
            dock_yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.05 else math.pi
            if self._align_to_yaw(dock_yaw, tol_deg=1.2, dt=step_dt):
                self.state = "DOCKING"

        elif self.state == "DOCKING":
            if self._drive_towards(self.pallet_pos, self.dock_linear_speed, arrival_tol=0.015, dt=step_dt):
                self._apply_wheel_velocities(0.0, 0.0, step_dt)
                stage = omni.usd.get_context().get_stage()
                if stage:
                    self._capture_native_pose_before_lift(stage)
                    self._prepare_carrier_kinematics(stage)
                self.carrier_base_yaw = float(self.current_yaw)
                self._set_bins_kinematic(True)
                self.state = "LIFTING"
                self._set_task_flag("LIFTING_PALLET")

        elif self.state == "LIFTING":
            self.lift_height = min(self.max_lift_height, self.lift_height + self.lift_speed * step_dt)
            self._set_lift_height(self.lift_height)
            if self.lift_height >= self.max_lift_height - 1e-4:
                self._attach_pallet_joint()
                self.state = "BACKING_OUT_WITH_LOAD"
                self._set_task_flag("HAULING_LOADED_PALLET", cargo_flag="PALLET_KLT_LOADED")

        elif self.state == "BACKING_OUT_WITH_LOAD":
            if self._drive_towards(self.approach_pos, self.dock_linear_speed, reverse=True, arrival_tol=0.05, dt=step_dt):
                self._plan_route_to(self.dropoff_pos)
                self.state = "TRANSPORT_TO_DROPOFF"

        elif self.state == "TRANSPORT_TO_DROPOFF":
            if self._follow_waypoints(self.max_linear_speed, dt=step_dt):
                self.state = "UNLOADING_AT_DROPOFF"
                self._set_task_flag("SWAPPING_EMPTY_PALLET")
                self.wait_timer = 0.0

        elif self.state == "UNLOADING_AT_DROPOFF":
            self.wait_timer += step_dt
            if self.wait_timer >= self.wait_duration:
                self.release_carried_bins(unload_callback=unload_bins_callback)
                self._plan_route_to(self.approach_pos)
                self.state = "RETURNING_WITH_EMPTY"
                self._set_task_flag("RETURNING_EMPTY_PALLET", cargo_flag="PALLET_EMPTY")

        elif self.state == "RETURNING_WITH_EMPTY":
            if self._follow_waypoints(self.max_linear_speed, dt=step_dt):
                self.state = "ALIGNING_DOCK_EMPTY"
                self._set_task_flag("PLACING_EMPTY_PALLET")

        elif self.state == "ALIGNING_DOCK_EMPTY":
            dx = float(self.pallet_pos[0] - self.current_pos[0])
            dy = float(self.pallet_pos[1] - self.current_pos[1])
            dock_yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.05 else math.pi
            if self._align_to_yaw(dock_yaw, tol_deg=1.2, dt=step_dt):
                self.state = "DOCKING_EMPTY"

        elif self.state == "DOCKING_EMPTY":
            if self._drive_towards(self.pallet_pos, self.dock_linear_speed, arrival_tol=0.015, dt=step_dt):
                self._apply_wheel_velocities(0.0, 0.0, step_dt)
                self.state = "LOWERING"
                self._set_task_flag("LOWERING_PALLET")

        elif self.state == "LOWERING":
            self.lift_height = max(0.0, self.lift_height - self.lift_speed * step_dt)
            self._set_lift_height(self.lift_height)
            if self.lift_height <= 1e-4:
                self._detach_pallet_joint()
                self.state = "UNDOCKING"
                self._set_task_flag("UNDOCKING", cargo_flag="EMPTY")

        elif self.state == "UNDOCKING":
            if self._drive_towards(self.approach_pos, self.dock_linear_speed, reverse=True, arrival_tol=0.05, dt=step_dt):
                self._plan_route_to(self.wait_pos)
                if spawner_reset_callback:
                    spawner_reset_callback()
                self.state = "ALIGNING_EXIT"
                self._set_task_flag("RETURNING_STANDBY")

        elif self.state == "ALIGNING_EXIT":
            if self._active_waypoints:
                dx = float(self._active_waypoints[0][0] - self.current_pos[0])
                dy = float(self._active_waypoints[0][1] - self.current_pos[1])
                exit_yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.05 else math.pi
            else:
                exit_yaw = math.pi
            if self._align_to_yaw(exit_yaw, tol_deg=2.0, dt=step_dt):
                self.state = "RETURNING_STANDBY"

        elif self.state == "RETURNING_STANDBY":
            if self._follow_waypoints(self.max_linear_speed, dt=step_dt):
                self.state = "ALIGNING_STANDBY"

        elif self.state == "ALIGNING_STANDBY":
            if self._align_to_yaw(0.0, tol_deg=2.0, dt=step_dt):
                self.state = "IDLE"
                self._set_task_flag("STANDBY", cargo_flag="EMPTY")
                print(f"[LOGISTICS] {self.robot_id} returned to standby bay.")


class AGVFleetManager:
    """Coordinates multiple iw.hub rovers, dispatches nearest FREE authorized rover, and exposes telemetry."""

    def __init__(self, rovers: List[AGVManager]):
        self.rovers = rovers
        self._authorized_rovers = [r for r in rovers if r.authorized]

    def setup(self, world=None) -> None:
        stage = omni.usd.get_context().get_stage()
        shared_planner = WarehousePathPlanner()
        for rover in self.rovers:
            rover.setup(world=world, planner=shared_planner)
        if stage and self.rovers:
            r0 = self.rovers[0]
            shared_planner.build_from_stage(
                stage,
                (float(r0.pallet_pos[0]), float(r0.pallet_pos[1])),
                approach_xy=(float(r0.approach_pos[0]), float(r0.approach_pos[1])),
            )

    def reset_all_carriers_to_nominal(self) -> None:
        """Restores pallet and blue cart back to their native pose ONLY if an AGV moved them."""
        for rover in self.rovers:
            try:
                rover.restore_pallet_nominal_pose()
            except Exception:
                pass

    def has_active_service(self) -> bool:
        return any(r.authorized and r.state in AGVManager.CELL_BUSY_STATES for r in self.rovers)

    def dispatch_next_available(self, stacked_bins: Optional[list] = None) -> Optional[AGVManager]:
        candidates = [
            r for r in self._authorized_rovers
            if r.state == "IDLE" and not r.external_safety_halt
        ]
        if not candidates:
            return None

        candidates.sort(key=lambda r: float(np.linalg.norm(r.current_pos[:2] - r.approach_pos[:2])))
        chosen = candidates[0]
        chosen.attach_stacked_bins(stacked_bins or [])
        chosen.trigger_pallet_swap()
        return chosen

    def update(self, dt: float, spawner_reset_callback=None, unload_bins_callback=None) -> None:
        peer_snapshot = [(r.robot_id, r.current_pos.copy(), r.priority) for r in self.rovers]
        for rover in self.rovers:
            rover.update(
                dt,
                spawner_reset_callback=spawner_reset_callback,
                unload_bins_callback=unload_bins_callback,
                peer_positions=peer_snapshot,
            )

    def get_all_telemetry(self) -> List[dict]:
        return [r.get_network_telemetry() for r in self.rovers]

    def apply_security_halts(self, blocked_ids: set) -> None:
        for rover in self.rovers:
            rover.set_external_safety_halt(rover.robot_id in blocked_ids)