import heapq
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni.physx
import omni.usd
from isaacsim.core.prims import SingleXFormPrim
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

LidarRtxClass = None
try:
    from isaacsim.sensors.rtx import LidarRtx as LidarRtxClass
except ModuleNotFoundError:
    try:
        from omni.isaac.sensor import LidarRtx as LidarRtxClass
    except ModuleNotFoundError:
        pass


class WarehousePathPlanner:
    """Builds a 2D occupancy grid from static USD scene obstacles and plans smoothed A* paths."""

    def __init__(
        self,
        bounds_min: Tuple[float, float] = (-5.5, -4.5),
        bounds_max: Tuple[float, float] = (3.5, 4.5),
        resolution: float = 0.20,
        robot_radius: float = 0.48,
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

    def build_from_stage(self, stage, pallet_xy: Tuple[float, float]) -> None:
        self.grid.fill(False)
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        obstacle_keywords = ("ur10table", "conveyor", "table", "wall", "rack", "shelf", "pillar", "barrier")
        ignore_keywords = ("floor", "ground", "ceiling", "bin_", "iw_hub", "personnel", "safetyzones", "pallet")

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

        self._mark_rect(-0.55, -0.35, 0.35, 1.85, blocked=True)
        px, py = pallet_xy
        self._mark_rect(px - 0.55, py - 1.55, px + 0.55, py + 0.25, blocked=False)
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

    def _has_line_of_sight(self, p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        steps = max(2, int(math.ceil(dist / (self.res * 0.5))))
        for i in range(steps + 1):
            t = i / steps
            x = p0[0] + (p1[0] - p0[0]) * t
            y = p0[1] + (p1[1] - p0[1]) * t
            gx, gy = self.world_to_grid(x, y)
            if self.grid[gx, gy]:
                return False
        return True

    def _smooth_waypoints(self, start_xy: Tuple[float, float], waypoints: List[np.ndarray]) -> List[np.ndarray]:
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
                ):
                    furthest_idx = candidate_idx
                    break
            smoothed.append(pts[furthest_idx])
            anchor_idx = furthest_idx
        return smoothed

    def plan_path(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]) -> List[np.ndarray]:
        sx, sy = self._nearest_free_cell(*self.world_to_grid(*start_xy))
        gx, gy = self._nearest_free_cell(*self.world_to_grid(*goal_xy))

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
                if self.grid[nx, ny]:
                    continue
                if dx != 0 and dy != 0 and (self.grid[cx + dx, cy] or self.grid[cx, cy + dy]):
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
        return self._smooth_waypoints(start_xy, raw_wps)


class AGVManager:
    """Autonomous idealworks iw.hub controller with smoothed A* navigation,
    zero-VRAM PhysX 2D Raycast Lidar, FixedJoint pallet handling, and dual-contour telemetry.
    """

    STATUS_COLORS = {
        "IDLE": (0.1, 0.85, 0.2),
        "MOVING": (0.1, 0.5, 1.0),
        "DOCKING": (1.0, 0.75, 0.0),
        "LOADED": (0.65, 0.2, 0.95),
        "LIDAR_STOP": (1.0, 0.1, 0.1),
        "NET_BLOCKED": (1.0, 0.0, 0.4),
    }

    def __init__(
        self,
        agv_prim_path: str = "/World/iw_hub_01",
        pallet_prim_path: str = "/World/Ur10Table/pallet",
        robot_id: str = "AGV-01",
        authorized: bool = True,
        standby_pos: Tuple[float, float, float] = (-3.20, 2.20, -1.175),
        dropoff_pos: Tuple[float, float, float] = (-3.20, -1.80, -1.175),
        priority: int = 1,
        wheel_radius: float = 0.08,
        wheel_base: float = 0.58,
    ):
        self.prim_path = agv_prim_path
        self.articulation_path = agv_prim_path
        self.pallet_prim_path = pallet_prim_path
        self.robot_id = robot_id
        self.authorized = authorized
        self.priority = priority
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base

        self.floor_z = float(standby_pos[2])
        self.wait_pos = np.array(standby_pos, dtype=float)
        self.pallet_pos = np.array([0.688, 0.575, self.floor_z], dtype=float)
        self.dropoff_pos = np.array(dropoff_pos, dtype=float)
        self.approach_pos = np.array([0.688, -0.75, self.floor_z], dtype=float)

        self.current_pos = self.wait_pos.copy()
        self.current_yaw = 0.0

        self.state = "IDLE"
        self.task_flag = "STANDBY"
        self.max_linear_speed = 1.10
        self.dock_linear_speed = 0.35
        self.max_angular_speed = 1.55

        self.lift_height = 0.0
        self.max_lift_height = 0.085
        self.lift_speed = 0.06

        self.planner: Optional[WarehousePathPlanner] = None
        self._active_waypoints: List[np.ndarray] = []

        self.lidar = None
        self.use_physx_lidar = True
        self.lidar_max_range = 3.5
        self.lidar_angles = np.linspace(-math.radians(60.0), math.radians(60.0), 41)
        self.last_lidar_scan: List[Tuple[float, float]] = []
        self.lidar_stop_distance = 0.85
        self.lidar_obstacle_detected = False
        self.min_front_obstacle_dist = 99.0

        self.pallet_joint_path = f"{self.prim_path}/PalletAttachFixedJoint"
        self._pallet_attached = False
        self.pallet_base_z: Optional[float] = None
        self.pallet_init_pos = np.array([0.688, 0.575, -0.82], dtype=float)
        self.pallet_init_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        self.wait_timer = 0.0
        self.wait_duration = 1.2
        self.external_safety_halt = False
        self._badge_paths: List[str] = []
        self._beacon_paths: List[str] = []

    def setup(self, world=None, planner: Optional[WarehousePathPlanner] = None) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        agv_prim = stage.GetPrimAtPath(self.prim_path)
        if not agv_prim.IsValid():
            print(f"[AGV ERROR] Prim not found at {self.prim_path}")
            return

        self._ensure_meshes_visible(agv_prim)
        self.articulation_path = self._find_articulation_root(agv_prim)
        self.pallet_joint_path = f"{self.prim_path}/PalletAttachFixedJoint"
        self._resolve_pallet_prim(stage)
        self._prepare_pallet_rigid_body(stage)

        self._sync_pose_from_stage()
        self.floor_z = float(self.current_pos[2])
        self.wait_pos = self.current_pos.copy()
        self.pallet_pos[2] = self.floor_z
        self.dropoff_pos[2] = self.floor_z
        self.approach_pos = np.array([self.pallet_pos[0], self.pallet_pos[1] - 1.30, self.floor_z], dtype=float)

        if planner is not None:
            self.planner = planner
        else:
            self.planner = WarehousePathPlanner()
            self.planner.build_from_stage(stage, (float(self.pallet_pos[0]), float(self.pallet_pos[1])))

        self._ensure_collisions_enabled(agv_prim)
        self._setup_rtx_lidar(stage)
        self._build_visual_markers(stage)
        self._set_task_flag("STANDBY", "IDLE")
        print(f"[AGV] {self.robot_id} ready at {self.prim_path} (authorized={self.authorized}, PhysX-Lidar=ON).")

    def _ensure_meshes_visible(self, root_prim) -> None:
        """Load native idealworks iw.hub payloads and preserve original materials and top shell."""
        root_prim.Load()
        if root_prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(root_prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)

        mesh_count = 0
        for prim in Usd.PrimRange(root_prim):
            if prim.HasPayload():
                prim.Load()
            if prim.IsA(UsdGeom.Imageable):
                img = UsdGeom.Imageable(prim)
                vis = img.GetVisibilityAttr()
                if vis and vis.Get() == UsdGeom.Tokens.invisible:
                    vis.Set(UsdGeom.Tokens.inherited)
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1

        print(f"[AGV MESH] {self.robot_id}: loaded {mesh_count} native iw.hub meshes.")

    def _build_visual_markers(self, stage) -> None:
        """Remove any legacy proxy chassis primitives and attach flush ID badges & status LEDs."""
        legacy_parts = (
            "Chassis_Lower", "Chassis_UpperDeck", "Lift_Platform",
            "Front_Lidar_Hood", "Front_Lidar_Lens",
            "Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR",
            "Skirt_Center", "Skirt_Front", "Skirt_Rear",
            "Body_Center", "Body_Front", "Body_Rear",
            "Collar_Center", "Collar_Front", "Collar_Rear",
            "Lift_Table", "Lidar_Window_Front", "Lidar_Beam_Front", "Lidar_Window_Rear",
            "Wheel_Drive_L", "Wheel_Drive_R",
            "Wheel_Caster_FL", "Wheel_Caster_FR", "Wheel_Caster_RL", "Wheel_Caster_RR",
            "Badge_Front", "Badge_Rear",
        )
        for part in legacy_parts:
            p = f"{self.prim_path}/{part}"
            if stage.GetPrimAtPath(p).IsValid():
                stage.RemovePrim(p)

        badge_specs = [
            ("Badge_Left", Gf.Vec3d(0.0, 0.345, 0.14), Gf.Vec3f(0.18, 0.006, 0.045)),
            ("Badge_Right", Gf.Vec3d(0.0, -0.345, 0.14), Gf.Vec3f(0.18, 0.006, 0.045)),
        ]
        self._badge_paths.clear()
        badge_rgb = (0.95, 0.95, 0.95) if self.robot_id == "AGV-01" else (0.10, 0.80, 0.95)
        if not self.authorized:
            badge_rgb = (0.2, 0.2, 0.2)

        for name, offset, scale in badge_specs:
            path = f"{self.prim_path}/{name}"
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(cube.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(offset)
            xf.AddScaleOp().Set(scale)
            cube.GetPrim().CreateAttribute("agv:robot_id", Sdf.ValueTypeNames.String).Set(self.robot_id)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*badge_rgb)])
            self._badge_paths.append(path)

        self._beacon_paths = [
            f"{self.prim_path}/StatusLED_Left",
            f"{self.prim_path}/StatusLED_Right",
        ]
        for led_path, offset in [
            (self._beacon_paths[0], Gf.Vec3d(0.0, 0.347, 0.095)),
            (self._beacon_paths[1], Gf.Vec3d(0.0, -0.347, 0.095)),
        ]:
            strip = UsdGeom.Cube.Define(stage, led_path)
            strip.CreateSizeAttr(1.0)
            s_xf = UsdGeom.Xformable(strip.GetPrim())
            s_xf.ClearXformOpOrder()
            s_xf.AddTranslateOp().Set(offset)
            s_xf.AddScaleOp().Set(Gf.Vec3f(0.42, 0.006, 0.015))
            strip.GetPrim().CreateAttribute("agv:status_flag", Sdf.ValueTypeNames.String).Set(self.task_flag)

    def _set_task_flag(self, flag: str, visual_key: Optional[str] = None) -> None:
        self.task_flag = flag
        if visual_key is not None:
            self._update_beacon_color(visual_key)
        else:
            stage = omni.usd.get_context().get_stage()
            if stage:
                for led_path in self._beacon_paths:
                    prim = stage.GetPrimAtPath(led_path)
                    if prim.IsValid():
                        attr = prim.GetAttribute("agv:status_flag")
                        if attr:
                            attr.Set(self.task_flag)

    def _find_articulation_root(self, root_prim) -> str:
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return str(prim.GetPath())
        return str(root_prim.GetPath())

    def _resolve_pallet_prim(self, stage) -> None:
        for path in (self.pallet_prim_path, "/World/Ur10Table/pallet", "/World/Ur10Table/Pallet", "/World/Pallet"):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                self.pallet_prim_path = path
                tf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                trans = tf.ExtractTranslation()
                rot = tf.ExtractRotationQuat()
                self.pallet_base_z = float(trans[2])
                self.pallet_init_pos = np.array([float(trans[0]), float(trans[1]), float(trans[2])], dtype=float)
                self.pallet_init_quat = np.array(
                    [
                        float(rot.GetReal()),
                        float(rot.GetImaginary()[0]),
                        float(rot.GetImaginary()[1]),
                        float(rot.GetImaginary()[2]),
                    ],
                    dtype=float,
                )
                self.pallet_pos[0] = float(trans[0])
                self.pallet_pos[1] = float(trans[1])
                return

    def restore_pallet_nominal_pose(self) -> None:
        """Snap pallet back to its exact initial position and orientation on the blue cart."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        pallet_prim = stage.GetPrimAtPath(self.pallet_prim_path)
        if not pallet_prim.IsValid():
            return
        self._set_pallet_kinematic(stage, is_kinematic=True)
        SingleXFormPrim(self.pallet_prim_path).set_world_pose(
            position=self.pallet_init_pos,
            orientation=self.pallet_init_quat,
        )

    def _find_rigid_body_prim(self, root_prim, preferred_keywords=("lift", "top_plate", "chassis", "base")):
        candidates = [p for p in Usd.PrimRange(root_prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        for kw in preferred_keywords:
            for prim in candidates:
                if kw in prim.GetName().lower():
                    return prim
        return candidates[0] if candidates else None

    def _prepare_pallet_rigid_body(self, stage) -> None:
        pallet_prim = stage.GetPrimAtPath(self.pallet_prim_path)
        if not pallet_prim.IsValid():
            return

        for prim in Usd.PrimRange(pallet_prim):
            if prim.IsA(UsdGeom.Mesh):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(prim)
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_col.CreateApproximationAttr().Set("convexHull")

        if not pallet_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI.Apply(pallet_prim)
            rb.CreateKinematicEnabledAttr().Set(True)
        else:
            UsdPhysics.RigidBodyAPI(pallet_prim).GetKinematicEnabledAttr().Set(True)

        if not pallet_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(pallet_prim)

    def _set_pallet_kinematic(self, stage, is_kinematic: bool) -> None:
        pallet_prim = stage.GetPrimAtPath(self.pallet_prim_path)
        if pallet_prim.IsValid() and pallet_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(pallet_prim).GetKinematicEnabledAttr().Set(bool(is_kinematic))

    def _ensure_collisions_enabled(self, root_prim) -> None:
        pass

    def _setup_rtx_lidar(self, stage) -> None:
        """Configure zero-VRAM PhysX 2D planar raycast lidar."""
        self.lidar = None
        self.use_physx_lidar = True
        self.lidar_max_range = 3.5
        self.lidar_angles = np.linspace(-math.radians(60.0), math.radians(60.0), 41)
        self.last_lidar_scan = []

    def _update_beacon_color(self, visual_key: str) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        rgb = self.STATUS_COLORS.get(visual_key, (0.5, 0.5, 0.5))
        for led_path in self._beacon_paths:
            prim = stage.GetPrimAtPath(led_path)
            if prim.IsValid():
                UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
                attr = prim.GetAttribute("agv:status_flag")
                if attr:
                    attr.Set(self.task_flag)

    def _poll_lidar_obstacles(self, peer_positions: Optional[List[Tuple[str, np.ndarray, int]]] = None) -> bool:
        blocked = False
        min_dist = 99.0
        self.last_lidar_scan.clear()

        if self.use_physx_lidar:
            try:
                sq = omni.physx.get_physx_scene_query_interface()
                sensor_z = float(self.current_pos[2]) + 0.16
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
                        hit_prim = str(hit.get("rigidBody", "") or hit.get("collision", ""))
                        if self.prim_path in hit_prim or "/pallet" in hit_prim.lower():
                            continue
                        dist = float(hit["distance"])
                        self.last_lidar_scan.append((float(az), dist))
                        if abs(az) <= math.radians(32.0) and dist < min_dist:
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
                if 1e-3 < dist < 1.25:
                    forward_proj = float(np.dot(rel / dist, heading))
                    if forward_proj > 0.45 and peer_prio <= self.priority:
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
        agv_prim = stage.GetPrimAtPath(self.prim_path)
        pallet_prim = stage.GetPrimAtPath(self.pallet_prim_path)
        if not agv_prim.IsValid() or not pallet_prim.IsValid():
            return

        body0_prim = self._find_rigid_body_prim(agv_prim)
        if body0_prim is None:
            return

        self._prepare_pallet_rigid_body(stage)
        self._set_pallet_kinematic(stage, is_kinematic=False)

        xform_cache = UsdGeom.XformCache()
        tf0 = xform_cache.GetLocalToWorldTransform(body0_prim)
        tf1 = xform_cache.GetLocalToWorldTransform(pallet_prim)
        rel_tf = tf1 * tf0.GetInverse()
        rel_pos = rel_tf.ExtractTranslation()
        rel_rot = rel_tf.ExtractRotationQuat()

        joint = UsdPhysics.FixedJoint.Define(stage, self.pallet_joint_path)
        joint.CreateBody0Rel().SetTargets([body0_prim.GetPath()])
        joint.CreateBody1Rel().SetTargets([pallet_prim.GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(rel_pos[0]), float(rel_pos[1]), float(rel_pos[2])))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(rel_rot.GetReal()),
                float(rel_rot.GetImaginary()[0]),
                float(rel_rot.GetImaginary()[1]),
                float(rel_rot.GetImaginary()[2]),
            )
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        self._pallet_attached = True
        print(f"[AGV] {self.robot_id} FixedJoint engaged on {pallet_prim.GetPath()}.")

    def _detach_pallet_joint(self) -> None:
        if not self._pallet_attached:
            return
        stage = omni.usd.get_context().get_stage()
        if stage:
            if stage.GetPrimAtPath(self.pallet_joint_path).IsValid():
                stage.RemovePrim(self.pallet_joint_path)
        self._pallet_attached = False
        self.restore_pallet_nominal_pose()
        print(f"[AGV] {self.robot_id} FixedJoint released: pallet seated at nominal pose.")

    def _plan_route_to(self, goal_pos: np.ndarray) -> None:
        self._sync_pose_from_stage()
        if self.planner is not None:
            raw_wps = self.planner.plan_path(
                (float(self.current_pos[0]), float(self.current_pos[1])),
                (float(goal_pos[0]), float(goal_pos[1])),
            )
            self._active_waypoints = [np.array([wp[0], wp[1], self.floor_z], dtype=float) for wp in raw_wps]
        else:
            self._active_waypoints = [goal_pos.copy()]

    def trigger_pallet_swap(self) -> None:
        if self.state == "IDLE":
            self._plan_route_to(self.approach_pos)
            self.state = "APPROACHING_PALLET"
            self._set_task_flag("FETCH_FULL_PALLET", "MOVING")
            print(f"[LOGISTICS] {self.robot_id} planned smoothed A* route ({len(self._active_waypoints)} wps) to pallet.")

    def trigger_patrol_intrusion(self, target_xy: Tuple[float, float]) -> None:
        goal = np.array([target_xy[0], target_xy[1], self.floor_z], dtype=float)
        self._plan_route_to(goal)
        self.state = "TRANSIT_PATROL"
        self._set_task_flag("UNSCHEDULED_TRANSIT", "MOVING")

    def set_external_safety_halt(self, halt: bool) -> None:
        self.external_safety_halt = halt
        if halt:
            self._stop_wheels()
            self._update_beacon_color("NET_BLOCKED")

    def get_network_telemetry(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "authorized": self.authorized,
            "task_flag": self.task_flag,
            "state": self.state,
            "net_pos": (float(self.current_pos[0]), float(self.current_pos[1])),
            "lidar_blocked": self.lidar_obstacle_detected,
            "lidar_min_dist": round(float(self.min_front_obstacle_dist), 2),
        }

    def _sync_pose_from_stage(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return

        tf = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        trans = tf.ExtractTranslation()
        rot = tf.ExtractRotationQuat()
        self.current_pos = np.array([float(trans[0]), float(trans[1]), float(trans[2])], dtype=float)
        w = float(rot.GetReal())
        x, y, z = float(rot.GetImaginary()[0]), float(rot.GetImaginary()[1]), float(rot.GetImaginary()[2])
        self.current_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _apply_wheel_velocities(self, linear_vel: float, angular_vel: float) -> None:
        dt = 1.0 / 60.0
        self.current_yaw = math.atan2(
            math.sin(self.current_yaw + angular_vel * dt),
            math.cos(self.current_yaw + angular_vel * dt),
        )
        heading = np.array([math.cos(self.current_yaw), math.sin(self.current_yaw), 0.0])
        self.current_pos += heading * (linear_vel * dt)
        self.current_pos[2] = self.floor_z

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.prim_path) if stage else None
        if prim and prim.IsValid():
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*self.current_pos.tolist()))
            xf.AddRotateZOp().Set(math.degrees(self.current_yaw))

    def _stop_wheels(self) -> None:
        pass

    def _drive_towards(self, target: np.ndarray, cruise_speed: float, reverse: bool = False) -> bool:
        diff = target[:2] - self.current_pos[:2]
        distance = float(np.linalg.norm(diff))

        if distance < 0.08:
            return True

        desired_yaw = math.atan2(float(diff[1]), float(diff[0]))
        if reverse:
            desired_yaw = math.atan2(math.sin(desired_yaw + math.pi), math.cos(desired_yaw + math.pi))

        yaw_error = math.atan2(
            math.sin(desired_yaw - self.current_yaw),
            math.cos(desired_yaw - self.current_yaw),
        )

        if abs(yaw_error) > math.radians(55.0):
            ang_vel = float(np.clip(1.6 * yaw_error, -self.max_angular_speed, self.max_angular_speed))
            self._apply_wheel_velocities(0.0, ang_vel)
            return False

        turn_factor = max(0.25, math.cos(yaw_error))
        lin_vel = min(cruise_speed, max(0.16, distance * 1.3)) * turn_factor
        if reverse:
            lin_vel = -lin_vel
        ang_vel = float(np.clip(1.35 * yaw_error, -self.max_angular_speed, self.max_angular_speed))
        self._apply_wheel_velocities(lin_vel, ang_vel)
        return False

    def _follow_waypoints(self, cruise_speed: float) -> bool:
        if not self._active_waypoints:
            return True
        if self._drive_towards(self._active_waypoints[0], cruise_speed):
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

        if not self._pallet_attached:
            pallet_prim = stage.GetPrimAtPath(self.pallet_prim_path)
            if pallet_prim.IsValid() and self.pallet_base_z is not None:
                p_xf = UsdGeom.Xformable(pallet_prim)
                for op in p_xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        cur = op.Get()
                        op.Set(Gf.Vec3d(cur[0], cur[1], float(self.pallet_base_z + height_m)))
                        break

    def update(
        self,
        dt: float,
        spawner_reset_callback=None,
        peer_positions: Optional[List[Tuple[str, np.ndarray, int]]] = None,
    ) -> None:
        if self.state == "IDLE":
            return

        if self.external_safety_halt:
            self._stop_wheels()
            return

        obstacle_check_states = {
            "APPROACHING_PALLET",
            "TRANSPORT_TO_DROPOFF",
            "RETURNING_WITH_EMPTY",
            "RETURNING_STANDBY",
            "TRANSIT_PATROL",
        }
        if self.state in obstacle_check_states and self._poll_lidar_obstacles(peer_positions):
            self._stop_wheels()
            self._update_beacon_color("LIDAR_STOP")
            return

        if self.state == "TRANSIT_PATROL":
            self._update_beacon_color("MOVING")
            if self._follow_waypoints(self.max_linear_speed):
                self.state = "IDLE"
                self._set_task_flag("STANDBY", "IDLE")

        elif self.state == "APPROACHING_PALLET":
            self._update_beacon_color("MOVING")
            if self._follow_waypoints(self.max_linear_speed):
                self.state = "DOCKING"
                self._set_task_flag("DOCKING_UNDER_PALLET", "DOCKING")

        elif self.state == "DOCKING":
            if self._drive_towards(self.pallet_pos, self.dock_linear_speed):
                self.state = "LIFTING"
                self._set_task_flag("LIFTING_PALLET", "DOCKING")

        elif self.state == "LIFTING":
            self.lift_height = min(self.max_lift_height, self.lift_height + self.lift_speed * dt)
            self._set_lift_height(self.lift_height)
            if self.lift_height >= self.max_lift_height - 1e-4:
                self._attach_pallet_joint()
                self._plan_route_to(self.dropoff_pos)
                self.state = "TRANSPORT_TO_DROPOFF"
                self._set_task_flag("HAULING_LOADED_PALLET", "LOADED")
                print(f"[LOGISTICS] {self.robot_id} hauling full pallet via A* route.")

        elif self.state == "TRANSPORT_TO_DROPOFF":
            if self._follow_waypoints(self.max_linear_speed):
                self.state = "UNLOADING_AT_DROPOFF"
                self._set_task_flag("SWAPPING_EMPTY_PALLET", "DOCKING")
                self.wait_timer = 0.0

        elif self.state == "UNLOADING_AT_DROPOFF":
            self.wait_timer += dt
            if self.wait_timer >= self.wait_duration:
                self._plan_route_to(self.approach_pos)
                self.state = "RETURNING_WITH_EMPTY"
                self._set_task_flag("RETURNING_EMPTY_PALLET", "MOVING")
                print(f"[LOGISTICS] {self.robot_id} returning with empty pallet.")

        elif self.state == "RETURNING_WITH_EMPTY":
            if self._follow_waypoints(self.max_linear_speed):
                self.state = "DOCKING_EMPTY"
                self._set_task_flag("PLACING_EMPTY_PALLET", "DOCKING")

        elif self.state == "DOCKING_EMPTY":
            if self._drive_towards(self.pallet_pos, self.dock_linear_speed):
                self.state = "LOWERING"
                self._set_task_flag("LOWERING_PALLET", "DOCKING")

        elif self.state == "LOWERING":
            self._detach_pallet_joint()
            self.lift_height = max(0.0, self.lift_height - self.lift_speed * dt)
            self._set_lift_height(self.lift_height)
            if self.lift_height <= 1e-4:
                if spawner_reset_callback:
                    spawner_reset_callback()
                self.state = "UNDOCKING"
                self._set_task_flag("UNDOCKING", "DOCKING")

        elif self.state == "UNDOCKING":
            if self._drive_towards(self.approach_pos, self.dock_linear_speed, reverse=True):
                self._plan_route_to(self.wait_pos)
                self.state = "RETURNING_STANDBY"
                self._set_task_flag("RETURNING_STANDBY", "MOVING")

        elif self.state == "RETURNING_STANDBY":
            if self._follow_waypoints(self.max_linear_speed):
                self.state = "IDLE"
                self._set_task_flag("STANDBY", "IDLE")
                print(f"[LOGISTICS] {self.robot_id} returned to standby bay.")


class AGVFleetManager:
    """Coordinates multiple iw.hub rovers, alternates cycle dispatch, and collects fleet telemetry."""

    def __init__(self, rovers: List[AGVManager]):
        self.rovers = rovers
        self._authorized_rovers = [r for r in rovers if r.authorized]
        self._next_dispatch_idx = 0

    def setup(self, world=None) -> None:
        stage = omni.usd.get_context().get_stage()
        shared_planner = WarehousePathPlanner()
        if stage:
            shared_planner.build_from_stage(stage, (0.688, 0.575))
        for rover in self.rovers:
            rover.setup(world=world, planner=shared_planner)

    def has_active_service(self) -> bool:
        return any(r.authorized and r.state != "IDLE" for r in self.rovers)

    def dispatch_next_available(self) -> Optional[AGVManager]:
        if not self._authorized_rovers:
            return None
        for offset in range(len(self._authorized_rovers)):
            idx = (self._next_dispatch_idx + offset) % len(self._authorized_rovers)
            candidate = self._authorized_rovers[idx]
            if candidate.state == "IDLE" and not candidate.external_safety_halt:
                self._next_dispatch_idx = (idx + 1) % len(self._authorized_rovers)
                candidate.trigger_pallet_swap()
                return candidate
        return None

    def update(self, dt: float, spawner_reset_callback=None) -> None:
        peer_snapshot = [(r.robot_id, r.current_pos.copy(), r.priority) for r in self.rovers]
        for rover in self.rovers:
            rover.update(dt, spawner_reset_callback=spawner_reset_callback, peer_positions=peer_snapshot)

    def get_all_telemetry(self) -> List[dict]:
        return [r.get_network_telemetry() for r in self.rovers]

    def apply_security_halts(self, blocked_ids: set) -> None:
        for rover in self.rovers:
            rover.set_external_safety_halt(rover.robot_id in blocked_ids)