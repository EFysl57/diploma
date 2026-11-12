import math
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics


class AGVManager:
    def __init__(
        self,
        agv_prim_path: str = "/agv",
        pallet_prim_path: str = "/World/Pallet",
    ):
        self.prim_path = agv_prim_path
        self.pallet_prim_path = pallet_prim_path

        # Waypoints (x, y, z) at floor level
        self.wait_pos = np.array([-3.419, 2.295, -0.95], dtype=float)
        self.pallet_pos = np.array([0.688, 0.575, -0.95], dtype=float)

        # Compute pre-docking staging point 1.2m in front of the pallet
        approach_vec = self.pallet_pos[:2] - self.wait_pos[:2]
        norm = np.linalg.norm(approach_vec)
        dir_2d = approach_vec / norm if norm > 1e-5 else np.array([1.0, 0.0])
        self.approach_pos = np.array([
            self.pallet_pos[0] - dir_2d[0] * 1.2,
            self.pallet_pos[1] - dir_2d[1] * 1.2,
            self.pallet_pos[2],
        ], dtype=float)

        self.current_pos = self.wait_pos.copy()
        self.current_yaw = math.degrees(math.atan2(dir_2d[1], dir_2d[0]))

        # State machine:
        # IDLE -> APPROACHING_PALLET -> DOCKING -> LIFTING -> UNLOADING -> LOWERING -> UNDOCKING -> RETURNING
        self.state = "IDLE"
        self.cruise_speed = 1.2   # m/s
        self.dock_speed = 0.45    # m/s slow precision docking under pallet

        # Lift parameters for iw.hub (meters)
        self.lift_height = 0.0
        self.max_lift_height = 0.08  # 8 cm lift stroke
        self.lift_speed = 0.06       # m/s

        self.wait_timer = 0.0
        self.wait_duration = 1.5

        # Cache initial pallet Z if pallet prim exists
        self.pallet_base_z = self._get_prim_z(self.pallet_prim_path)

    def trigger_pallet_swap(self):
        """Start the autonomous pallet pickup and swap cycle."""
        if self.state == "IDLE":
            self.state = "APPROACHING_PALLET"
            print("[LOGISTICS] iw.hub dispatched: approaching pallet station.")

    def _resolve_agv_prim(self, stage):
        for candidate in (self.prim_path, "/World/AGV", "/World/agv", "/World/iw_hub"):
            prim = stage.GetPrimAtPath(candidate)
            if prim.IsValid():
                return prim
        return None

    def _get_prim_z(self, prim_path: str):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return None
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return float(op.Get()[2])
        return 0.0

    def _set_world_pose(self, pos: np.ndarray, yaw_deg: float):
        """Set world translation and Z-rotation of the iw.hub root prim."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        prim = self._resolve_agv_prim(stage)
        if not prim:
            return

        xform = UsdGeom.Xformable(prim)
        translate_op = None
        rotate_op = None

        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                rotate_op = op

        if translate_op is None or rotate_op is None:
            xform.ClearXformOpOrder()
            translate_op = xform.AddTranslateOp()
            rotate_op = xform.AddRotateZOp()

        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        rotate_op.Set(float(yaw_deg))

    def _set_lift_height(self, height_m: float):
        """Actuate iw.hub lift platform and visually raise the pallet if present."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        agv_prim = self._resolve_agv_prim(stage)
        if agv_prim:
            # 1. Drive physics lift_joint if simulation physics is active
            for desc in agv_prim.GetAllDescendants():
                if "lift" in desc.GetName().lower():
                    drive = UsdPhysics.DriveAPI.Get(desc, "linear")
                    if drive:
                        drive.GetTargetPositionAttr().Set(float(height_m))

                    # Also update kinematic mesh offset if present
                    if desc.GetName().lower() in ("lift", "lift_link", "top_plate"):
                        lift_xform = UsdGeom.Xformable(desc)
                        for op in lift_xform.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                cur = op.Get()
                                op.Set(Gf.Vec3d(cur[0], cur[1], float(height_m)))
                                break

        # 2. Elevate pallet prim synchronously during lift phase
        pallet_prim = stage.GetPrimAtPath(self.pallet_prim_path)
        if pallet_prim.IsValid() and self.pallet_base_z is not None:
            p_xform = UsdGeom.Xformable(pallet_prim)
            for op in p_xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    cur = op.Get()
                    op.Set(Gf.Vec3d(cur[0], cur[1], float(self.pallet_base_z + height_m)))
                    break

    def _move_towards(self, target: np.ndarray, speed: float, dt: float, reverse: bool = False) -> bool:
        """Move linearly toward target; returns True once within arrival threshold."""
        direction = target - self.current_pos
        distance = float(np.linalg.norm(direction))

        if distance < 0.03:
            self.current_pos = target.copy()
            self._set_world_pose(self.current_pos, self.current_yaw)
            return True

        step_dir = direction / distance
        heading_vec = -step_dir if reverse else step_dir
        self.current_yaw = math.degrees(math.atan2(float(heading_vec[1]), float(heading_vec[0])))

        step_dist = min(speed * dt, distance)
        self.current_pos += step_dir * step_dist
        self._set_world_pose(self.current_pos, self.current_yaw)
        return False

    def update(self, dt: float, spawner_reset_callback=None):
        if self.state == "IDLE":
            return

        # 1. Fast transit to staging point in front of pallet
        if self.state == "APPROACHING_PALLET":
            if self._move_towards(self.approach_pos, self.cruise_speed, dt):
                self.state = "DOCKING"
                print("[LOGISTICS] iw.hub aligned. Docking under pallet...")

        # 2. Slow precision entry beneath pallet structure
        elif self.state == "DOCKING":
            if self._move_towards(self.pallet_pos, self.dock_speed, dt):
                self.state = "LIFTING"
                print("[LOGISTICS] iw.hub centered under pallet. Raising lift platform...")

        # 3. Engage lift mechanism to pick up pallet
        elif self.state == "LIFTING":
            self.lift_height = min(self.max_lift_height, self.lift_height + self.lift_speed * dt)
            self._set_lift_height(self.lift_height)
            if self.lift_height >= self.max_lift_height - 1e-4:
                self.state = "UNLOADING"
                self.wait_timer = 0.0
                print("[LOGISTICS] Pallet lifted. Swapping loaded pallet for empty one...")

        # 4. Dwell time + trigger box cleanup & empty pallet reset
        elif self.state == "UNLOADING":
            self.wait_timer += dt
            if self.wait_timer >= self.wait_duration:
                if spawner_reset_callback:
                    spawner_reset_callback()
                self.state = "LOWERING"
                print("[LOGISTICS] Empty pallet placed. Lowering lift platform...")

        # 5. Lower lift platform to seat the empty pallet
        elif self.state == "LOWERING":
            self.lift_height = max(0.0, self.lift_height - self.lift_speed * dt)
            self._set_lift_height(self.lift_height)
            if self.lift_height <= 1e-4:
                self.state = "UNDOCKING"
                print("[LOGISTICS] Lift lowered. Backing out from under pallet...")

        # 6. Reverse out from under the pallet to staging point
        elif self.state == "UNDOCKING":
            if self._move_towards(self.approach_pos, self.dock_speed, dt, reverse=True):
                self.state = "RETURNING"
                print("[LOGISTICS] Undocked. Returning to standby zone...")

        # 7. Return to standby position
        elif self.state == "RETURNING":
            if self._move_towards(self.wait_pos, self.cruise_speed, dt):
                self.state = "IDLE"
                print("[LOGISTICS] iw.hub returned to standby zone. Ready.")