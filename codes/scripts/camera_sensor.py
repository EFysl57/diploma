import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom

CameraSensorClass = None
try:
    from isaacsim.sensors.camera import Camera as CameraSensorClass
except ModuleNotFoundError:
    try:
        from omni.isaac.sensor import Camera as CameraSensorClass
    except ModuleNotFoundError:
        pass


class SafetyCamera:
    """Wrapper for an individual Isaac Sim RGB camera sensor."""

    def __init__(
        self,
        camera_prim_path: str,
        position: Tuple[float, float, float],
        orientation_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        focal_length: float = 12.0,
        resolution: Tuple[int, int] = (320, 240),
    ):
        self.prim_path = camera_prim_path
        self.position = np.array(position, dtype=float)
        self.orientation_euler_deg = orientation_euler_deg
        self.focal_length = focal_length
        self.resolution = resolution
        self.sensor = None

    def setup(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage or CameraSensorClass is None:
            print(f"[CAMERA ERROR] Unable to initialize {self.prim_path}")
            return

        camera_prim = stage.GetPrimAtPath(self.prim_path)
        if camera_prim.IsValid():
            self.sensor = CameraSensorClass(prim_path=self.prim_path, resolution=self.resolution)
            print(f"[CAMERA] Connected to existing camera: {self.prim_path}")
        else:
            self.sensor = CameraSensorClass(
                prim_path=self.prim_path,
                position=self.position,
                resolution=self.resolution,
            )
            prim = stage.GetPrimAtPath(self.prim_path)
            if prim.IsValid():
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(*self.position.tolist()))
                xf.AddRotateXYZOp().Set(Gf.Vec3f(*self.orientation_euler_deg))
            print(f"[CAMERA] Created camera: {self.prim_path}")

        if hasattr(self.sensor, "set_focal_length"):
            self.sensor.set_focal_length(self.focal_length)
        self.sensor.initialize()

    def get_rgb_frame(self) -> Optional[np.ndarray]:
        if self.sensor is not None and hasattr(self.sensor, "get_rgba"):
            rgba = self.sensor.get_rgba()
            if rgba is not None and rgba.size > 0:
                return rgba[:, :, :3]
        return None


class DualContourCameraSystem:
    """Manages both the Overhead Safety Camera (zone/personnel monitoring)
    and the Side Entrance Camera (AGV side badge ID & LED beacon verification).
    """

    def __init__(
        self,
        overhead_path: str = "/World/SafetyCamera_Overhead",
        side_path: str = "/World/SafetyCamera_Side",
    ):
        self.overhead_cam = SafetyCamera(
            camera_prim_path=overhead_path,
            position=(0.35, 0.30, 6.20),
            orientation_euler_deg=(0.0, 0.0, 0.0),
            focal_length=11.0,
        )
        self.side_cam = SafetyCamera(
            camera_prim_path=side_path,
            position=(2.35, -1.65, -0.45),
            orientation_euler_deg=(78.0, 0.0, 48.0),
            focal_length=16.0,
        )

    def setup(self) -> None:
        self.overhead_cam.setup()
        self.side_cam.setup()

    def capture_frames(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            "overhead": self.overhead_cam.get_rgb_frame(),
            "side": self.side_cam.get_rgb_frame(),
        }

    def read_visual_detections_from_stage(self) -> List[dict]:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return []

        detections: List[dict] = []
        xform_cache = UsdGeom.XformCache()

        personnel_root = stage.GetPrimAtPath("/World/Personnel")
        if personnel_root.IsValid():
            for child in personnel_root.GetChildren():
                role_attr = child.GetAttribute("safety:role")
                helmet_attr = child.GetAttribute("safety:has_helmet")
                if not role_attr:
                    continue
                trans = xform_cache.GetLocalToWorldTransform(child).ExtractTranslation()
                detections.append(
                    {
                        "class": str(role_attr.Get()),
                        "pos": (float(trans[0]), float(trans[1])),
                        "has_helmet": bool(helmet_attr.Get()) if helmet_attr else True,
                        "source": "overhead_cam",
                    }
                )

        for prim in stage.Traverse():
            if prim.GetName().startswith("iw_hub"):
                badge_prim = stage.GetPrimAtPath(f"{prim.GetPath()}/Badge_Left")
                led_prim = stage.GetPrimAtPath(f"{prim.GetPath()}/StatusLED_Left")
                if not badge_prim.IsValid():
                    continue

                cv_id = str(badge_prim.GetAttribute("agv:robot_id").Get())
                cv_flag = str(led_prim.GetAttribute("agv:status_flag").Get()) if led_prim.IsValid() else "UNKNOWN"
                trans = xform_cache.GetLocalToWorldTransform(badge_prim).ExtractTranslation()

                dist_to_side_cam = math.hypot(
                    float(trans[0]) - float(self.side_cam.position[0]),
                    float(trans[1]) - float(self.side_cam.position[1]),
                )
                side_verified = dist_to_side_cam <= 4.5

                detections.append(
                    {
                        "class": "agv",
                        "cv_id": cv_id,
                        "cv_flag": cv_flag,
                        "pos": (float(trans[0]), float(trans[1])),
                        "side_cam_verified": side_verified,
                        "source": "dual_cam" if side_verified else "overhead_cam",
                    }
                )

        return detections
