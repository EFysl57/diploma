import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom


class LightweightUsdCamera:
    """Lightweight UsdGeom.Camera wrapper that preserves manual viewport transforms
    if the camera prim already exists in the stage.
    """

    def __init__(
        self,
        camera_prim_path: str,
        position: Tuple[float, float, float],
        orientation_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        focal_length: float = 12.0,
        focus_distance: Optional[float] = None,
        fstop: Optional[float] = None,
    ):
        self.prim_path = camera_prim_path
        self.position = np.array(position, dtype=float)
        self.orientation_euler_deg = orientation_euler_deg
        self.scale = scale
        self.focal_length = focal_length
        self.focus_distance = focus_distance
        self.fstop = fstop

    def setup(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            print(f"[CAMERA ERROR] No active USD stage for {self.prim_path}")
            return

        cam_prim = stage.GetPrimAtPath(self.prim_path)
        if not cam_prim.IsValid():
            usd_cam = UsdGeom.Camera.Define(stage, self.prim_path)
            cam_prim = usd_cam.GetPrim()

            xf = UsdGeom.Xformable(cam_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*self.position.tolist()))
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*self.orientation_euler_deg))
            xf.AddScaleOp().Set(Gf.Vec3f(*self.scale))

            usd_cam.CreateFocalLengthAttr(float(self.focal_length))
            if self.focus_distance is not None:
                usd_cam.CreateFocusDistanceAttr(float(self.focus_distance))
            if self.fstop is not None:
                usd_cam.CreateFStopAttr(float(self.fstop))
            usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
            print(f"[CAMERA] Created lightweight UsdGeom.Camera: {self.prim_path}")
        else:
            # Keep user's manual position/rotation/optics intact and sync live world position
            w_pos = UsdGeom.XformCache().GetLocalToWorldTransform(cam_prim).ExtractTranslation()
            self.position = np.array([float(w_pos[0]), float(w_pos[1]), float(w_pos[2])], dtype=float)
            print(f"[CAMERA] Using existing UsdGeom.Camera transform: {self.prim_path}")


class DualContourCameraSystem:
    """Manages lightweight Overhead and Side UsdGeom cameras and reads ground-truth
    CV detections directly from the USD stage without GPU render-target overhead.
    """

    def __init__(
        self,
        overhead_path: str = "/World/SafetyCamera_Overhead",
        side_path: str = "/World/SafetyCamera_Side",
    ):
        self.overhead_cam = LightweightUsdCamera(
            camera_prim_path=overhead_path,
            position=(0.35, 0.45, 6.20),
            orientation_euler_deg=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            focal_length=11.0,
        )
        self.side_cam = LightweightUsdCamera(
            camera_prim_path=side_path,
            position=(-6.97198, -8.05845, 4.20584),
            orientation_euler_deg=(66.38284, 0.0, -35.64631),
            scale=(1.0, 1.0, 1.0),
            focal_length=24.0,
            focus_distance=0.9,
            fstop=0.0,
        )

    def setup(self) -> None:
        self.overhead_cam.setup()
        self.side_cam.setup()

    def capture_frames(self) -> Dict[str, Optional[np.ndarray]]:
        return {"overhead": None, "side": None}

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
                id_attr = prim.GetAttribute("agv:robot_id")
                flag_attr = prim.GetAttribute("agv:status_flag")
                plate_prim = stage.GetPrimAtPath(f"{prim.GetPath()}/NumberPlate_Left")

                has_plate = plate_prim.IsValid()
                target_prim = plate_prim if has_plate else prim
                trans = xform_cache.GetLocalToWorldTransform(target_prim).ExtractTranslation()

                cv_id = str(id_attr.Get()) if (has_plate and id_attr) else "UNKNOWN"
                cv_flag = str(flag_attr.Get()) if (has_plate and flag_attr) else "UNKNOWN"

                dist_to_side_cam = math.hypot(
                    float(trans[0]) - float(self.side_cam.position[0]),
                    float(trans[1]) - float(self.side_cam.position[1]),
                )
                side_verified = has_plate and (dist_to_side_cam <= 12.5)

                detections.append(
                    {
                        "class": "agv",
                        "cv_id": cv_id,
                        "cv_flag": cv_flag,
                        "unmarked_plate": not has_plate,
                        "pos": (float(trans[0]), float(trans[1])),
                        "side_cam_verified": side_verified,
                        "source": "dual_cam" if side_verified else "overhead_cam",
                    }
                )

        return detections