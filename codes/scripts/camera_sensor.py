import numpy as np
import omni.usd

CameraSensorClass = None
try:
    from isaacsim.sensors.camera import Camera as CameraSensorClass
except ModuleNotFoundError:
    try:
        from omni.isaac.sensor import Camera as CameraSensorClass
    except ModuleNotFoundError:
        pass


class SafetyCamera:
    def __init__(self, camera_prim_path="/World/SafetyCamera"):
        self.prim_path = camera_prim_path
        self.sensor = None

    def setup(self):
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(self.prim_path)
        
        if CameraSensorClass is not None:
            if camera_prim.IsValid():
                self.sensor = CameraSensorClass(prim_path=self.prim_path)
                print(f"[CAMERA] Подключились к камере: {self.prim_path}")
            else:
                cam_position = np.array([0.68868, 0.5758, 6.38909])
                self.sensor = CameraSensorClass(
                    prim_path=self.prim_path, 
                    position=cam_position, 
                    resolution=(640, 480)
                )
                if hasattr(self.sensor, "set_focal_length"):
                    self.sensor.set_focal_length(10.9)
                print(f"[CAMERA] Создали новую камеру: {self.prim_path}")
                
            self.sensor.initialize()
            if hasattr(self.sensor, "add_motion_vectors_to_frame"):
                self.sensor.add_motion_vectors_to_frame()
        else:
            print("[CAMERA ERROR] Класс CameraSensorClass не найден!")

    def get_rgb_frame(self):
        if self.sensor is not None and hasattr(self.sensor, "get_rgba"):
            rgba = self.sensor.get_rgba()
            if rgba is not None and rgba.size > 0:
                return rgba[:, :, :3]
        return None