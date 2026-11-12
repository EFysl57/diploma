import numpy as np
import omni.usd
from pxr import Gf, UsdGeom


class AGVManager:
    def __init__(self, agv_prim_path="/agv"):
        self.prim_path = agv_prim_path
        # Координаты точек (x, y, z)
        self.wait_pos = np.array([-3.419, 2.295, -0.95])  # Начальная позиция ожидания
        self.pallet_pos = np.array([0.688, 0.575, -0.95]) # Позиция у паллета
        
        self.current_pos = self.wait_pos.copy()
        self.state = "IDLE"  # Возможные состояния: IDLE, MOVING_TO_PALLET, WAITING_UNLOAD, RETURNING
        self.speed = 1.2     # Скорость движения (м/с)
        self.wait_timer = 0.0
        self.wait_duration = 2.0  # Пауза у паллета (сек)

    def trigger_pallet_swap(self):
        """Запуск цикла смены паллета."""
        if self.state == "IDLE":
            self.state = "MOVING_TO_PALLET"
            print("[LOGISTICS] AGV получена команда: выезд за паллетом!")

    def _set_world_pose(self, pos):
        """Прямая установка позиций прима в сцену USD."""
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return

        prim = stage.GetPrimAtPath(self.prim_path)
        # Проверяем запасной путь на случай если перенесли в /World/AGV
        if not prim.IsValid():
            prim = stage.GetPrimAtPath("/World/AGV")
        if not prim.IsValid():
            prim = stage.GetPrimAtPath("/World/agv")

        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            transform_attr = xform.GetXformOpOrderAttr()
            
            # Поиск или создание операции перевода Translate
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            
            if not translate_op:
                translate_op = xform.AddTranslateOp()

            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def update(self, dt: float, spawner_reset_callback=None):
        if self.state == "IDLE":
            return

        # 1. Движение к паллету
        if self.state == "MOVING_TO_PALLET":
            direction = self.pallet_pos - self.current_pos
            distance = np.linalg.norm(direction)

            if distance < 0.05:
                self.current_pos = self.pallet_pos.copy()
                self._set_world_pose(self.current_pos)
                self.state = "WAITING_UNLOAD"
                self.wait_timer = 0.0
                print("[LOGISTICS] AGV прибыл к паллету. Очистка поддона...")
            else:
                step_move = (direction / distance) * self.speed * dt
                self.current_pos += step_move
                self._set_world_pose(self.current_pos)

        # 2. Пауза и сброс паллета
        elif self.state == "WAITING_UNLOAD":
            self.wait_timer += dt
            if self.wait_timer >= self.wait_duration:
                if spawner_reset_callback:
                    spawner_reset_callback()
                self.state = "RETURNING"
                print("[LOGISTICS] Новый паллет установлен. AGV возвращается...")

        # 3. Возврат в зону ожидания
        elif self.state == "RETURNING":
            direction = self.wait_pos - self.current_pos
            distance = np.linalg.norm(direction)

            if distance < 0.05:
                self.current_pos = self.wait_pos.copy()
                self._set_world_pose(self.current_pos)
                self.state = "IDLE"
                print("[LOGISTICS] AGV вернулся в базу. Готов к работе!")
            else:
                step_move = (direction / distance) * self.speed * dt
                self.current_pos += step_move
                self._set_world_pose(self.current_pos)