# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import traceback
import numpy as np
import omni.usd

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base import BaseSample
from isaacsim.robot_motion.examples.manipulation.ur10_palletizing import (
    BinSpawner,
    BinStackingContext,
    PalletizerController,
    Ur10,
    Ur10Assets,
    configure_conveyor,
)

# Безопасный импорт класса камеры
CameraSensor = None
try:
    from isaacsim.sensors.camera import Camera as CameraSensor
except ModuleNotFoundError:
    try:
        from omni.isaac.sensor import Camera as CameraSensor
    except ModuleNotFoundError:
        pass

USD_SCENE_PATH = r"D:\дипломмммммкаааааааааааааааааааааааааааааааааааааАААААААААААААААААААААААААААААААААААААААААА\scene2.usd"


class Palletizing(BaseSample):
    """Класс логики паллетизирования с интегрированной виртуальной камерой."""

    def __init__(self) -> None:
        super().__init__()
        self._env_path = "/World/Ur10Table"
        self._assets: Ur10Assets | None = None
        self._robot: Ur10 | None = None
        self._spawner: BinSpawner | None = None
        self._context: BinStackingContext | None = None
        self._controller: PalletizerController | None = None
        self._physics_callback_id: int | None = None
        self._is_executing = False

        # Камера системы безопасности
        self._safety_camera = None
        self._camera_prim_path = "/World/SafetyCamera"
        self._frame_counter = 0

    async def load_saved_scene(self) -> None:
        """Загрузка сцены из USD-файла с ожиданиями."""
        if not os.path.exists(USD_SCENE_PATH):
            raise FileNotFoundError(f"Файл сцены не найден по пути: {USD_SCENE_PATH}")

        print(f"[INFO] Открываем сцену: {USD_SCENE_PATH}...")
        await omni.usd.get_context().open_stage_async(USD_SCENE_PATH)

        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

        stage = omni.usd.get_context().get_stage()
        robot_prim_path = f"{self._env_path}/ur10"

        if not stage.GetPrimAtPath(robot_prim_path).IsValid():
            print(f"[WARNING] Прим {robot_prim_path} не найден. Проверяем /World/ur10...")
            if stage.GetPrimAtPath("/World/ur10").IsValid():
                robot_prim_path = "/World/ur10"
                self._env_path = "/World"

        self._assets = Ur10Assets()
        self._robot = Ur10(prim_path=robot_prim_path)
        print(f"[INFO] Робот успешно инициализирован по пути: {robot_prim_path}")

    def _setup_safety_camera(self) -> None:
        """Подключение к существующей камере или создание с точными параметрами с потолка."""
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(self._camera_prim_path)

        # Координаты и оптика со скриншота
        cam_position = np.array([0.68868, 0.5758, 6.38909])
        focal_length = 10.9

        if CameraSensor is not None:
            # 1. Если камера уже выстроена вручную на сцене — просто привязываем обертку сенсора
            if camera_prim.IsValid():
                self._safety_camera = CameraSensor(prim_path=self._camera_prim_path)
                print(f"[INFO] Подключились к существующей камере на сцене: {self._camera_prim_path}")
            else:
                # 2. Если камеры нет — спавним с вашими точными параметрами
                self._safety_camera = CameraSensor(
                    prim_path=self._camera_prim_path,
                    position=cam_position,
                    resolution=(640, 480),
                )
                if hasattr(self._safety_camera, "set_focal_length"):
                    self._safety_camera.set_focal_length(focal_length)
                print(f"[INFO] Камера безопасности создана с потолочными параметрами: {self._camera_prim_path}")

            self._safety_camera.initialize()
            if hasattr(self._safety_camera, "add_motion_vectors_to_frame"):
                self._safety_camera.add_motion_vectors_to_frame()
        else:
            # Резервное создание через чистый USD Prim
            if not camera_prim.IsValid():
                from pxr import Gf, UsdGeom
                cam = UsdGeom.Camera.Define(stage, self._camera_prim_path)
                cam.AddTranslateOp().Set(Gf.Vec3d(float(cam_position[0]), float(cam_position[1]), float(cam_position[2])))
                cam.GetFocalLengthAttr().Set(focal_length)
                print(f"[INFO] USD Прим камеры создан по пути: {self._camera_prim_path}")

    async def setup_post_load(self) -> None:
        """Инициализация физики, логики и камеры."""
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

        if self._assets is None or self._robot is None:
            raise RuntimeError("Сцена не была инициализирована.")

        configure_conveyor(self._env_path)
        self._robot.initialize()
        self._spawner = BinSpawner(self._env_path, self._assets)
        self._context = BinStackingContext(self._robot, self._spawner)
        self._controller = PalletizerController()

        # Инициализируем камеру безопасности
        self._setup_safety_camera()

        self._reset_runtime_state()
        ViewportManager.set_camera_view(
            eye=[3.0, 3.0, 1.5],
            target=[0.35, 0.2, -0.4],
            camera="/OmniverseKit_Persp",
        )

    async def setup_pre_reset(self) -> None:
        """Очистка коробок и сброс состояния при остановке."""
        self._remove_callback()
        self._is_executing = False

        if self._robot is not None and hasattr(self._robot, "suction_gripper"):
            self._robot.suction_gripper.open()

        if self._spawner is not None:
            self._spawner.reset()

        await app_utils.update_app_async()

    async def setup_post_reset(self) -> None:
        self._reset_runtime_state()

    async def setup_post_clear(self) -> None:
        self.physics_cleanup()

    def physics_cleanup(self) -> None:
        self._remove_callback()
        self._assets = None
        self._robot = None
        self._spawner = None
        self._context = None
        self._controller = None
        self._safety_camera = None
        self._is_executing = False

    async def start_palletizing_async(self) -> bool:
        if not self._backend_ready() or self._is_executing:
            print("[WARNING] Не все компоненты готовы для запуска паллетизирования.")
            return False
        self._reset_runtime_state()
        self._is_executing = True
        self._physics_callback_id = SimulationManager.register_callback(
            self._physics_step,
            event=SimulationEvent.PHYSICS_POST_STEP,
        )
        app_utils.play()
        await app_utils.update_app_async()
        return True

    def _physics_step(self, dt: float, context: object) -> None:
        if not self._is_executing or not self._backend_ready():
            return

        # Шаг алгоритма укладки
        self._spawner.step()
        observation = self._context.read(SimulationManager.get_simulation_time())
        command = self._controller.step(observation)
        self._context.apply(command, dt)

        # Мониторинг камеры: захват кадров каждые 30 тиков
        self._frame_counter += 1
        if self._frame_counter % 30 == 0:
            self._capture_and_process_frame()

        if self._controller.state == "done":
            self._is_executing = False
            self._remove_callback()

    def _capture_and_process_frame(self) -> None:
        """Получение RGB-кадра для последующего анализа моделью YOLO."""
        if self._safety_camera is None:
            return

        rgba_frame = None
        if hasattr(self._safety_camera, "get_rgba"):
            rgba_frame = self._safety_camera.get_rgba()

        if rgba_frame is not None and rgba_frame.size > 0:
            rgb_frame = rgba_frame[:, :, :3]
            print(f"[CAMERA LOG] Кадр {self._frame_counter} успешно получен! Формат NumPy: {rgb_frame.shape}")
        else:
            print(f"[CAMERA LOG] Кадр {self._frame_counter}: сенсор подготавливается...")

    def _backend_ready(self) -> bool:
        return all(component is not None for component in (self._robot, self._spawner, self._context, self._controller))

    def _reset_runtime_state(self) -> None:
        if not self._backend_ready():
            return
        self._robot.reset()
        self._context.reset()
        self._controller.reset()

    def _remove_callback(self) -> None:
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None


# ==============================================================================
# БЛОК ЗАПУСКА С АВТОМАТИЧЕСКОЙ ОЧИСТКОЙ КОРОБОК ПРИ СТОПЕ
# ==============================================================================
if "_active_palletizing_sample" not in globals():
    _active_palletizing_sample = None
if "_timeline_sub" not in globals():
    _timeline_sub = None

def _on_timeline_event(event):
    global _active_palletizing_sample
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        if _active_palletizing_sample is not None and _active_palletizing_sample._spawner is not None:
            _active_palletizing_sample._spawner.reset()
            print("[INFO] Кнопка STOP нажата: спавнер коробок сброшен.")

async def _run_my_palletizing():
    global _active_palletizing_sample, _timeline_sub

    if omni.usd.get_context().get_stage():
        print("[INFO] Сохраняем текущие изменения сцены на диск...")
        omni.usd.get_context().save_stage()
        await app_utils.update_app_async()

    try:
        if _active_palletizing_sample is not None:
            _active_palletizing_sample.physics_cleanup()
            _active_palletizing_sample = None

        stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline_event)

        app_utils.stop()
        await app_utils.update_app_async()

        sample = Palletizing()
        _active_palletizing_sample = sample

        await sample.load_saved_scene()

        app_utils.play()
        await app_utils.update_app_async()

        await sample.setup_post_load()
        started = await sample.start_palletizing_async()

        if started:
            print("[SUCCESS] Скрипт и камера успешно запущены!")

    except Exception as e:
        print(f"[ERROR] Исключение: {e}")
        traceback.print_exc()

asyncio.ensure_future(_run_my_palletizing())