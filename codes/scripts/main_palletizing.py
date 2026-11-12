# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
import traceback
import importlib
import omni.usd

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager
from isaacsim.examples.base import BaseSample
from isaacsim.robot_motion.examples.manipulation.ur10_palletizing import (
    BinSpawner, BinStackingContext, PalletizerController, Ur10, Ur10Assets, configure_conveyor
)

# 1. Абсолютный путь к скриптам
SCRIPTS_DIR = os.path.abspath(r"D:\дипломка\codes\scripts")

if os.path.exists(SCRIPTS_DIR):
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    print(f"[INFO] Путь {SCRIPTS_DIR} успешно добавлен в sys.path")
else:
    print(f"[ERROR] Папка не найдена: {SCRIPTS_DIR}")

# 2. Динамический импорт локальных модулей
try:
    safety_module = importlib.import_module("safety_module")
    agv_manager = importlib.import_module("agv_manager")
    camera_sensor = importlib.import_module("camera_sensor")

    importlib.reload(safety_module)
    importlib.reload(agv_manager)
    importlib.reload(camera_sensor)

    SafetyDecisionModule = safety_module.SafetyDecisionModule
    AGVManager = agv_manager.AGVManager
    SafetyCamera = camera_sensor.SafetyCamera

    print("[SUCCESS] Все 3 модуля успешно импортированы!")
except ModuleNotFoundError as e:
    print(f"[CRITICAL ERROR] Не удалось найти модуль: {e}")
    print(f"Текущие пути sys.path: {sys.path[:3]}")


class Palletizing(BaseSample):
    def __init__(self):
        super().__init__()
        self._env_path = "/World/Ur10Table"
        self._assets = None
        self._robot = None
        self._spawner = None
        self._context = None
        self._controller = None
        self._physics_callback_id = None
        self._is_executing = False

        # Менеджеры
        self.safety = SafetyDecisionModule()
        self.agv = AGVManager()
        self.camera = SafetyCamera()
        self._frame_counter = 0
        self._current_safety_state = "NORMAL"
        
        # Безопасный счетчик коробок
        self._placed_boxes_count = 0
        self._prev_controller_state = "idle"

    async def load_saved_scene(self):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            raise RuntimeError("В Isaac Sim нет открытой сцены!")

        robot_prim_path = f"{self._env_path}/ur10"
        if not stage.GetPrimAtPath(robot_prim_path).IsValid():
            if stage.GetPrimAtPath("/World/ur10").IsValid():
                robot_prim_path = "/World/ur10"
                self._env_path = "/World"

        self._assets = Ur10Assets()
        self._robot = Ur10(prim_path=robot_prim_path)
        print(f"[INFO] Подключились к роботу в текущей сцене: {robot_prim_path}")

    async def setup_post_load(self):
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

        configure_conveyor(self._env_path)
        self._robot.initialize()
        self._spawner = BinSpawner(self._env_path, self._assets)
        self._context = BinStackingContext(self._robot, self._spawner)
        self._controller = PalletizerController()

        self.camera.setup()

        ViewportManager.set_camera_view(
            eye=[3.0, 3.0, 1.5],
            target=[0.35, 0.2, -0.4],
            camera="/OmniverseKit_Persp",
        )

    def safe_spawner_reset(self):
        """Безопасный сброс спавнера без обрушения PhysX-тензоров."""
        print("[LOGISTICS] Сброс коробок и очистка паллета...")
        if self._context:
            self._context.reset()
        if self._controller:
            self._controller.reset()
        if self._spawner:
            self._spawner.reset()
        self._placed_boxes_count = 0

    def _physics_step(self, dt: float, context: object):
        if not self._is_executing or not self._backend_ready():
            return

        self._frame_counter += 1

        # 1. Логистика AGV
        self.agv.update(dt, spawner_reset_callback=self.safe_spawner_reset)

        # 2. Мониторинг безопасности
        if self._frame_counter % 30 == 0:
            detected = []
            if self.agv.state != "IDLE":
                detected.append({'class': 'agv', 'pos': (self.agv.current_pos[0], self.agv.current_pos[1])})

            new_state, log_msg = self.safety.evaluate_detected_objects(detected)
            if new_state != self._current_safety_state:
                self._current_safety_state = new_state
                print(f"[SAFETY MODULE] Status: {self._current_safety_state} -> {log_msg}")

        # 3. Учёт коробок по переходу состояний контроллера
        curr_state = self._controller.state
        if self._prev_controller_state == "placing" and curr_state == "idle":
            self._placed_boxes_count += 1
            print(f"[COUNTER] Уложена коробка #{self._placed_boxes_count}")

        self._prev_controller_state = curr_state

        # Диагностика в консоль каждые ~2 сек
        if self._frame_counter % 120 == 0:
            print(f"[DEBUG] Frame: {self._frame_counter} | Handled Boxes: {self._placed_boxes_count} | Controller: {curr_state} | AGV: {self.agv.state}")

        # 4. ТРИГГЕР: Вызов AGV при выполнении укладки 4 коробок
        if (curr_state == "done" or self._placed_boxes_count >= 4) and self.agv.state == "IDLE":
            print(f"[LOGISTICS] !!! 4 КОРОБКИ УЛОЖЕНЫ. ВЫЗОВ AGV !!!")
            self.agv.trigger_pallet_swap()

        # 5. Движение UR10 (только если AGV на месте)
        if self._current_safety_state != "STOP" and self.agv.state == "IDLE":
            try:
                effective_dt = dt * 0.3 if self._current_safety_state == "REDUCED_SPEED" else dt
                self._spawner.step()
                obs = self._context.read(SimulationManager.get_simulation_time())
                cmd = self._controller.step(obs)
                self._context.apply(cmd, effective_dt)
            except Exception as e:
                # Перехватываем микро-сбои удаленных примов во время цикла
                pass

    async def start_palletizing_async(self):
        if not self._backend_ready() or self._is_executing:
            return False

        self._is_executing = True
        self._physics_callback_id = SimulationManager.register_callback(
            self._physics_step, event=SimulationEvent.PHYSICS_POST_STEP
        )
        
        self._reset_runtime_state()
        return True

    def physics_cleanup(self):
        if self._physics_callback_id is not None:
            SimulationManager.deregister_callback(self._physics_callback_id)
            self._physics_callback_id = None
        self._is_executing = False

    def _backend_ready(self):
        return all(c is not None for c in (self._robot, self._spawner, self._context, self._controller))

    def _reset_runtime_state(self):
        if self._backend_ready():
            self._robot.reset()
            self._context.reset()
            self._controller.reset()
            self._placed_boxes_count = 0


# ==============================================================================
# БЛОК ИНИЦИАЛИЗАЦИИ И ЗАПУСКА
# ==============================================================================
if "_active_palletizing_sample" not in globals():
    _active_palletizing_sample = None
if "_timeline_sub" not in globals():
    _timeline_sub = None

def _on_timeline_event(event):
    global _active_palletizing_sample
    if event.type == int(omni.timeline.TimelineEventType.STOP):
        if _active_palletizing_sample is not None:
            _active_palletizing_sample.safe_spawner_reset()
            print("[INFO] Кнопка STOP нажата: спавнер сброшен.")

async def _run_my_palletizing():
    global _active_palletizing_sample, _timeline_sub

    timeline = omni.timeline.get_timeline_interface()

    try:
        if _active_palletizing_sample is not None:
            _active_palletizing_sample.physics_cleanup()
            _active_palletizing_sample = None

        stream = timeline.get_timeline_event_stream()
        _timeline_sub = stream.create_subscription_to_pop(_on_timeline_event)

        sample = Palletizing()
        _active_palletizing_sample = sample

        await sample.load_saved_scene()
        await sample.setup_post_load()

        timeline.play()
        await app_utils.update_app_async()

        started = await sample.start_palletizing_async()
        if started:
            print("[SUCCESS] Скрипт успешно привязан к сцене и запущен!")

    except Exception as e:
        print(f"[ERROR] Ошибка запуска: {e}")
        traceback.print_exc()

asyncio.ensure_future(_run_my_palletizing())