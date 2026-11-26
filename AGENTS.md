# Diploma project

**Development of an Intelligent Vision-Based Safety Monitoring System for Robotic Manipulator in NVIDIA Isaac Sim**

The warehouse simulation uses a UR10 manipulator to stack products on a pallet. An assigned automated guided vehicle (AGV) collects the full pallet for transport to the shelving area. Other mobile robots may cross into the palletizing cell. The planned YOLO pipeline detects workers and mobile robots in camera frames so the safety logic can respond to entries into the warning and danger zones.

## Run the simulation

1. Launch NVIDIA Isaac Sim 5.0 on Windows. This machine has a standalone 5.0 installation at `C:\isaacsim\isaac-sim.bat`. The `isaacsim.exe` on the Windows Python PATH belongs to a separate 5.1 installation.
2. Open `sdggdgd.usd`, the main warehouse scene. In Isaac Sim's Script Editor, open and run `scripts/main_palletizing.py`. The script initializes the simulation and starts playback.
3. Run ROS 2 CLI commands in WSL from a shell with **Jazzy** sourced. Humble is the earlier setup. Match the WSL `ROS_DOMAIN_ID` to the Windows bridge. The script defaults to `42` when the Windows environment does not set a domain.

The Windows simulation uses the native `isaacsim.ros2.bridge` OmniGraph extension. Its ROS libraries are configured separately from WSL. The installed 5.0 `setup_ros_env.bat` defaults to Humble on Windows.

The saved warehouse scene references NVIDIA's 5.1 asset URLs and three AGV decal textures under `D:/diploma/scripts/decals/`. If assets are missing on another installation, check those references and the project textures in `scripts/decals/` first.

From the sourced WSL shell, check discovery and status while the simulation is playing:

```bash
export ROS_DOMAIN_ID=42
ros2 topic list
ros2 topic echo /ur10/status
```

`/ur10/safety_cmd` accepts `std_msgs/msg/String` commands `ESTOP`, `SLOWDOWN`, and `RESUME`. `/ur10/status` publishes JSON inside `std_msgs/msg/String`, with the active command, pallet state, and fleet telemetry. See the [manual ROS 2 commands](ros2commands.txt).

## Code and current behavior

- `scripts/main_palletizing.py` builds the cell, runs the UR10 palletizing cycle, dispatches an available authorized AGV when the pallet is full, and handles the ROS bridge. The UR10 step is gated by pallet and AGV service state and the current ROS command.
- `scripts/agv_manager.py` handles AGV routing, pallet pickup and return, side ID decals, and simulated telemetry. The current unload callback hides the bins at a configured drop-off point.
- `scripts/camera_sensor.py` sets up overhead and side USD cameras. Its stage reader produces ground-truth detections from prim positions and attributes for simulated personnel and AGVs. `capture_frames()` is a stub. YOLO inference is planned.
- `scripts/safety_module.py` evaluates detections and simulated fleet telemetry using world XY positions in meters, red/yellow/green zones, personnel roles and helmets, and AGV identity and task state. `evaluate_detected_objects()` is defined but has no caller in the current simulation loop.

## Intended safety workflow

Use the [block diagram](block-scheme.png) as the reference for the intended workflow. After packing, dispatch an available AGV, retain its ID/IP, and use the warning-zone and danger-zone checks to decide the response.

- The assigned AGV can enter the danger zone for the pallet handoff. For another AGV, the diagram calls for a slowdown while it passes or a stop during delivery, with resumption after departure.
- A person without a hard hat triggers a high alert and a stop. A helmeted robotics engineer triggers a slowdown in the warning zone and a stop in the danger zone. For another helmeted person, warn in the warning zone and raise a high alert and stop in the danger zone.
- An unidentified object triggers a medium alert and calls personnel in the warning zone. In the danger zone it triggers a high alert and stops the manipulator.

When implementing YOLO, keep the USD stage reader available for ground-truth comparison and convert camera detections to the world XY coordinates expected by the safety module. Use the diagram to check intended behavior and compare it with the current access matrix and AGV rules before changing outcomes. When changing simulation behavior, exercise a full pallet handoff and an intrusion scenario in Isaac Sim. For ROS bridge changes, also verify command and status traffic from WSL.
