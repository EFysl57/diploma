import math
from typing import Dict, List, Optional, Set, Tuple


class SafetyDecisionModule:
    """Dual-contour industrial safety decision module:
    1. Spatial 3-zone (Red/Yellow/Green) + 4-role personnel access control matrix.
    2. Cross-verification between CV camera detections (overhead + side entrance camera)
       and network telemetry packets (robot_id, authorized, task_flag, net_pos).
    """

    PERSONNEL_ROLES = {
        "robotics_engineer",
        "electrician",
        "tech_staff",
        "warehouse_worker",
    }

    ACCESS_MATRIX: Dict[Tuple[str, str], Tuple[str, str]] = {
        ("robotics_engineer", "GREEN"): ("NORMAL", "Robotics engineer in Green logistics zone"),
        ("robotics_engineer", "YELLOW"): ("REDUCED_SPEED", "Robotics engineer in Yellow buffer: UR10 speed reduced"),
        ("robotics_engineer", "RED"): ("STOP", "Safe stop: robotics engineer in Red UR10 zone"),
        ("electrician", "GREEN"): ("NORMAL", "Electrician in Green logistics zone"),
        ("electrician", "YELLOW"): ("REDUCED_SPEED", "Electrician in Yellow buffer: UR10 speed reduced"),
        ("electrician", "RED"): ("STOP", "LOTO safety stop: electrician in Red UR10 zone"),
        ("tech_staff", "GREEN"): ("NORMAL", "Tech staff in Green logistics zone"),
        ("tech_staff", "YELLOW"): ("REDUCED_SPEED", "Tech staff in Yellow buffer: UR10 speed reduced"),
        ("tech_staff", "RED"): ("STOP", "Unauthorized entry: tech staff in Red UR10 zone"),
        ("warehouse_worker", "GREEN"): ("NORMAL", "Warehouse worker in Green logistics corridor"),
        ("warehouse_worker", "YELLOW"): ("STOP", "Access violation: warehouse worker in Yellow buffer zone"),
        ("warehouse_worker", "RED"): ("STOP", "Emergency stop: warehouse worker in Red UR10 zone"),
    }

    VALID_SERVICE_TASK_FLAGS = {
        "FETCH_FULL_PALLET",
        "DOCKING_UNDER_PALLET",
        "LIFTING_PALLET",
        "HAULING_LOADED_PALLET",
        "SWAPPING_EMPTY_PALLET",
        "RETURNING_EMPTY_PALLET",
        "PLACING_EMPTY_PALLET",
        "LOWERING_PALLET",
        "UNDOCKING",
        "RETURNING_STANDBY",
    }

    STATE_PRIORITY = {
        "NORMAL": 0,
        "REDUCED_SPEED": 1,
        "AGV_SERVICE": 2,
        "STOP": 3,
    }

    def __init__(
        self,
        robot_origin: Tuple[float, float] = (0.35, 0.30),
        red_radius: float = 1.25,
        yellow_radius: float = 2.40,
        max_cv_net_pos_error: float = 0.65,
    ):
        self.robot_origin = robot_origin
        self.red_radius = red_radius
        self.yellow_radius = yellow_radius
        self.max_cv_net_pos_error = max_cv_net_pos_error
        self.blocked_agv_ids: Set[str] = set()

    def get_zone_for_position(self, pos: Tuple[float, float]) -> Tuple[str, float]:
        dist = math.hypot(pos[0] - self.robot_origin[0], pos[1] - self.robot_origin[1])
        if dist <= self.red_radius:
            return "RED", dist
        if dist <= self.yellow_radius:
            return "YELLOW", dist
        return "GREEN", dist

    def _elevate_state(
        self,
        current_state: str,
        current_msg: str,
        candidate_state: str,
        candidate_msg: str,
    ) -> Tuple[str, str]:
        if self.STATE_PRIORITY[candidate_state] > self.STATE_PRIORITY[current_state]:
            return candidate_state, candidate_msg
        return current_state, current_msg

    def _verify_agv_dual_contour(
        self,
        cv_det: dict,
        telemetry_by_id: Dict[str, dict],
        zone: str,
        dist: float,
    ) -> Tuple[bool, str]:
        cv_id = cv_det.get("cv_id", "UNKNOWN")
        cv_pos = cv_det.get("pos", (99.0, 99.0))
        cv_flag = cv_det.get("cv_flag", "UNKNOWN")

        net_pkt = telemetry_by_id.get(cv_id)
        if net_pkt is None:
            return False, f"CV-Network mismatch: visual '{cv_id}' has no matching network telemetry"

        if not net_pkt.get("authorized", False):
            return False, f"Unauthorized AGV '{cv_id}' detected in {zone} zone ({dist:.2f}m)"

        net_pos = net_pkt.get("net_pos", (99.0, 99.0))
        pos_err = math.hypot(cv_pos[0] - net_pos[0], cv_pos[1] - net_pos[1])
        if pos_err > self.max_cv_net_pos_error:
            return False, f"Telemetry spoofing/drift on '{cv_id}': CV vs Net error {pos_err:.2f}m"

        net_flag = net_pkt.get("task_flag", "STANDBY")
        if cv_flag != "UNKNOWN" and cv_flag != net_flag:
            return False, f"Task flag mismatch on '{cv_id}': CV='{cv_flag}' vs Net='{net_flag}'"

        if zone in ("RED", "YELLOW") and net_flag not in self.VALID_SERVICE_TASK_FLAGS:
            return False, f"Unscheduled entry by '{cv_id}' into {zone} zone (flag={net_flag})"

        return True, f"Dual-contour verified {cv_id} [{net_flag}] in {zone} zone ({dist:.2f}m)"

    def evaluate_detected_objects(
        self,
        detected_objects: List[dict],
        agv_telemetry: Optional[List[dict]] = None,
    ) -> Tuple[str, str]:
        self.blocked_agv_ids.clear()
        if not detected_objects:
            return "NORMAL", "All safety zones clear"

        telemetry_by_id = {pkt["robot_id"]: pkt for pkt in (agv_telemetry or []) if "robot_id" in pkt}
        best_state = "NORMAL"
        best_msg = "All safety zones clear"

        for obj in detected_objects:
            obj_class = obj.get("class", "unknown")
            pos = obj.get("pos", (99.0, 99.0))
            zone, dist = self.get_zone_for_position(pos)

            if obj_class == "agv":
                cv_id = obj.get("cv_id", "UNKNOWN")
                is_valid, verify_msg = self._verify_agv_dual_contour(obj, telemetry_by_id, zone, dist)
                if not is_valid:
                    self.blocked_agv_ids.add(cv_id)
                    if zone in ("RED", "YELLOW"):
                        return "STOP", f"SECURITY HALT: {verify_msg}"
                    best_state, best_msg = self._elevate_state(
                        best_state, best_msg, "REDUCED_SPEED", f"Security warning: {verify_msg}"
                    )
                elif zone in ("RED", "YELLOW"):
                    best_state, best_msg = self._elevate_state(
                        best_state, best_msg, "AGV_SERVICE", f"{verify_msg} -> UR10 paused"
                    )
                continue

            if obj_class in self.PERSONNEL_ROLES:
                if not obj.get("has_helmet", True):
                    return "STOP", f"PPE violation: {obj_class} without helmet in {zone} zone ({dist:.2f}m)"

                action_state, reason = self.ACCESS_MATRIX.get(
                    (obj_class, zone),
                    ("STOP", f"Access denied for {obj_class} in {zone}"),
                )
                best_state, best_msg = self._elevate_state(
                    best_state, best_msg, action_state, f"{reason} ({dist:.2f}m)"
                )
                if best_state == "STOP":
                    return best_state, best_msg
                continue

            if zone in ("RED", "YELLOW"):
                return "STOP", f"Emergency stop: unknown object '{obj_class}' in {zone} zone ({dist:.2f}m)"

        return best_state, best_msg
