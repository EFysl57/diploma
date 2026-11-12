import math

class SafetyDecisionModule:
    def __init__(self, robot_origin=(0.0, 0.0), danger_radius=1.1, warning_radius=2.3):
        self.robot_origin = robot_origin
        self.danger_radius = danger_radius
        self.warning_radius = warning_radius

    def evaluate_detected_objects(self, detected_objects):
        if not detected_objects:
            return "NORMAL", "Zone is clear"

        highest_priority_state = "NORMAL"
        status_message = "Norman status"

        for obj in detected_objects:
            obj_class = obj.get("class", "unknown")
            pos = obj.get("pos", (99.0, 99.0))
            has_helmet = obj.get("has_helmet", True)

            dist = math.hypot(pos[0] - self.robot_origin[0], pos[1] - self.robot_origin[1])
            in_danger = dist <= self.danger_radius
            in_warning = self.danger_radius < dist <= self.warning_radius

            if obj_class == "agv" and (in_danger or in_warning):
                return "AGV_SERVICE", f"AGV in working zone ({dist:.2f}м). Pause."

            if obj_class in ["engineer", "visitor"] and not has_helmet:
                return "STOP", f"Emergency stop! Person with no hat! ({dist:.2f}м)"

            if in_danger:
                if obj_class == "engineer":
                    return "STOP", f"Safe stop: robot engineer in Danger Zone ({dist:.2f}м)"
                elif obj_class == "visitor":
                    return "STOP", f"Emergency!: Someone in Danger Zone! ({dist:.2f}м)"
                else:
                    return "STOP", f"Emergency stop: Unknown object ({dist:.2f}м)"
            elif in_warning:
                if obj_class == "engineer":
                    highest_priority_state = "REDUCED_SPEED"
                    status_message = f"Warning: robot engineer Warning Zone ({dist:.2f}м)."

        return highest_priority_state, status_message