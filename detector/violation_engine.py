import time

class ViolationEngine:
    def __init__(self):
        self.look_history = []
        self.last_reset = time.time()

    def analyze(self, direction, objects):
        self.look_history.append(direction)

        # keep last 30 frames
        if len(self.look_history) > 30:
            self.look_history.pop(0)

        violations = []

        # Rule 1: Looking away frequently
        if self.look_history.count("LEFT") + self.look_history.count("RIGHT") > 15:
            violations.append("Looking Away Frequently")

        # Rule 2: Phone detection
        if "cell phone" in objects:
            violations.append("Mobile Phone Detected")

        # Rule 3: Face not visible
        if direction == "NO_FACE":
            violations.append("Face Not Visible")

        return violations