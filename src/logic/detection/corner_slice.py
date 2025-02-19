class CornersManager:
    def __init__(self):
        # Initial state of the corners
        self.initial_state = {"h1": [50, -100], "a1": [0, -100], "a8": [0, -150], "h8": [50, -150]}
        self.state = self.initial_state.copy()

    def corners_set(self, key, xy):
        """Sets the coordinates for a specific corner key."""
        if key in self.state:
            self.state[key] = xy
        else:
            raise KeyError(f"{key} is not a valid corner key")

    def corners_reset(self):
        """Resets the corners to the initial state."""
        self.state = self.initial_state.copy()

    def get_corners(self):
        """Returns the current state of the corners."""
        return self.state

# Usage
corners_manager = CornersManager()

# Set new coordinates for a corner
corners_manager.corners_set("h1", [100, -200])

# Get the current state
print(corners_manager.get_corners())

# Reset to the initial state
corners_manager.corners_reset()
print(corners_manager.get_corners())
