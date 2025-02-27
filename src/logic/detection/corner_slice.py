# Initial state of the corners
_initial_state = {"h1": [0, 0], "a1": [0, 0], "a8": [0, 0], "h8": [0, 0]}
_state = _initial_state.copy()

def corners_set(key, xy):
    """Sets the coordinates for a specific corner key."""
    if key in _state:
        _state[key] = xy
    else:
        raise KeyError(f"{key} is not a valid corner key")

def corners_reset():
    """Resets the corners to the initial state."""
    global _state
    _state = _initial_state.copy()

def get_corners():
    """Returns the current state of the corners."""
    return _state

# Usage
# Set new coordinates for a corner
corners_set("h1", [100, -200])

corners_reset()
