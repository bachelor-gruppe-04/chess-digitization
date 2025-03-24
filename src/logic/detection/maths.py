def clamp(x: float, min_val: float, max_val: float) -> float:
    # Clamp x so that min_val <= x <= max_val
    return max(min_val, min(x, max_val))
