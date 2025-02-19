def clamp(x, min_val, max_val):
    # Clamp x so that min_val <= x <= max_val
    return max(min_val, min(x, max_val))
