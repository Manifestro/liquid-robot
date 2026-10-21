"""Versioned task presets; distances below configure forward road-goal placement."""

CURRICULA = ("navigation", "straight", "gps_long", "gps_30_50", "gps_50_100")
GPS_RANGES = {"gps_long": (10, 30), "gps_30_50": (30, 50), "gps_50_100": (50, 100)}


def environment_parameters(curriculum):
    if curriculum not in CURRICULA:
        raise ValueError(f"Unknown curriculum: {curriculum}")
    if curriculum == "straight":
        return {"goal_min_distance": 3, "goal_max_distance": 5, "straight_goal": 1}
    if curriculum in GPS_RANGES:
        low, high = GPS_RANGES[curriculum]
        return {"goal_min_distance": low, "goal_max_distance": high, "gps_navigation": 1, "straight_goal": 0}
    return {}


def validate_gps_start(curriculum, start):
    if curriculum not in GPS_RANGES:
        return
    low, high = GPS_RANGES[curriculum]
    travel = start.get("goal", {}).get("z", 0) - start.get("spawn", {}).get("z", 0)
    if not start.get("gps_navigation") or not low - 0.1 <= travel <= high + 0.1:
        raise RuntimeError(f"GPS task requires the GPS-navigation build and {low}-{high} m forward goal placement")
