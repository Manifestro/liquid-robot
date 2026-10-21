import pytest
from steppe_robot.curricula import environment_parameters, validate_gps_start


def test_longer_task_keeps_gps_navigation_and_changes_only_goal_range():
    old = environment_parameters("gps_long")
    new = environment_parameters("gps_30_50")
    assert old["gps_navigation"] == new["gps_navigation"] == 1
    assert old["straight_goal"] == new["straight_goal"] == 0
    assert (new["goal_min_distance"], new["goal_max_distance"]) == (30, 50)


def test_start_validation_rejects_old_range_or_missing_gps_mode():
    start = {"gps_navigation": True, "spawn": {"z": -97}, "goal": {"z": -57}}
    validate_gps_start("gps_30_50", start)
    with pytest.raises(RuntimeError):
        validate_gps_start("gps_long", start)
    start["gps_navigation"] = False
    with pytest.raises(RuntimeError):
        validate_gps_start("gps_30_50", start)


def test_50_100_profile_stays_within_existing_map_and_uses_gps():
    params = environment_parameters("gps_50_100")
    assert params["goal_min_distance"] == 50
    assert params["goal_max_distance"] == 100
    assert params["gps_navigation"] == 1
    validate_gps_start("gps_50_100", {"gps_navigation": True, "spawn": {"z": -97}, "goal": {"z": 3}})
