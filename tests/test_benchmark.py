from steppe_robot.benchmark import ranking, summarize


def test_checkpoint_ranking_prioritizes_success_then_safety_then_return():
    summaries = [
        {"success_rate": 0.9, "collision_rate": 0, "mean_return": 10},
        {"success_rate": 1, "collision_rate": 0.1, "mean_return": 1},
        {"success_rate": 1, "collision_rate": 0, "mean_return": 0.5},
        {"success_rate": 1, "collision_rate": 0, "mean_return": 2},
    ]
    assert max(range(4), key=lambda i: ranking(summaries[i])) == 3


def test_summary_does_not_count_timeout_as_success():
    rows = [
        {"success": True, "collision": False, "return": 2, "elapsed_seconds": 10},
        {"success": False, "collision": False, "return": -0.5, "elapsed_seconds": 90},
    ]
    summary = summarize(rows)
    assert summary["episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["collision_rate"] == 0
    assert summary["mean_episode_seconds"] == 50
