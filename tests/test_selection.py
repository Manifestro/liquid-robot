import pytest

from steppe_robot.selection import should_replace_best, validation_score, validate_seed_sets


def summary(success, collisions=0, reward=1):
    return {"episodes": 12, "success_rate": success, "collision_rate": collisions, "mean_return": reward}


def test_later_regression_does_not_replace_best_checkpoint():
    incumbent = summary(1)
    assert should_replace_best(incumbent, None)
    assert not should_replace_best(summary(0.7, reward=2), incumbent)
    assert not should_replace_best(incumbent, incumbent)
    assert should_replace_best(summary(1, reward=1.1), incumbent)
    assert should_replace_best(summary(1, 0, 0.5), summary(1, 0.1, 2))


def test_invalid_validation_results_cannot_select_checkpoint():
    with pytest.raises(ValueError, match="empty"):
        validation_score({"episodes": 0})
    with pytest.raises(ValueError, match="non-finite"):
        validation_score(summary(1, reward=float("nan")))
    with pytest.raises(ValueError, match="non-finite"):
        should_replace_best(summary(1, reward=float("nan")), None)


@pytest.mark.parametrize("validation,test", [([1], [1]), ([-1], [3]), ([1, 1], [3]), ([2], [3]), ([], [3])])
def test_seed_sets_prevent_test_leakage_and_random_replay(validation, test):
    with pytest.raises(ValueError):
        validate_seed_sets(validation, test)
