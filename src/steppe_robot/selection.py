"""Checkpoint selection uses validation outcomes, never training return or test outcomes."""

import math


def validation_score(summary):
    if summary.get("episodes", 0) < 1:
        raise ValueError("Cannot rank an empty validation set")
    score = (summary["success_rate"], -summary["collision_rate"], summary["mean_return"])
    if not all(math.isfinite(value) for value in score):
        raise ValueError("Cannot rank non-finite validation metrics")
    return score


def should_replace_best(candidate, incumbent):
    candidate_score = validation_score(candidate)
    return incumbent is None or candidate_score > validation_score(incumbent)


def validate_seed_sets(validation, test):
    if not validation or not test:
        raise ValueError("Validation and test sets must be nonempty")
    for seeds in (validation, test):
        if len(set(seeds)) != len(seeds) or any(seed < 0 or seed % 2 != 1 for seed in seeds):
            raise ValueError("Each seed set must contain unique nonnegative odd seeds")
    if set(validation) & set(test):
        raise ValueError("Validation and test seed sets must be disjoint")
