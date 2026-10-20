import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from steppe_robot.contract import CAMERA_SHAPE, FIELDS, VECTOR_SIZE
from steppe_robot.inference import Policy
from steppe_robot.rl.model import CfCActorCritic


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(17)
    model = CfCActorCritic()
    path = tmp_path / "policy.pt"
    torch.save(
        {
            "format_version": 1,
            "algorithm": "CfC-sequence-PPO",
            "step": 256,
            "config": {"hidden_size": 128, "curriculum": "gps_long"},
            "model": model.state_dict(),
        },
        path,
    )
    return path


def observation():
    vector = np.zeros(VECTOR_SIZE, dtype=np.float32)
    vector[1090], vector[1092], vector[1095] = 20, 1, 0.1
    return np.zeros(CAMERA_SHAPE, dtype=np.uint8), vector


def test_policy_reset_replays_sequence_and_failure_preserves_state(checkpoint):
    policy = Policy(checkpoint)
    image, vector = observation()
    first, second = policy.act(image, vector), policy.act(image, vector)
    policy.reset()
    np.testing.assert_array_equal(first, policy.act(image, vector))
    bad = vector.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        policy.act(image, bad)
    np.testing.assert_array_equal(second, policy.act(image, vector))
    assert np.abs(first).max() <= 1


def test_policy_rejects_wrong_image_and_checkpoint(checkpoint, tmp_path):
    policy = Policy(checkpoint)
    image, vector = observation()
    with pytest.raises(ValueError, match="uint8"):
        policy.act(image.astype(np.float32), vector)
    invalid = tmp_path / "invalid.pt"
    torch.save({"format_version": 2}, invalid)
    with pytest.raises(ValueError, match="Unsupported"):
        Policy(invalid)


def test_inference_does_not_import_unity():
    code = "import steppe_robot.inference, sys; assert not any(k.startswith('mlagents') for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_vendored_ml_agents_matches_recorded_upstream_source():
    root = Path(__file__).parents[1]
    provenance = json.loads((root / "vendor/provenance.json").read_text())
    assert provenance["commit"] == "ee0a08ccae597094003844d0121317f9790a1676"
    assert provenance["modifications"] == []
    for name, expected in provenance["files_sha256"].items():
        assert hashlib.sha256((root / "vendor/mlagents-envs" / name).read_bytes()).hexdigest() == expected


def test_protocol_snapshot_matches_python_indices():
    snapshot = Path(__file__).parents[1] / "contracts/observation-v1.json"
    from importlib.resources import files

    assert files("steppe_robot").joinpath("schemas/observation-v1.json").read_bytes() == snapshot.read_bytes()
    schema = json.loads(snapshot.read_text())
    assert schema["vector_size"] == VECTOR_SIZE
    assert tuple(schema["camera"]["shape"]) == CAMERA_SHAPE
    occupied = []
    for name, value in FIELDS.items():
        expected = (
            {"start": value.start, "stop_exclusive": value.stop} if isinstance(value, slice) else {"index": value}
        )
        assert schema["fields"][name] == expected
        occupied.extend(range(value.start, value.stop) if isinstance(value, slice) else [value])
    assert sorted(occupied) == list(range(VECTOR_SIZE))


def test_predict_cli_matches_api(checkpoint, tmp_path):
    image, vector = observation()
    recorded = tmp_path / "observations.npz"
    output = tmp_path / "actions.npz"
    np.savez(
        recorded,
        images=np.stack([image, image]),
        vectors=np.stack([vector, vector]),
        episode_starts=np.array([True, True]),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "steppe_robot.inference",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(recorded),
            "--output",
            str(output),
        ],
        check=True,
    )
    with np.load(output) as result:
        assert result["actions"].shape == (2, 2)
        np.testing.assert_array_equal(result["actions"][0], result["actions"][1])
