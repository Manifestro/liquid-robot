from types import SimpleNamespace
import numpy as np
import torch
import pytest
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from steppe_robot.rl.model import CfCActorCritic, action_log_prob
from steppe_robot.rl.ppo import Rollout, compute_gae, sequence_batch, update_ppo
from steppe_robot.rl.unity_env import Observation, Transition
from steppe_robot.training import collect, refresh_state


def inputs(batch=1, time=5):
    generator = torch.Generator().manual_seed(11)
    images = torch.randint(0, 256, (batch, time, 3, 120, 160), dtype=torch.uint8, generator=generator)
    vectors = torch.zeros(batch, time, 1118)
    vectors[..., :360] = 12
    vectors[..., 360:720] = 1
    vectors[..., 1090] = 20
    vectors[..., 1092] = 1
    vectors[..., 1095] = 0.1
    vectors[..., 1101:1105] = 1
    vectors[..., 1114:1116] = 1
    vectors[..., 1116:1118] = torch.tensor([0.02, 0.1])
    starts = torch.zeros(batch, time, dtype=torch.bool)
    starts[:, 0] = True
    return images, vectors, starts


def test_memory_sequence_matches_steps_and_resets():
    torch.manual_seed(3)
    model = CfCActorCritic()
    images, vectors, starts = inputs(batch=2)
    starts[0, 3] = True
    state = torch.randn(2, model.hidden_size)
    with torch.no_grad():
        means, _, values, final = model.forward_sequence(images, vectors, state, starts)
        pieces, value_pieces = [], []
        for i in range(5):
            mean, _, value, state = model.forward_sequence(
                images[:, i : i + 1], vectors[:, i : i + 1], state, starts[:, i : i + 1]
            )
            pieces.append(mean)
            value_pieces.append(value)
    torch.testing.assert_close(means, torch.cat(pieces, 1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(values, torch.cat(value_pieces, 1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(final, state, atol=1e-6, rtol=1e-5)


def test_reset_removes_previous_episode_and_padding_keeps_state():
    model = CfCActorCritic()
    images, vectors, starts = inputs()
    starts[0, 3] = True
    changed = images.clone()
    changed[:, :3] = 0
    with torch.no_grad():
        a = model.forward_sequence(images, vectors, model.initial_state(1, "cpu"), starts)
        b = model.forward_sequence(changed, vectors, torch.ones(1, 128) * 100, starts)
        torch.testing.assert_close(a[0][:, 3:], b[0][:, 3:])
        valid = torch.ones(1, 5, dtype=torch.bool)
        valid[:, 3:] = False
        padded = model.forward_sequence(images, vectors, model.initial_state(1, "cpu"), starts, valid)
        short = model.forward_sequence(images[:, :3], vectors[:, :3], model.initial_state(1, "cpu"), starts[:, :3])
        torch.testing.assert_close(padded[3], short[3], atol=1e-6, rtol=1e-5)


def test_gae_bootstraps_truncation_but_never_crosses_episode():
    advantages, returns = compute_gae(
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
        np.array([2.0, 0.0]),
        np.array([True, True]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages, [2.3, 0.5], atol=1e-6)
    np.testing.assert_allclose(returns, [2.8, 1], atol=1e-6)
    advantages, _ = compute_gae(
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([False, True]),
        gamma=0.9,
        gae_lambda=1,
    )
    np.testing.assert_allclose(advantages, [1.9, 1], atol=1e-6)


def test_squashed_probability_includes_jacobian_and_is_stable():
    mean, log_std = torch.tensor([[0.2, -0.3]]), torch.tensor([[-0.4, -0.6]])
    z = torch.tensor([[0.5, -0.8]])
    expected = TransformedDistribution(Normal(mean, log_std.exp()), [TanhTransform()]).log_prob(z.tanh()).sum(-1)
    torch.testing.assert_close(action_log_prob(z, mean, log_std), expected)
    extreme = torch.tensor([[30.0, -30.0]], requires_grad=True)
    loss = action_log_prob(extreme, mean, log_std).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()


def synthetic_rollout(model, size=19):
    rollout = Rollout.allocate(size, model.hidden_size)
    images, vectors, starts = inputs(time=size)
    rollout.images[:] = images[0].numpy()
    rollout.vectors[:] = vectors[0].numpy()
    device = next(model.parameters()).device
    state = model.initial_state(1, device)
    for i in range(size):
        begin = i % 9 == 0
        if begin:
            state = model.initial_state(1, device)
        rollout.starts[i] = begin
        rollout.states[i] = state[0].cpu().numpy()
        _, z, logp, value, state = model.act(rollout.images[i], rollout.vectors[i], state, begin)
        rollout.z[i], rollout.log_probs[i], rollout.values[i] = z, logp, value
    rollout.advantages = np.linspace(-1, 1, size).astype(np.float32)
    rollout.returns = rollout.values + rollout.advantages
    rollout.rewards[:] = 0.01
    rollout.next_values[:] = 0
    return rollout


def test_chronological_burn_in_reproduces_old_probabilities():
    model = CfCActorCritic()
    rollout = synthetic_rollout(model)
    batch = sequence_batch(rollout, [0, 8, 16], 8, 4, "cpu")
    assert batch["mask"].sum().item() == 19
    assert not batch["mask"][:, :4].any()
    with torch.no_grad():
        mean, std, _, _ = model.forward_sequence(
            batch["images"], batch["vectors"], batch["state"], batch["episode_starts"], batch["valid"], burn_in=4
        )
        logp = action_log_prob(batch["z"], mean, std)
    torch.testing.assert_close(logp[batch["mask"]], batch["old_log_probs"][batch["mask"]], atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_ppo_updates_cfc_and_camera_with_finite_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = CfCActorCritic().to(device)
    rollout = synthetic_rollout(model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    metrics = update_ppo(
        model,
        optimizer,
        rollout,
        epochs=2,
        sequence_length=8,
        burn_in=4,
        sequences_per_batch=2,
        rng=np.random.default_rng(2),
    )
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name.startswith("memory."))
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name.startswith("camera."))


def test_hidden_refresh_matches_episode_history():
    model = CfCActorCritic()
    images, vectors, starts = inputs(time=6)
    history = [Observation(images[0, t].numpy(), vectors[0, t].numpy()) for t in range(6)]
    with torch.no_grad():
        expected = model.forward_sequence(images, vectors, model.initial_state(1, "cpu"), starts)[3]
        actual = refresh_state(model, history, "cpu")
    torch.testing.assert_close(actual, expected)


def test_collector_uses_terminal_observation_for_truncation_and_resets_hidden(tmp_path):
    def observation(value):
        vector = np.zeros(1118, np.float32)
        vector[0] = value
        return Observation(np.zeros((3, 120, 160), np.uint8), vector)

    class FakeModel:
        hidden_size = 1

        def initial_state(self, batch, device):
            return torch.zeros(batch, 1)

        def act(self, image, vector, state, episode_start, deterministic=False):
            return np.zeros(2), np.zeros(2), 0.0, float(vector[0]), state + 1

    class FakeEnv:
        transitions = [
            Transition(observation(3), 2, True, True, observation(7), []),
            Transition(observation(4), 3, False, False, None, []),
            Transition(observation(5), 4, True, False, observation(8), []),
        ]

        def step(self, action):
            return self.transitions.pop(0)

    config = SimpleNamespace(device="cpu", gamma=0.99, gae_lambda=0.95)
    rollout, _, _, _, history, _ = collect(
        FakeModel(), FakeEnv(), observation(1), torch.zeros(1, 1), True, [], 3, config, tmp_path, 0
    )
    np.testing.assert_array_equal(rollout.next_values, [7, 4, 0])
    np.testing.assert_array_equal(rollout.starts, [True, True, False])
    np.testing.assert_array_equal(rollout.states[:, 0], [0, 0, 1])
    assert history == []
