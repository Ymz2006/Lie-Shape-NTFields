"""MPPI on SE(3), with unit body tangents and a common geodesic step.

The chart is (position, rotvec / 2pi).  The sampling metric is
    d(g, h)^2 = |p_g - p_h|^2 + (angle(R_g^T R_h) / 2pi)^2.
A body tangent (v, w) uses turns for w, so its metric norm is |(v, w)|.
The Riemannian exponential is (p + t R v, R exp(2pi t [w]x)).
Consequently every unit sample has distance t, provided t < 1/2, and
the existing evaluator's step 0.015 gives exactly t = 0.015.

This is the Riemannian exponential for the normalized product metric,
not the SE(3) matrix exponential (whose translation includes a Jacobian).
See https://doi.org/10.1007/s00022-025-00754-9, Remark 2, equation (4).
The metric sets the sampling budget; the learned field still sets the costs.
"""

import math

import torch

from models.rnl import so3_exp, so3_log


TWO_PI = 2.0 * math.pi


def pose_distance(start, goal):
    """Normalized product geodesic distance; supports broadcast leading axes."""
    start_r = so3_exp(TWO_PI * start[..., 3:6])
    goal_r = so3_exp(TWO_PI * goal[..., 3:6])
    relative = start_r.transpose(-1, -2) @ goal_r
    angle_turns = so3_log(relative).norm(dim=-1) / TWO_PI
    return torch.sqrt((goal[..., :3] - start[..., :3]).square().sum(-1)
                      + angle_turns.square())


def riemannian_exp(current, tangent, t=1.0):
    """Map a body tangent at current to a pose, using the exact exponential.

    current and tangent have shape (..., 6).  t is a scalar or a tensor
    broadcastable to (..., 1).  Tangents need not be unit (the MPPI mean isn't).
    Returned rotation vectors are canonical, with angle at most pi.
    """
    displacement = tangent * t
    rotation = so3_exp(TWO_PI * current[..., 3:6])
    position = current[..., :3] + (
        rotation @ displacement[..., :3, None]).squeeze(-1)
    next_rotation = rotation @ so3_exp(TWO_PI * displacement[..., 3:6])
    return torch.cat([position, so3_log(next_rotation) / TWO_PI], dim=-1)


def sample_unit_tangents(batch, samples, horizon, device, dtype,
                         planar=False, generator=None):
    """Uniform directions, retaining the original shared/independent noise mix.

    Each horizon's marginal is uniform on S^5 (S^2 in planar mode).
    Normalizing, rather than capping, puts every proposal on the same sphere.
    """
    common = torch.randn(batch, samples, 1, 6, device=device, dtype=dtype,
                         generator=generator)
    independent = torch.randn(batch, samples, horizon, 6, device=device,
                              dtype=dtype, generator=generator)
    tangent = common + independent
    if planar:
        mask = tangent.new_tensor([1, 1, 0, 0, 0, 1])
        tangent = tangent * mask
    norm = tangent.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(tangent)
    fallback[..., 0] = 1.0
    return torch.where(norm > 0, tangent / norm.clamp_min(1e-12), fallback)


def sample_rollouts(current, tangents, t):
    """Compose each horizon step at its own current pose (no chart cumsum).

    current: (B, 6); tangents: (B, S, H, 6); t: scalar or (B, 1, 1).
    """
    pose = current[:, None, :].expand(-1, tangents.shape[1], -1)
    rollout = []
    for h in range(tangents.shape[2]):
        pose = riemannian_exp(pose, tangents[:, :, h], t)
        rollout.append(pose)
    return torch.stack(rollout, dim=2)


@torch.no_grad()
def mppi_batched(travel_times, pairs, steps=200, step=0.015,
                 temperature=50.0, samples=50, horizon=5, taper=0.0,
                 planar=False, alternating=False, candidate_cost=None,
                 generator=None):
    """Return the original evaluator's (paths, iterations, success, min_dist).

    pairs is (B, 12), with normalized start and goal configurations.  Costs,
    sample count, horizon, weighting and iteration budget match the original
    forward MPPI.  Mean controls are formed in one tangent space and executed
    with the same exponential; averaging chart positions is never used.
    Momentum is zero, matching the reference's explicit momentum = 0.
    Convergence and taper use physical pose distance with threshold 0.01.

    candidate_cost(pairs_of_rollouts, mover) optionally supplies the original
    alternating/local/horizon costs.  Sample t is shared across all directions;
    taper > 0 changes t per episode as in the original optional taper mode.
    """
    if pairs.ndim != 2 or pairs.shape[1] != 12 or not pairs.is_floating_point():
        raise ValueError('pairs must be a floating (B, 12) tensor')
    if not 0 < step < 0.5:
        raise ValueError('step must be in (0, 0.5) for shortest rotation geodesics')
    if steps < 1 or samples < 1 or horizon < 2 or temperature < 0 or taper < 0:
        raise ValueError('invalid MPPI budget, temperature, or taper')

    state = pairs.clone()
    batch = len(state)
    distance = pose_distance(state[:, :6], state[:, 6:])
    done = distance < 0.01
    min_distance = distance.clone()
    conv_step = torch.full((batch,), steps - 1, device=state.device,
                           dtype=torch.long)
    conv_step[done] = 0
    chain_a = [state[:, :6].clone()]
    chain_b = [state[:, 6:].clone()]
    lengths_a = torch.ones(batch, dtype=torch.long)
    lengths_b = torch.ones(batch, dtype=torch.long)
    moving_a = True

    for iteration in range(steps):
        if bool(done.all()):
            break
        if taper:
            radius = (pose_distance(state[:, :6], state[:, 6:]) * taper
                      ).clamp(min=step * 0.1, max=step)
        else:
            radius = state.new_full((batch,), step)
        tangent = sample_unit_tangents(batch, samples, horizon, state.device,
                                       state.dtype, planar, generator)
        candidates = sample_rollouts(state[:, :6], tangent,
                                     radius[:, None, None])
        targets = state[:, None, None, 6:].expand_as(candidates)
        rollout_pairs = torch.cat([candidates, targets], dim=-1)
        if candidate_cost is None:
            endpoints = rollout_pairs[:, :, [0, -1], :]
            costs = travel_times(endpoints.reshape(-1, 12)).reshape(
                batch, samples, 2)
            cost = 10 * costs[..., 0] + costs[..., 1]
        else:
            cost = candidate_cost(rollout_pairs, state[:, :6])
        weight = torch.softmax(-temperature * cost, dim=1)
        mean_tangent = torch.bmm(weight[:, None, :], tangent[:, :, 0]).squeeze(1)
        proposed = riemannian_exp(state[:, :6], mean_tangent, radius[:, None])
        live = ~done
        state[:, :6] = torch.where(live[:, None], proposed, state[:, :6])

        distance = pose_distance(state[:, :6], state[:, 6:])
        min_distance = torch.minimum(min_distance, distance)
        if moving_a:
            chain_a.append(state[:, :6].clone())
            lengths_a += live.cpu().long()
        else:
            chain_b.append(state[:, :6].clone())
            lengths_b += live.cpu().long()
        newly_done = (distance < 0.01) & live
        conv_step[newly_done] = iteration
        done |= newly_done
        if alternating:
            state = torch.cat([state[:, 6:], state[:, :6]], dim=1)
            moving_a = not moving_a

    paths = []
    for b in range(batch):
        path = [pose[b:b + 1] for pose in chain_a[:int(lengths_a[b])]]
        path += [pose[b:b + 1]
                 for pose in reversed(chain_b[:int(lengths_b[b])])]
        paths.append(path)
    return (paths, conv_step.cpu().tolist(), done.cpu().tolist(),
            min_distance.cpu().numpy())
