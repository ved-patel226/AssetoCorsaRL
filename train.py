import argparse
import time
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from env import create_gym_env
from sac import SACPolicy, SACConfig, get_device

from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--log-interval", type=int, default=1_000)
    p.add_argument("--save-interval", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--load-replay",
        type=str,
        default=None,
        help="path to a saved LazyTensorStorage state dict (torch.save output) to load",
    )
    return p.parse_args()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device() if args.device is None else torch.device(args.device)
    print("Using device:", device)

    cfg = SACConfig()

    env = create_gym_env(device=device, num_envs=cfg.num_envs)
    td = env.reset()

    pixels = td["pixels"]

    # ===== Agent =====
    agent = SACPolicy(env=env, num_cells=cfg.num_cells, device=device)
    modules = agent.modules()

    actor = modules["actor"]
    value = modules["value"]
    q1 = modules["q1"]
    q2 = modules["q2"]

    print("Initializing lazy modules...")
    with torch.no_grad():
        sample_pixels = td["pixels"][:1].to(device)  # take first env, single batch
        sample_action = torch.zeros(1, env.action_spec.shape[-1], device=device)

        actor_input = TensorDict({"pixels": sample_pixels}, batch_size=1)
        actor(actor_input)

        value(sample_pixels)

        q1(sample_pixels, sample_action)
        q2(sample_pixels, sample_action)

    print("Lazy modules initialized")

    # ===== Optimizers =====
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    critic_opt = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=cfg.lr
    )
    value_opt = torch.optim.Adam(value.parameters(), lr=cfg.lr)

    print("using ReplayBuffer with LazyTensorStorage")
    storage = LazyTensorStorage(max_size=cfg.replay_size, device=device)
    if args.load_replay is not None:
        path = args.load_replay
        try:
            state = torch.load(path, map_location=device)
            storage.load_state_dict(state)
            print(f"Loaded replay storage from {path}")
        except Exception as e:
            print(f"Warning: could not load replay storage from {path}: {e}")
    rb = ReplayBuffer(storage=storage, batch_size=cfg.batch_size)

    def _reduce_value_to_batch(x, batch_size):
        """Reduce a value-shaped tensor or TensorDict to a (batch_size, 1) tensor.

        - If x is a Tensor with first dim == batch_size, average remaining dims to get a scalar per batch.
        - If x has a different leading dimension, attempt to reshape to (batch_size, -1) and average.
        - If x is a mapping (e.g., TensorDict), try to extract ['value'] and repeat the logic.
        Returns None on failure.
        """
        try:
            if isinstance(x, dict) or hasattr(x, "get"):
                v = x["value"] if "value" in x else x.get("value")
            else:
                v = x
            if not isinstance(v, torch.Tensor):
                return None
            if v.shape[0] == batch_size:
                if v.ndim == 1:
                    return v.view(-1, 1)
                return v.flatten(1).mean(dim=1, keepdim=True)
            # try to reshape
            return v.view(batch_size, -1).mean(dim=1, keepdim=True)
        except Exception:
            return None

    def sample_random_actions(num_envs, device=None):
        """Sample actions for CarRacing-v3 per-env.

        CarRacing action layout: [steer, gas, brake]
          - steer:  [-1, 1]
          - gas:   [0,  1]
          - brake: [0,  1]
        """
        device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        steer = torch.empty(num_envs, 1, device=device).uniform_(-1, 1)
        gas = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
        brake = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
        return torch.cat([steer, gas, brake], dim=-1)

    def sample_random_action(n=1, device=None):
        # thin wrapper to keep backwards compatibility with existing calls
        return sample_random_actions(n, device=device or device)

    total_steps = 0
    episode_returns = []
    current_episode_return = torch.zeros(cfg.num_envs, device=device)
    current_td = td

    print("Collecting initial random data...")
    while len(rb) < cfg.start_steps:
        actions = sample_random_action(cfg.num_envs)
        target_batch = current_td.batch_size
        if (
            isinstance(target_batch, (tuple, list, torch.Size))
            and len(target_batch) > 1
        ):
            if actions.shape[0] != target_batch[0]:
                raise ValueError(
                    f"Action batch size ({actions.shape[0]}) does not match env batch ({target_batch[0]})"
                )
            extra = target_batch[1:]
            new_shape = (actions.shape[0],) + (1,) * len(extra) + (actions.shape[1],)
            expand_shape = (actions.shape[0],) + tuple(extra) + (actions.shape[1],)
            actions_step = actions.view(new_shape).expand(expand_shape)
        else:
            actions_step = actions
        action_td = TensorDict({"action": actions_step}, batch_size=target_batch)

        next_td = env.step(action_td)

        td_next = next_td["next"] if "next" in next_td.keys() else next_td

        if "reward" in td_next.keys():
            rewards = td_next["reward"].flatten().to(device)
        elif "rewards" in td_next.keys():
            rewards = td_next["rewards"].flatten().to(device)
        else:
            raise KeyError(f"Unexpected TensorDict structure. Keys: {td_next.keys()}")

        dones = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
        dones |= td_next["done"].flatten().to(device).to(torch.bool)
        dones |= td_next["terminated"].flatten().to(device).to(torch.bool)
        dones |= td_next["truncated"].flatten().to(device).to(torch.bool)

        next_pixels = td_next["pixels"].to(device)

        # store each env separately
        pixels = current_td["pixels"].to(device)
        for i in range(cfg.num_envs):
            transition = TensorDict(
                {
                    "pixels": pixels[i],
                    "action": actions[i],
                    "reward": rewards[i].unsqueeze(0),
                    "next_pixels": next_pixels[i],
                    "done": dones[i].unsqueeze(0),
                },
                batch_size=[],
            )
            rb.add(transition)

        current_episode_return += rewards
        for i, d in enumerate(dones):
            if d.item():
                episode_returns.append(current_episode_return[i].item())
                current_episode_return[i] = 0.0

        current_td = next_td
        if "next" in next_td.keys() and "pixels" in next_td["next"].keys():
            current_td = next_td["next"]

    print(f"Initialized replay buffer with {len(rb)} transitions")
    torch.save(storage.state_dict(), "replay_buffer_init.pt")
    print("Replay buffer saved to replay_buffer_init.pt")

    start_time = time.time()

    while total_steps < args.total_steps:
        for _ in range(cfg.frames_per_batch):
            target_batch = current_td.batch_size
            if (
                isinstance(target_batch, (tuple, list, torch.Size))
                and len(target_batch) > 1
            ):
                if actions.shape[0] != target_batch[0]:
                    raise ValueError(
                        f"Action batch size ({actions.shape[0]}) does not match env batch ({target_batch[0]})"
                    )
                extra = target_batch[1:]
                new_shape = (
                    (actions.shape[0],) + (1,) * len(extra) + (actions.shape[1],)
                )
                expand_shape = (actions.shape[0],) + tuple(extra) + (actions.shape[1],)
                actions_step = actions.view(new_shape).expand(expand_shape)
            else:
                actions_step = actions
            action_td = TensorDict({"action": actions_step}, batch_size=target_batch)

            next_td = env.step(action_td)

            td_next = next_td["next"] if "next" in next_td.keys() else next_td

            # some envs rewards? some envs reward? idk. idrc.
            if "reward" in td_next.keys():
                rewards = td_next["reward"].flatten().to(device)
            elif "rewards" in td_next.keys():
                rewards = td_next["rewards"].flatten().to(device)
            else:
                raise KeyError(
                    f"Unexpected TensorDict structure. Keys: {td_next.keys()}"
                )

            # |= is OR assignment for tensors
            dones = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
            dones |= td_next["done"].flatten().to(device).to(torch.bool)
            dones |= td_next["terminated"].flatten().to(device).to(torch.bool)
            dones |= td_next["truncated"].flatten().to(device).to(torch.bool)

            pixels = current_td["pixels"].to(device)
            next_pixels = td_next["pixels"].to(device)

            for i in range(cfg.num_envs):
                transition = TensorDict(
                    {
                        "pixels": pixels[i],
                        "action": actions[i],
                        "reward": rewards[i].unsqueeze(0),
                        "next_pixels": next_pixels[i],
                        "done": dones[i].unsqueeze(0),
                    },
                    batch_size=[],
                )
                rb.add(transition)

            current_episode_return += rewards
            for i, d in enumerate(dones):
                if d.item():
                    episode_returns.append(current_episode_return[i].item())
                    current_episode_return[i] = 0.0

            current_td = next_td
            if "next" in next_td.keys() and "pixels" in next_td["next"].keys():
                current_td = next_td["next"]
            total_steps += cfg.num_envs

        updates_per_batch = max(1, cfg.frames_per_batch // cfg.batch_size)
        for _ in range(updates_per_batch):
            if len(rb) < cfg.batch_size:
                continue

            batch = rb.sample(cfg.batch_size)

            pixels_b = batch["pixels"].to(device)
            actions_b = batch["action"].to(device)
            rewards_b = batch["reward"].to(device).view(-1, 1)
            next_pixels_b = batch["next_pixels"].to(device)
            # Ensure done masks are float for arithmetic (avoid bool subtraction errors)
            dones_b = batch["done"].to(device).view(-1, 1).to(dtype=rewards_b.dtype)

            with torch.no_grad():
                next_v_raw = value(next_pixels_b)
                next_v = _reduce_value_to_batch(next_v_raw, next_pixels_b.shape[0])
                if next_v is None:
                    next_v = torch.zeros_like(rewards_b)

                q_target = rewards_b + cfg.gamma * (1.0 - dones_b) * next_v

            def _fix_action_shape(a, batch_size, action_dim=None):
                """Normalize action tensor to shape (batch_size, action_dim).

                Strategies applied in order:
                - Collapse extra dims so tensor has shape (batch_size, L)
                - If L == action_dim: done
                - If L is a multiple of action_dim: reshape to (batch, K, action_dim) and average over K
                - Otherwise: take the first `action_dim` elements along dim=1 as a fallback
                """
                if not isinstance(a, torch.Tensor):
                    return a
                if a.ndim == 1:
                    a = a.view(batch_size, -1)
                elif a.ndim > 2:
                    a = a.view(batch_size, -1)
                if action_dim is None:
                    return a
                L = a.shape[1]
                if L == action_dim:
                    return a
                if L % action_dim == 0:
                    return a.view(batch_size, L // action_dim, action_dim).mean(dim=1)
                return a[:, :action_dim]

            actions_b = _fix_action_shape(
                actions_b, pixels_b.shape[0], action_dim=env.action_spec.shape[-1]
            )
            q1_pred = q1(pixels_b, actions_b).view(-1, 1)
            q2_pred = q2(pixels_b, actions_b).view(-1, 1)

            # ===== critic loss =====
            q1_loss = F.mse_loss(q1_pred, q_target)
            q2_loss = F.mse_loss(q2_pred, q_target)
            critic_loss = q1_loss + q2_loss

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(q1.parameters()) + list(q2.parameters()), cfg.max_grad_norm
            )
            critic_opt.step()

            # ===== value loss =====
            td_in = TensorDict({"pixels": pixels_b}, batch_size=pixels_b.shape[0])
            out = actor(td_in)
            sampled_action = out["action"]
            sampled_action = _fix_action_shape(
                sampled_action,
                pixels_b.shape[0],
                action_dim=env.action_spec.shape[-1],
            )

            log_prob = out.get("log_prob")
            if log_prob is None:
                log_prob = out.get("log_prob")

            q1_for_v = q1(pixels_b, sampled_action).view(-1, 1)
            q2_for_v = q2(pixels_b, sampled_action).view(-1, 1)
            min_q = torch.min(q1_for_v, q2_for_v)

            value_pred_raw = value(pixels_b)
            value_pred = _reduce_value_to_batch(value_pred_raw, pixels_b.shape[0])
            if value_pred is None:
                value_pred = torch.zeros_like(min_q)

            if log_prob is not None and log_prob.ndim == 1:
                log_prob = log_prob.view(-1, 1)

            value_target = min_q - cfg.alpha * (
                log_prob if log_prob is not None else 0.0
            )
            value_loss = F.mse_loss(value_pred, value_target.detach())

            value_opt.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), cfg.max_grad_norm)
            value_opt.step()

            # ===== actor loss =====
            out = actor(td_in)
            new_actions = _fix_action_shape(
                out["action"],
                pixels_b.shape[0],
                action_dim=env.action_spec.shape[-1],
            )
            log_prob_new = out.get("log_prob")
            if log_prob_new is None:
                log_prob_new = torch.zeros((new_actions.shape[0], 1), device=device)

            q1_new = q1(pixels_b, new_actions).view(-1, 1)
            q2_new = q2(pixels_b, new_actions).view(-1, 1)
            min_q_new = torch.min(q1_new, q2_new)

            actor_loss = (cfg.alpha * log_prob_new - min_q_new).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

        if total_steps % args.log_interval < cfg.num_envs:
            elapsed = time.time() - start_time
            avg_return = sum(episode_returns[-100:]) / max(
                1, len(episode_returns[-100:])
            )
            print(
                f"Steps: {total_steps}, AvgReturn(100): {avg_return:.2f}, Buffer: {len(rb)}, Time: {elapsed:.1f}s"
            )

        if total_steps % args.save_interval < cfg.num_envs:
            torch.save(
                {
                    "actor_state": actor.state_dict(),
                    "q1_state": q1.state_dict(),
                    "q2_state": q2.state_dict(),
                    "value_state": value.state_dict(),
                    "actor_opt": actor_opt.state_dict(),
                    "critic_opt": critic_opt.state_dict(),
                    "value_opt": value_opt.state_dict(),
                    "steps": total_steps,
                },
                f"sac_checkpoint_{total_steps}.pt",
            )
            print(f"Saved checkpoint at step {total_steps}")

    print("Training finished")


if __name__ == "__main__":
    train()
