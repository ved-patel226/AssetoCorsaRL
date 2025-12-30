import torch
import gc
import argparse
from env import create_gym_env


def sample_random_actions(num_envs, device):
    """Sample actions for CarRacing-v3 per-env.

    CarRacing action layout: [steer, gas, brake]
      - steer:  [-1, 1]
      - gas:   [0,  1]
      - brake: [0,  1]
    """
    steer = torch.empty(num_envs, 1, device=device).uniform_(-1, 1)
    gas = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
    brake = torch.empty(num_envs, 1, device=device).uniform_(0, 1)
    return torch.cat([steer, gas, brake], dim=-1)


def run_random_sim(num_envs: int = 4, max_steps: int = 1000, device=None, env=None):
    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Using device: {device}")

    if env is None:
        env = create_gym_env(
            num_envs=num_envs,
            device=device,
            # render_mode="human",
            render_mode=None,
        )

    td = env.reset()
    print("Reset done. TD keys:", list(td.keys()))

    cum_rewards = torch.zeros(num_envs, device=device)
    lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
    done_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for step in range(max_steps):
        actions = sample_random_actions(num_envs, device=device)
        from tensordict import TensorDict

        input_td = TensorDict({"action": actions}, batch_size=[num_envs])
        td = env.step(input_td)

        td_next = td["next"] if "next" in td.keys() else td

        if "reward" in td_next.keys():
            reward = td_next["reward"].view(num_envs)
        elif "rewards" in td_next.keys():
            reward = td_next["rewards"].view(num_envs)
        else:
            raise RuntimeError(
                f"Couldn't find reward. Top keys: {list(td.keys())}, nested keys: {list(td_next.keys())}"
            )

        cum_rewards += reward

        dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
        dones |= td_next["done"].view(num_envs).to(torch.bool)
        dones |= td_next["terminated"].view(num_envs).to(torch.bool)
        dones |= td_next["truncated"].view(num_envs).to(torch.bool)

        still_active = ~done_mask
        lengths = lengths + still_active.int()
        done_mask |= dones

        if (step + 1) % 100 == 0:
            print(
                f"Step {step+1}: cum_rewards={cum_rewards.cpu().numpy()}, done={done_mask.cpu().numpy()}"
            )

        if done_mask.all():
            print(f"All envs finished at step {step+1}")
            break

    print("Simulation finished")
    print("Steps executed:", step + 1)
    print("Episode lengths:", lengths.cpu().numpy())
    print("Cumulative rewards:", cum_rewards.cpu().numpy())

    try:
        env.close()
    except Exception:
        pass
    del env
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="number of runs to perform")
    parser.add_argument(
        "--num-envs", type=int, default=4, help="number of parallel envs per run"
    )
    parser.add_argument("--max-steps", type=int, default=1000, help="max steps per run")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = create_gym_env(num_envs=args.num_envs, device=device, render_mode=None)

    for run in range(1, args.runs + 1):
        print(f"\n=== Run {run}/{args.runs} ===")
        run_random_sim(
            num_envs=args.num_envs, max_steps=args.max_steps, device=device, env=env
        )

    try:
        env.close()
    except Exception:
        pass
    del env
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
