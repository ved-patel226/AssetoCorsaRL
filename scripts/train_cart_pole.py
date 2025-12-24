import os
import sys
import random


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.agent.iqn_agent import IQN_Agent


def train_cartpole(
    episodes=500,
    max_t=500,
    batch_size=64,
    buffer_size=10000,
    lr=1e-3,
    seed=0,
):
    import gym
    import numpy as np
    import wandb
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # older torch versions may not support this
        pass

    wandb.init(
        project="cartpole-iqn",
        config={
            "episodes": episodes,
            "max_t": max_t,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "lr": lr,
            "seed": seed,
        },
    )

    # run multiple independent envs in parallel (interleaved stepping)
    n_envs = 4
    envs = [gym.make("CartPole-v1") for _ in range(n_envs)]
    obs_shape = envs[0].observation_space.shape
    action_size = envs[0].action_space.n

    # choose GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        wandb.config.update({"device": device})
    except Exception:
        pass

    class WandbWriter:
        def __init__(self):
            self._step = 0

        def _next_step(self, step):
            if step is None:
                self._step += 1
            else:
                if step <= self._step:
                    self._step += 1
                else:
                    self._step = step
            return self._step

        def add_scalar(self, tag, value, step=None, **kwargs):
            s = self._next_step(step)
            wandb.log({tag: value}, step=s)

        def log_dict(self, data, step=None):
            s = self._next_step(step)
            wandb.log(data, step=s)

    writer = WandbWriter()

    agent = IQN_Agent(
        state_size=obs_shape,
        action_size=action_size,
        network="",
        munchausen=False,
        layer_size=128,
        n_step=1,
        BATCH_SIZE=batch_size,
        BUFFER_SIZE=buffer_size,
        LR=lr,
        TAU=1e-3,
        GAMMA=0.99,
        N=32,
        worker=n_envs,
        device=device,
        seed=seed,
    )

    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995
    eps = eps_start

    def _reset_env(e):
        s = e.reset()
        return s[0] if isinstance(s, tuple) else s

    states = [_reset_env(e) for e in envs]
    scores = []
    scores_per_env = [0.0 for _ in range(n_envs)]
    completed_episodes = 0

    while completed_episodes < episodes:
        for i in range(n_envs):
            if completed_episodes >= episodes:
                break

            state = states[i]
            action = agent.act(state, eps=eps)
            if isinstance(action, (list, tuple, np.ndarray)):
                a = int(action[0])
            else:
                a = int(action)

            step_ret = envs[i].step(a)
            if len(step_ret) == 5:
                next_state, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_ret

            agent.step(state, a, reward, next_state, done, writer)
            states[i] = next_state
            scores_per_env[i] += reward

            if done:
                score = scores_per_env[i]
                scores.append(score)
                completed_episodes += 1

                eps = max(eps_end, eps_decay * eps)

                recent = scores[-10:]
                avg10 = sum(recent) / len(recent)
                best10 = max(recent)

                writer.log_dict(
                    {
                        "score": score,
                        "avg10": avg10,
                        "best10": best10,
                        "eps": eps,
                        "episode": completed_episodes,
                    },
                    step=completed_episodes,
                )

                if completed_episodes % 20 == 1:
                    print()
                    print(
                        "{:<8} {:>8} {:>10} {:>8} {:>8}".format(
                            "Episode", "Score", "Avg(10)", "Best10", "Eps"
                        )
                    )
                    print("-" * 48)

                print(
                    "{:<8d} {:>8.2f} {:>10.2f} {:>8.2f} {:>8.3f}".format(
                        completed_episodes, score, avg10, best10, eps
                    )
                )

                s = envs[i].reset()
                states[i] = s[0] if isinstance(s, tuple) else s
                scores_per_env[i] = 0.0

    for e in envs:
        e.close()
    wandb.finish()


if __name__ == "__main__":
    train_cartpole()
