import os
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gym
import numpy as np
import torch

from models.iqn import IQN
from config import Config
from scripts.car_racing.utils import to_chw, init_frame_stacks, reset_env


def load_and_run(
    model_path: str = "iqn_agent_final.pth",
    episodes: int = 3,
    frame_stack: int = 4,
    render: bool = False,
):
    cfg = Config()

    env = gym.make("CarRacing-v2", continuous=False)

    obs_shape_hwc = env.observation_space.shape
    obs_shape = (frame_stack, obs_shape_hwc[0], obs_shape_hwc[1])
    action_size = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = IQN(
        state_size=obs_shape,
        action_size=action_size,
        layer_size=cfg.layer_size,
        n_step=cfg.n_step,
        seed=cfg.seed,
        N=cfg.n,
        dueling=False,
        noisy=False,
        device=device,
    ).to(device)

    if not os.path.exists(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    envs = [env]
    stacks, states, start_times = init_frame_stacks(envs, frame_stack)

    for ep in range(1, episodes + 1):
        done = False
        score = 0.0
        # reset already done by init_frame_stacks; use states[0]
        state = states[0]
        while not done:
            if render:
                try:
                    env.render()
                except Exception:
                    pass

            s = np.array(state).astype(np.float32)
            s_t = torch.from_numpy(s).float().unsqueeze(0).to(device)
            with torch.no_grad():
                qvals = model.get_qvalues(s_t)
            action = int(qvals.cpu().numpy().argmax(axis=1)[0])

            step_ret = env.step(action)
            if len(step_ret) == 5:
                next_state, reward, terminated, truncated, info = step_ret
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_ret

            score += float(reward)

            next_chw = to_chw(next_state)
            stacks[0].append(next_chw)
            state = np.concatenate(list(stacks[0]), axis=0)

        print(f"Episode {ep} score: {score:.2f}")

        # reset for next episode
        s = reset_env(env)
        s_chw = to_chw(s)
        stacks[0].clear()
        for _ in range(frame_stack):
            stacks[0].append(s_chw)
        state = np.concatenate(list(stacks[0]), axis=0)

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    load_and_run()
