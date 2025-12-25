import itertools
from pathlib import Path
from getpass import getuser
from datetime import datetime
import warnings

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.vector import AsyncVectorEnv

from sac.sac import SAC
from sac.replay_memory import ReplayMemory, PrioritizedReplayMemory
from perception.utils import load_model, process_observation
from perception.generate_AE_data import generate_action


torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Set the device globally if a GPU is available."""


def train(
    seed: int = 69,
    batch_size: int = 256,
    num_steps: int = 5_000_000,
    updates_per_step: int = 1,
    start_steps: int = 100_000,
    replay_size: int = 5_000_000,
    eval: bool = True,
    eval_interval: int = 50,
    accelerated_exploration: bool = True,
    save_models: bool = True,
    load_models: bool = False,
    save_memory: bool = True,
    load_memory: bool = True,
    path_to_actor: str = "./models/sac_actor_carracer_klein_6_24_18.pt",
    path_to_critic: str = "./models/sac_critic_carracer_klein_6_24_18.pt",
    path_to_buffer: str = "./memory/buffer_talk2_6h7jpbd_12_25_15.pkl",
    num_envs: int = 4,
    use_per: bool = True,
    per_alpha: float = 0.6,
    per_beta_start: float = 0.4,
    per_beta_frames: int = 1_000_000,
    per_eps: float = 1e-6,
):
    """
    ## The train function consist of:

    - Setting up the environment, agent and replay buffer
    - Logging hyperparameters and training results
    - Loading previously saved actor and critic models
    - Training loop
    - Evaluation (every *eval_interval* episodes)
    - Saving actor and critic models

    ## Parameters:

    - **seed** *(int)*: Seed value to generate random numbers.
    - **batch_size** *(int)*: Number of samples that will be propagated through the Q, V, and policy network.
    - **num_steps** *(int)*: Number of steps that the agent takes in the environment. Determines the training duration.
    - **updates_per_step** *(int)*: Number of network parameter updates per step in the environment.
    - **start_steps** *(int)*:  Number of steps for which a random action is sampled. After reaching *start_steps* an action
    according to the learned policy is chosen.
    - **replay_size** *(int)*: Size of the replay buffer.
    - **eval** *(bool)*:  If *True* the trained policy is evaluated every *eval_interval* episodes.
    - **eval_interval** *(int)*: Interval of episodes after which to evaluate the trained policy.
    - **accelerated_exploration** *(bool)*: If *True* an action with acceleration bias is sampled.
    - **save_memory** *(bool)*: If *True* the experience replay buffer is saved to the harddrive.
    - **save_models** *(bool)*: If *True* actor and critic models are saved to the harddrive.
    - **load_models** *(bool)*: If *True* actor and critic models are loaded from *path_to_actor* and *path_to_critic*.
    - **path_to_actor** *(str)*: Path to actor model.
    - **path_to_critic** *(str)*: Path to critic model.

    """

    def make_env_fn(rank):
        def _init():
            env_ = gym.make("CarRacing-v3")
            # env_.seed(seed + rank)

            return env_

        return _init

    envs = AsyncVectorEnv([make_env_fn(i) for i in range(num_envs)])
    max_steps_per_env = envs.get_attr("_max_episode_steps")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # NOTE: ALWAYS CHECK PARAMETERS BEFORE TRAINING
    agent = SAC(
        envs.single_action_space,
        policy="Gaussian",
        gamma=0.99,
        lr=0.0003,
        alpha=0.2,
        automatic_temperature_tuning=True,
        batch_size=batch_size,
        hidden_size=512,
        target_update_interval=2,
        input_dim=32,
    )

    def beta_by_frame(frame_idx: int):
        return min(
            1.0, per_beta_start + (1.0 - per_beta_start) * frame_idx / per_beta_frames
        )

    memory = (
        PrioritizedReplayMemory(replay_size, alpha=per_alpha, eps=per_eps)
        if use_per
        else ReplayMemory(replay_size)
    )

    if load_memory:
        # load memory and deactivate random exploration
        memory.load(path_to_buffer)

    if load_memory or load_models:
        start_steps = 0

    # Training Loop
    total_numsteps = 0
    updates = 0

    # Log Settings and training results
    date = datetime.now()
    log_dir = Path(f"runs/{date.year}_SAC_{date.month}_{date.day}_{date.hour}")

    writer = SummaryWriter(log_dir=log_dir)

    settings_msg = (
        f"Training SAC for {num_steps} steps"
        "\n\nTRAINING SETTINGS:\n"
        f"Seed={seed}, Batch size: {batch_size}, Updates per step: {updates_per_step}\n"
        f"Accelerated exploration: {accelerated_exploration}, Start steps: {start_steps}, Replay size: {replay_size}"
        "\n\nALGORITHM SETTINGS:\n"
        f"Policy: {agent.policy_type}, Automatic temperature tuning: {agent.automatic_temperature_tuning}\n"
        f"Gamma: {agent.gamma}, Tau: {agent.tau}, Alpha: {agent.alpha}, LR: {agent.lr}\n"
        f"Target update interval: {agent.target_update_interval}, Latent dim: {agent.input_dim}, Hidden size: {agent.hidden_size}"
    )
    with open(log_dir / "settings.txt", "w") as file:
        file.write(settings_msg)

    if load_models:
        try:
            agent.load_model(path_to_actor, path_to_critic)
        except FileNotFoundError:
            warnings.warn(
                "Couldn't locate models in the specified paths. Training from scratch.",
                RuntimeWarning,
            )

    # Vectorized training loop
    # Initialize vector env states
    states, _ = envs.reset(seed=[seed + i for i in range(num_envs)])
    # Place cars at default starting positions for each env

    # Process and encode initial states per env
    processed_states = [encoder.sample(process_observation(s)) for s in states]

    episode_rewards = np.zeros(num_envs)
    episode_steps = np.zeros(num_envs, dtype=int)
    episode_counts = np.zeros(num_envs, dtype=int)

    while total_numsteps < num_steps:
        # Select actions for each env
        actions = []
        for i in range(num_envs):
            if total_numsteps < start_steps and not load_models:
                if accelerated_exploration:
                    # sample random action then apply acceleration bias
                    a = envs.single_action_space.sample()
                    a = generate_action(a)
                else:
                    a = envs.single_action_space.sample()
            else:
                a = agent.select_action(processed_states[i])
            actions.append(a)
        actions = np.stack(actions)

        # Update networks if we have enough samples
        if len(memory) > batch_size:
            beta = beta_by_frame(total_numsteps) if use_per else None
            for _ in range(updates_per_step):
                if use_per:
                    (
                        batch_state,
                        batch_action,
                        batch_reward,
                        batch_next_state,
                        batch_done,
                        batch_weight,
                        batch_idx,
                    ) = memory.sample(batch_size, beta)
                else:
                    (
                        batch_state,
                        batch_action,
                        batch_reward,
                        batch_next_state,
                        batch_done,
                    ) = memory.sample(batch_size)
                    batch_weight, batch_idx = None, None

                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = (
                    agent.update_parameters(
                        memory,
                        batch_size,
                        updates,
                        batch=(
                            batch_state,
                            batch_action,
                            batch_reward,
                            batch_next_state,
                            batch_done,
                        ),
                        weights=batch_weight,
                        idxs=batch_idx,
                    )
                )

                writer.add_scalar("loss/critic_1", critic_1_loss, total_numsteps)
                writer.add_scalar("loss/critic_2", critic_2_loss, total_numsteps)
                writer.add_scalar("loss/policy", policy_loss, total_numsteps)
                writer.add_scalar("loss/entropy_loss", ent_loss, total_numsteps)
                writer.add_scalar("entropy_temperature/alpha", alpha, total_numsteps)
                updates += 1

        # Step all envs
        next_states, rewards, terminated, truncated, infos = envs.step(actions)
        done_mask = np.asarray(terminated) | np.asarray(truncated)
        reset_obs = None
        if done_mask.any():
            reset_obs, _ = envs.reset()

        # Handle individual env done/resets and process observations
        for i in range(num_envs):
            ns = next_states[i]
            r = rewards[i]
            d = bool(done_mask[i])
            if d and reset_obs is not None:
                ns = reset_obs[i]

            ps = process_observation(ns)
            ps = encoder.sample(ps)

            # Determine mask for episode termination (time horizon)
            max_steps = max_steps_per_env[i] if max_steps_per_env is not None else None
            ep_step = episode_steps[i] + 1
            mask = (
                1 if (max_steps is not None and ep_step == max_steps) else float(not d)
            )

            # push transition for this env
            memory.push(processed_states[i], actions[i], float(r), ps, mask)

            # update trackers
            episode_steps[i] = ep_step if not d else 0
            episode_rewards[i] += r
            total_numsteps += 1

            if d:
                episode_counts[i] += 1
                writer.add_scalar(
                    f"reward/train_env_{i}", episode_rewards[i], total_numsteps
                )
                writer.add_scalar(
                    "reward/train", float(np.mean(episode_rewards)), total_numsteps
                )

                print(
                    f"Env {i} Episode: {episode_counts[i]}, total numsteps: {total_numsteps}, episode steps: {ep_step}, reward: {round(episode_rewards[i],2)}"
                )
                if episode_counts[i] % eval_interval == 0 and eval:
                    avg_reward = 0.0
                    episodes = 10
                    if save_models:
                        agent.save_model(
                            "carracer",
                            f"{getuser()}_{date.month}_{date.day}_{date.hour}",
                        )
                    for _ in range(episodes):
                        eval_env = gym.make("CarRacing-v3")
                        s, _ = eval_env.reset()
                        s = process_observation(s)
                        s = encoder.sample(s)
                        done_eval = False
                        ep_r = 0
                        while not done_eval:
                            a_eval = agent.select_action(s, eval=True)
                            s_next, r_eval, term_eval, trunc_eval, _ = eval_env.step(
                                a_eval
                            )
                            done_eval = term_eval or trunc_eval
                            s = process_observation(s_next)
                            s = encoder.sample(s)
                            ep_r += r_eval
                        avg_reward += ep_r
                        eval_env.close()
                    avg_reward /= episodes
                    if save_memory:
                        memory.save(
                            f"buffer_{getuser()}_{date.month}_{date.day}_{date.hour}"
                        )
                    writer.add_scalar("avg_reward/test", avg_reward, total_numsteps)
                    if save_models:
                        agent.save_model(
                            "carracer",
                            f"{getuser()}_{date.month}_{date.day}_{date.hour}",
                        )

                # reset episode reward for this env
                episode_rewards[i] = 0.0

            # store new processed state
            processed_states[i] = ps

    envs.close()


if __name__ == "__main__":
    encoder = load_model("models/weights.pt", vae=False)
    encoder.to(DEVICE)
    train(
        batch_size=512,
        load_memory=False,
        eval_interval=50,
        load_models=False,
        num_envs=4,
    )
