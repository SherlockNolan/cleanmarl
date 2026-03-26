#!/usr/bin/env python3
"""
Training script for MAPPO on 3chaser1 Unity environment.

Usage:
    python scripts/train_mappo_3chase1_unity.py --total_timesteps 1000000

Environment must be built and available at the specified path.
"""

import datetime
import random
from multiprocessing import Pipe, Process

import numpy as np
import torch
import tyro
from dataclasses import dataclass, field
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from cleanmarl.algorithms import MAPPO, MAPPOConfig, RolloutBuffer
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


@dataclass
class Args:
    """Command line arguments for training."""
    batch_size: int = 1
    """ Number of parallel environments"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    log_every: int = 10
    """ Logging steps """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 42
    """ Random seed"""
    # MAPPO specific args (forwarded to MAPPOConfig)
    actor_hidden_dim: int = 32
    actor_num_layers: int = 3
    critic_hidden_dim: int = 64
    critic_num_layers: int = 1
    optimizer: str = "Adam"
    learning_rate_actor: float = 0.0008
    learning_rate_critic: float = 0.0008
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = False
    normalize_return: bool = False
    epochs: int = 3
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    clip_gradients: float = -1
    # Unity specific
    unity_build_path: str = "/home/fins/UnderwaterSim/Code/UnityProject/RLChase/build/RLChase.x86_64"
    """ Path to Unity binary build"""
    max_steps: int = 900
    """ Max steps per episode """


class UnityEnvWrapper:
    """Wrapper for Unity Parallel Environment to provide unified interface."""

    def __init__(self, unity_parallel_env, max_steps: int = 900):
        self.env = unity_parallel_env
        self.n_agents = len(unity_parallel_env.possible_agents)
        self.agents = unity_parallel_env.possible_agents

        self.agent_obs_sizes = []
        for agent in self.agents:
            obs_space = unity_parallel_env.observation_space(agent)
            if hasattr(obs_space, 'spaces'):
                obs_size = sum(int(np.prod(space.shape)) for space in obs_space.spaces)
            elif hasattr(obs_space, 'shape'):
                obs_size = int(np.prod(obs_space.shape))
            else:
                obs_size = obs_space.n
            self.agent_obs_sizes.append(obs_size)

        self.obs_size = max(self.agent_obs_sizes)

        self.agent_action_sizes = []
        for agent in self.agents:
            action_space = unity_parallel_env.action_space(agent)
            if hasattr(action_space, 'n'):
                action_size = action_space.n
            elif hasattr(action_space, 'spaces'):
                action_size = 0
                for space in action_space.spaces:
                    if hasattr(space, 'n'):
                        action_size += space.n
                    elif hasattr(space, 'shape') and space.shape is not None:
                        action_size += int(np.prod(space.shape))
            elif hasattr(action_space, 'shape') and action_space.shape is not None:
                action_size = int(np.prod(action_space.shape))
            else:
                action_size = 1
            self.agent_action_sizes.append(action_size)

        self.action_size = max(self.agent_action_sizes)
        self.state_size = self.obs_size * self.n_agents
        self.max_steps = max_steps
        self._current_step = 0
        self._last_obs = None

    def reset(self, seed: int = None):
        if seed is not None:
            self.env.seed(seed)
        obs_dict = self.env.reset()
        self._last_obs = obs_dict
        self._current_step = 0
        return self._process_obs(obs_dict)

    def step(self, actions):
        action_dict = {}
        for i, agent in enumerate(self.agents):
            action = np.array(actions[i]).flatten()
            agent_action_size = self.agent_action_sizes[i]
            if action.shape[0] < agent_action_size:
                action = np.pad(action, (0, agent_action_size - action.shape[0]), constant_values=0.0)
            elif action.shape[0] > agent_action_size:
                action = action[:agent_action_size]
            action_dict[agent] = action.astype(np.float32)

        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        self._last_obs = obs_dict
        self._current_step += 1

        obs_array = self._process_obs(obs_dict)
        reward = sum(reward_dict.values()) / len(reward_dict)
        done = all(done_dict.values())
        truncated = self._current_step >= self.max_steps

        return obs_array, reward, done, truncated, info_dict

    def _process_obs(self, obs_dict):
        obs_list = []
        for i, agent in enumerate(self.agents):
            obs = obs_dict[agent]
            if isinstance(obs, dict) and "observation" in obs:
                obs = np.concatenate([np.array(o).flatten() for o in obs["observation"]])
            else:
                obs = np.array(obs).flatten()

            if len(obs) < self.obs_size:
                obs = np.pad(obs, (0, self.obs_size - len(obs)), constant_values=0)
            elif len(obs) > self.obs_size:
                obs = obs[:self.obs_size]
            obs_list.append(obs)
        return np.array(obs_list)

    def get_obs_size(self):
        return self.obs_size

    def get_action_size(self):
        return self.action_size

    def get_state_size(self):
        return self.state_size

    def get_state(self):
        if self._last_obs is None:
            return np.zeros(self.state_size)
        return self._process_obs(self._last_obs).flatten()

    def sample(self):
        actions = []
        for i, agent in enumerate(self.agents):
            action = self.env.action_space(agent).sample()
            action = np.array(action).flatten()
            if action.shape[0] < self.action_size:
                action = np.pad(action, (0, self.action_size - action.shape[0]), constant_values=0.0)
            elif action.shape[0] > self.action_size:
                action = action[:self.action_size]
            actions.append(action)
        return np.array(actions, dtype=np.float32)

    def close(self):
        self.env.close()


def get_unity_env(env_config, seed):
    """Create Unity environment for training workers."""
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=10.0)
    worker_id = env_config["worker_index"]

    unity_env = UnityEnvironment(
        file_name=env_config.get("build_path", "/home/fins/UnderwaterSim/Code/UnityProject/RLChase/build/RLChase.x86_64"),
        side_channels=[config_channel],
        no_graphics=True,
        seed=seed + worker_id,
        worker_id=worker_id,
    )
    pz_env = UnityParallelEnv(unity_env)
    return UnityEnvWrapper(pz_env)


def get_unity_env_eval(seed, build_path: str, max_steps: int):
    """Create Unity environment for evaluation."""
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=1.0)
    unity_env = UnityEnvironment(
        file_name=build_path,
        side_channels=[config_channel],
        no_graphics=True,
        seed=seed,
        worker_id=100,
    )
    return UnityEnvWrapper(UnityParallelEnv(unity_env), max_steps=max_steps)


class CloudpickleWrapper:
    """Uses cloudpickle to serialize contents for multiprocessing."""

    def __init__(self, env):
        self.env = env

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.env)

    def __setstate__(self, env):
        import pickle
        self.env = pickle.loads(env)


def env_worker(conn, env_config, seed):
    """Environment worker process."""
    env = None
    try:
        while True:
            task, content = conn.recv()
            if task == "create_env":
                env = get_unity_env(env_config, seed)
                conn.send({"status": "created"})
            elif task == "reset":
                if env is None:
                    conn.send({"error": "Environment not created"})
                    continue
                obs = env.reset(seed=random.randint(0, 100000))
                state = env.get_state()
                conn.send({"obs": obs, "state": state})
            elif task == "step":
                if env is None:
                    conn.send({"error": "Environment not created"})
                    continue
                next_obs, reward, done, truncated, infos = env.step(content)
                state = env.get_state()
                conn.send({
                    "next_obs": next_obs,
                    "reward": reward,
                    "done": done,
                    "truncated": truncated,
                    "infos": infos,
                    "next_state": state,
                })
            elif task == "close":
                if env is not None:
                    env.close()
                conn.close()
                break
    except Exception as e:
        print(f"Error in env_worker: {e}")
        import traceback
        traceback.print_exc()
        if env is not None:
            env.close()
        conn.close()


def main():
    args = Args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    # Create environment connections
    conns = [Pipe() for _ in range(args.batch_size)]
    mappo_conns, env_conns = zip(*conns)

    # Start worker processes
    processes = [
        Process(target=env_worker, args=(
            env_conns[i],
            {"worker_index": i, "build_path": args.unity_build_path},
            args.seed + i
        ))
        for i in range(args.batch_size)
    ]
    for process in processes:
        process.daemon = True
        process.start()

    # Create environments in workers
    print("Creating environments in worker processes...")
    for mappo_conn in mappo_conns:
        mappo_conn.send(("create_env", None))
    for i, mappo_conn in enumerate(mappo_conns):
        response = mappo_conn.recv()
        if "error" in response:
            raise RuntimeError(f"Failed to create environment {i}: {response['error']}")
        print(f"Environment {i} created successfully")

    # Create eval environment
    eval_env = get_unity_env_eval(args.seed, args.unity_build_path, args.max_steps)

    obs_size = eval_env.get_obs_size()
    action_size = eval_env.get_action_size()
    n_agents = eval_env.n_agents
    state_size = eval_env.get_state_size()

    print(f"obs_size: {obs_size}, action_size: {action_size}, n_agents: {n_agents}, state_size: {state_size}")
    print(f"Max steps per episode: {args.max_steps}")

    # Initialize MAPPO
    config = MAPPOConfig(
        actor_hidden_dim=args.actor_hidden_dim,
        actor_num_layers=args.actor_num_layers,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        optimizer=args.optimizer,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        normalize_advantage=args.normalize_advantage,
        normalize_return=args.normalize_return,
        epochs=args.epochs,
        ppo_clip=args.ppo_clip,
        entropy_coef=args.entropy_coef,
        clip_gradients=args.clip_gradients,
        device=args.device,
    )
    mappo = MAPPO(
        obs_dim=obs_size,
        action_dim=action_size,
        state_dim=state_size,
        n_agents=n_agents,
        config=config,
    )

    # Setup logging
    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"3chase1_unity__{time_token}"
    if args.use_wnb:
        import wandb
        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MAPPO-{run_name}",
        )
    writer = SummaryWriter(f"runs/MAPPO-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Rollout buffer
    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=obs_size,
        state_space=state_size,
        action_space=action_size,
        num_agents=n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
    )

    ep_rewards = []
    ep_lengths = []
    training_step = 0
    num_episodes = 0
    step = 0
    pbar = tqdm(total=args.total_timesteps, desc="Training Progress")

    while step < args.total_timesteps:
        episodes = [
            {
                "obs": [],
                "actions": [],
                "log_prob": [],
                "reward": [],
                "states": [],
                "done": [],
            }
            for _ in range(args.batch_size)
        ]

        # Reset environments
        for mappo_conn in mappo_conns:
            mappo_conn.send(("reset", None))
        contents = [mappo_conn.recv() for mappo_conn in mappo_conns]
        obs = np.stack([content["obs"] for content in contents], axis=0)
        state = np.stack([content["state"] for content in contents])
        alive_envs = list(range(args.batch_size))
        ep_reward = [0] * args.batch_size
        ep_length = [0] * args.batch_size

        # Collect rollouts
        while len(alive_envs) > 0:
            with torch.no_grad():
                actions, log_probs = mappo.get_action(obs)
            for i, j in enumerate(alive_envs):
                mappo_conns[j].send(("step", actions[i]))
            contents = [mappo_conns[i].recv() for i in alive_envs]
            next_obs = [content["next_obs"] for content in contents]
            reward = [content["reward"] for content in contents]
            done = [content["done"] for content in contents]
            truncated = [content["truncated"] for content in contents]
            next_state = [content["next_state"] for content in contents]

            for i, j in enumerate(alive_envs):
                episodes[j]["obs"].append(obs[i])
                episodes[j]["actions"].append(actions[i])
                episodes[j]["log_prob"].append(log_probs[i])
                episodes[j]["reward"].append(reward[i])
                episodes[j]["states"].append(state[i])
                episodes[j]["done"].append(done[i])
                ep_reward[j] += reward[i]
                ep_length[j] += 1

            step += len(alive_envs)
            obs = []
            state = []
            for i, j in enumerate(alive_envs[:]):
                if done[i] or truncated[i]:
                    alive_envs.remove(j)
                    rb.add(episodes[j])
                    episodes[j] = dict()
                else:
                    obs.append(next_obs[i])
                    state.append(next_state[i])
            if obs:
                obs = np.stack(obs, axis=0)
                state = np.stack(state, axis=0)

        # Update progress bar
        pbar.update(args.batch_size * np.mean(ep_length))
        pbar.set_postfix({'reward': f"{np.mean(ep_reward):.2f}", 'episodes': num_episodes})

        ep_rewards.extend(ep_reward)
        ep_lengths.extend(ep_length)
        num_episodes += args.batch_size

        # Logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/num_episodes", num_episodes, step)
            ep_rewards = []
            ep_lengths = []

        # Get batch and train
        batch = rb.get_batch()
        if batch is None:
            continue
        b_obs, b_actions, b_log_probs, b_reward, b_states, b_done, b_mask = batch

        train_metrics = mappo.evaluate_actions(
            b_obs, b_actions, b_log_probs, b_reward, b_states, b_done, b_mask
        )

        writer.add_scalar("train/critic_loss", train_metrics["critic_loss"], step)
        writer.add_scalar("train/actor_loss", train_metrics["actor_loss"], step)
        writer.add_scalar("train/entropy", train_metrics["entropy"], step)
        writer.add_scalar("train/kl_divergence", train_metrics["kl_divergence"], step)
        writer.add_scalar("train/clipped_ratios", train_metrics["clipped_ratio"], step)
        writer.add_scalar("train/actor_gradients", train_metrics["actor_gradient"], step)
        writer.add_scalar("train/critic_gradients", train_metrics["critic_gradient"], step)
        writer.add_scalar("train/num_updates", training_step, step)
        training_step += 1

        # Evaluation
        if training_step % args.eval_steps == 0:
            eval_obs = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            current_reward = 0
            current_ep_length = 0
            eval_pbar = tqdm(total=args.num_eval_ep, desc="Evaluation", leave=False)

            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    actions, _ = mappo.get_action(eval_obs)
                next_obs_, reward, done, truncated, infos = eval_env.step(actions)
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep += 1
                    eval_pbar.update(1)

            eval_pbar.close()
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)

    # Cleanup
    pbar.close()
    writer.close()
    if args.use_wnb:
        wandb.finish()
    eval_env.close()
    for conn in mappo_conns:
        conn.send(("close", None))
    for process in processes:
        process.join()

    print("Training completed!")


if __name__ == "__main__":
    main()