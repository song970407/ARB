# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")


@dataclass
class TrainConfig:
    device: str = "cuda"
    log_freq: int = 100
    """
    # Config for Locomotion environments
    # Experiment
    env: str = "hopper-random-v2"  # OpenAI gym environment name
    # env: str = "hopper-medium-replay-v2"  # OpenAI gym environment name
    # env: str = "hopper-medium-v2"  # OpenAI gym environment name
    # env: str = "hopper-medium-expert-v2"  # OpenAI gym environment name
    # env: str = "halfcheetah-random-v2"  # OpenAI gym environment name
    # env: str = "halfcheetah-medium-replay-v2"  # OpenAI gym environment name
    # env: str = "halfcheetah-medium-v2"  # OpenAI gym environment name
    # env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    # env: str = "walker2d-random-v2"  # OpenAI gym environment name
    # env: str = "walker2d-medium-replay-v2"  # OpenAI gym environment name
    # env: str = "walker2d-medium-v2"  # OpenAI gym environment name
    # env: str = "walker2d-medium-expert-v2"  # OpenAI gym environment name

    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed
    eval_freq: int = int(5e4)  # How often (time steps) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Number of offline updates
    # online_iterations: int = int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] = "checkpoints/finetune"  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    actor_dropout: float = 0.0  # Dropout in actor network
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    expl_noise: float = 0.03  # Std of Gaussian exploration noise
    noise_clip: float = 0.5  # Range to clip noise
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    # PEX
    inverse_alpha: float = 3.0  # Inverse temperature for PEX
    """

    # Config for Antmaze environments
    # Experiment
    env: str = "antmaze-umaze-v2"
    # env: str = "antmaze-umaze-diverse-v2"
    # env: str = "antmaze-medium-play-v2"
    # env: str = "antmaze-medium-diverse-v2"
    # env: str = "antmaze-large-play-v2"
    # env: str = "antmaze-large-diverse-v2"

    seed: int = 300  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 300  # Eval environment seed
    model_update_freq: int = 1000  # How often (time steps) we update the model
    eval_freq: int = int(5e4)  # How often (time steps) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    # offline_iterations: int = int(1e3)  # Number of offline updates
    online_iterations: int = int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] = "checkpoints/finetune"  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    actor_dropout: float = 0.0  # Dropout in actor network
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = (
        10.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    )
    iql_tau: float = 0.9  # Coefficient for asymmetric loss
    expl_noise: float = 0.03  # Std of Gaussian exploration noise
    noise_clip: float = 0.5  # Range to clip noise
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    # PEX
    inverse_alpha: float = 10.0  # Inverse temperature for PEX

    # Config for Adroit environments
    # Experiment
    """
    device: str = "cuda"
    env: str = "relocate-cloned-v1"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed
    eval_freq: int = int(5e4)  # How often (time steps) we evaluate
    n_episodes: int = 100  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Number of offline updates
    # online_iterations: int = int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] = "checkpoints/pretrain"  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    actor_dropout: float = 0.1  # Dropout in actor network
    buffer_size: int = 1_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.8  # Coefficient for asymmetric loss
    expl_noise: float = 0.03  # Std of Gaussian exploration noise
    noise_clip: float = 0.5  # Range to clip noise
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    """

    # Replay Buffer
    # replay_buffer: str = "naive"
    # replay_buffer: str = "parallel"
    replay_buffer: str = "top_n"
    # replay_buffer: str = "adaptive_traj"

    # Parallel Replay Buffer Hyperparameters
    mixing_ratio: float = 0.5
    # Top N Replay Buffer Hyperparameters
    top_n: int = 50000
    # Adaptive Replay Buffer Hyperparameters
    weight_update_freq: int = 1000
    # weight_alpha: float = 0.1
    # weight_alpha: float = 0.2
    weight_alpha: float = 0.5
    # weight_alpha: float = 1.0
    # weight_alpha: float = 2.0
    # weight_alpha: float = 5.0
    log_weight_min: float = -12.0
    log_weight_max: float = 7.0

    # Wandb logging
    project: str = "ARB"
    group: str = "PEX-D4RL"
    name: str = "PEX"

    def __post_init__(self):
        self.load_model = (
            f"checkpoints/pretrain/{self.name}-{self.env}-{self.seed}/model.pt"
        )
        # self.name = f"{self.name}-{self.env}-{self.replay_buffer}-{self.seed}-{str(uuid.uuid4())[:8]}"
        if self.replay_buffer == "adaptive" or self.replay_buffer == "adaptive_traj":
            self.name = f"{self.name}-{self.env}-{self.seed}-{self.replay_buffer}-{self.weight_alpha}-{str(uuid.uuid4())[:8]}"
        else:
            self.name = f"{self.name}-{self.env}-{self.seed}-{self.replay_buffer}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        self.name = f"finetune-{self.name}"


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        self._offline_size = n_transitions

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones], indices

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def switch_online(self):
        pass

    def update_sample_weight(self, actor):
        pass


class ReplayBufferParallel(ReplayBuffer):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        mixing_ratio: float,
        device: str = "cpu",
    ):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._buffer_size = buffer_size
        self._mixing_ratio = mixing_ratio
        self._offline_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            device=device,
        )
        self._online_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            device=device,
        )
        self._device = device

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        self._offline_buffer.load_d4rl_dataset(data)
        self._offline_size = self._offline_buffer._size

    def sample(self, batch_size: int) -> TensorBatch:
        if self._online_buffer._size == 0:  # only offline data
            return self._offline_buffer.sample(batch_size)
        else:
            offline_batch_size = int(batch_size * self._mixing_ratio)
            online_batch_size = batch_size - offline_batch_size
            offline_batch, offline_indices = self._offline_buffer.sample(
                offline_batch_size
            )
            online_batch, online_indices = self._online_buffer.sample(online_batch_size)
            online_indices = online_indices + self._offline_size
            states = torch.cat([offline_batch[0], online_batch[0]], dim=0)
            actions = torch.cat([offline_batch[1], online_batch[1]], dim=0)
            rewards = torch.cat([offline_batch[2], online_batch[2]], dim=0)
            next_states = torch.cat([offline_batch[3], online_batch[3]], dim=0)
            dones = torch.cat([offline_batch[4], online_batch[4]], dim=0)
            indices = np.concatenate([offline_indices, online_indices])
            return [states, actions, rewards, next_states, dones], indices

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self._online_buffer.add_transition(state, action, reward, next_state, done)


class ReplayBufferTopN(ReplayBuffer):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        top_n: int,
        device: str = "cpu",
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            device=device,
        )
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._top_n = top_n
        self._top_indices = []

    def switch_online(self):  # keep only the top n episodes
        episode_start_idx_list = [0]
        states = self._states.cpu().numpy()
        next_states = self._next_states.cpu().numpy()
        dones = self._dones.cpu().numpy()
        for i in range(1, self._size):
            if np.linalg.norm(next_states[i - 1] - states[i]) > 1e-6 or dones[i - 1]:
                episode_start_idx_list.append(i)
        episode_start_idx_list.append(self._size)  # Add the last index

        sum_reward_list = []
        for i in range(len(episode_start_idx_list) - 1):
            start_idx = episode_start_idx_list[i]
            end_idx = episode_start_idx_list[i + 1]
            sum_reward_list.append(self._rewards[start_idx:end_idx].sum().item())
        sorted_idx = np.argsort(sum_reward_list)[::-1][: self._top_n]
        idx = []
        number_of_transitions = 0
        for i in sorted_idx:
            if number_of_transitions > self._top_n:
                break
            number_of_transitions += (
                episode_start_idx_list[i + 1] - episode_start_idx_list[i]
            )
            idx += list(range(episode_start_idx_list[i], episode_start_idx_list[i + 1]))
        states = self._states[idx]
        actions = self._actions[idx]
        rewards = self._rewards[idx]
        next_states = self._next_states[idx]
        dones = self._dones[idx]
        print(f"Number of total episodes: {len(episode_start_idx_list) - 1}")
        print(f"Number of episodes remained: {i}")
        self._states = torch.zeros(
            (self._buffer_size, self._state_dim),
            dtype=torch.float32,
            device=self._device,
        )
        self._actions = torch.zeros(
            (self._buffer_size, self._action_dim),
            dtype=torch.float32,
            device=self._device,
        )
        self._rewards = torch.zeros(
            (self._buffer_size, 1), dtype=torch.float32, device=self._device
        )
        self._next_states = torch.zeros(
            (self._buffer_size, self._state_dim),
            dtype=torch.float32,
            device=self._device,
        )
        self._dones = torch.zeros(
            (self._buffer_size, 1), dtype=torch.float32, device=self._device
        )
        self._states[: len(idx)] = states
        self._actions[: len(idx)] = actions
        self._rewards[: len(idx)] = rewards
        self._next_states[: len(idx)] = next_states
        self._dones[: len(idx)] = dones
        self._pointer = len(idx) % self._buffer_size
        self._size = min(len(idx), self._buffer_size)
        self._offline_size = len(idx)


class ReplayBufferAdaptive(ReplayBuffer):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        weight_alpha: float,
        log_weight_min: float,
        log_weight_max: float,
        device: str = "cpu",
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            device=device,
        )
        self._action_dim = action_dim
        self._weight_alpha = weight_alpha
        self._log_weight_min = log_weight_min
        self._log_weight_max = log_weight_max
        self._weight = None

    @torch.no_grad()
    def update_sample_weight(self, actor):
        self._weight = np.zeros(self._buffer_size, dtype=np.float32)
        actor.eval().cpu()
        # Update weights of offline data
        states = self._states[: self._size].detach().cpu()
        actions = self._actions[: self._size].detach().cpu()
        log_prob = torch.clamp(
            actor.log_prob_standard(states, actions) / self._action_dim,
            min=self._log_weight_min,
            max=self._log_weight_max,
        )
        log_prob = log_prob - torch.max(log_prob)  # for numerical stability
        self._weight[: self._size] = torch.exp(log_prob / self._weight_alpha).numpy()
        actor.train().to(self._device)

    def sample(self, batch_size: int) -> TensorBatch:
        if self._weight is None:
            indices = np.random.randint(0, self._size, size=batch_size)
        else:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                # replace=False,
                p=self._weight[: self._size] / np.sum(self._weight[: self._size]),
            )
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones], indices

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._weight[self._pointer] = 1.0

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def switch_online(self):
        self._weight = np.zeros(self._buffer_size, dtype=np.float32)


class ReplayBufferAdaptiveTrajectory(ReplayBuffer):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        weight_alpha: float,
        log_weight_min: float,
        log_weight_max: float,
        device: str = "cpu",
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=buffer_size,
            device=device,
        )
        self._action_dim = action_dim
        self._weight_alpha = weight_alpha
        self._log_weight_min = log_weight_min
        self._log_weight_max = log_weight_max
        self._weight = None

    @torch.no_grad()
    def update_sample_weight(self, actor):
        episode_start_idx_list = [0]
        states = self._states[: self._size].cpu().numpy()
        next_states = self._next_states[: self._size].cpu().numpy()
        dones = self._dones[: self._size].cpu().numpy()
        for i in range(1, self._size):
            if np.linalg.norm(next_states[i - 1] - states[i]) > 1e-6 or dones[i - 1]:
                # if dones[i - 1]:
                # if np.linalg.norm(next_states[i - 1] - states[i]) > 1e-6:
                episode_start_idx_list.append(i)
        episode_start_idx_list.append(self._size)  # Add the last index

        self._weight = np.zeros(self._buffer_size, dtype=np.float32)
        actor.eval().cpu()

        states = self._states[: self._size].detach().cpu()
        actions = self._actions[: self._size].detach().cpu()
        log_prob = torch.clamp(
            actor.log_prob_standard(states, actions) / self._action_dim,
            min=self._log_weight_min,
            max=self._log_weight_max,
        )
        log_prob = log_prob - torch.max(log_prob)
        episode_length_list = []
        episode_log_prob_list = []
        for i in range(len(episode_start_idx_list) - 1):
            start_idx = episode_start_idx_list[i]
            end_idx = episode_start_idx_list[i + 1]
            log_prob[start_idx:end_idx] = log_prob[start_idx:end_idx].mean().item()
            episode_length_list.append(end_idx - start_idx)
            episode_log_prob_list.append(log_prob[start_idx:end_idx].mean().item())
        """print("First 10")
        print(episode_log_prob_list[:10])
        print("Last 10")
        print(episode_log_prob_list[-10:])"""
        self._weight[: self._size] = torch.exp(log_prob / self._weight_alpha).numpy()
        actor.train().to(self._device)

    def sample(self, batch_size: int) -> TensorBatch:
        if self._weight is None:
            indices = np.random.randint(0, self._size, size=batch_size)
        else:
            indices = np.random.choice(
                self._size,
                size=batch_size,
                # replace=False,
                p=self._weight[: self._size] / np.sum(self._weight[: self._size]),
            )
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones], indices

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._weight[self._pointer] = 1.0

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def switch_online(self):
        self._weight = np.zeros(self._buffer_size, dtype=np.float32)


def set_env_seed(env: Optional[gym.Env], seed: int):
    env.seed(seed)
    env.action_space.seed(seed)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    # wandb.run.save()


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor_offline: nn.Module,
    actor_online: nn.Module,
    q_network: nn.Module,
    config,
    max_action: float,
    n_episodes: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    env.seed(seed)
    actor_offline.eval()
    actor_online.eval()
    q_network.eval()
    episode_rewards = []
    successes = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            action = composite_actor(
                state,
                q_network,
                actor_offline,
                actor_online,
                config=config,
                max_action=max_action,
            )
            state, reward, done, env_infos = env.step(action)
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)

    actor_offline.train()
    actor_online.train()
    q_network.train()
    return np.asarray(episode_rewards), np.mean(successes)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def log_prob_standard(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        mean = self.net(obs)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        action_distribution = Normal(mean, std)
        return torch.sum(action_distribution.log_prob(actions) + log_std, dim=-1)

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(
                self(state) * self.max_action, -self.max_action, self.max_action
            )
            .cpu()
            .data.numpy()
            .flatten()
        )


def composite_actor(obs, q_network, actor_offline, actor_online, config, max_action):
    torch_obs = torch.tensor(
        obs.reshape(1, -1), device=config.device, dtype=torch.float32
    )
    offline_action = actor_offline(torch_obs)
    online_action = actor_online(torch_obs)
    if not config.iql_deterministic:
        offline_action = offline_action.sample()
        online_action = online_action.sample()
    else:
        noise = (torch.randn_like(offline_action) * config.expl_noise).clamp(
            -config.noise_clip, config.noise_clip
        )
        offline_action += noise
        noise = (torch.randn_like(online_action) * config.expl_noise).clamp(
            -config.noise_clip, config.noise_clip
        )
        online_action += noise
    offline_action = (
        torch.clamp(offline_action * max_action, -max_action, max_action)
        .cpu()
        .data.numpy()
        .flatten()
    )
    online_action = (
        torch.clamp(online_action * max_action, -max_action, max_action)
        .cpu()
        .data.numpy()
        .flatten()
    )

    torch_offline_action = torch.tensor(
        offline_action.reshape(1, -1), device=config.device, dtype=torch.float32
    )
    torch_online_action = torch.tensor(
        online_action.reshape(1, -1), device=config.device, dtype=torch.float32
    )
    prob_offline = (
        q_network(torch_obs, torch_offline_action) * config.inverse_alpha
    ).exp()
    prob_online = (
        q_network(torch_obs, torch_online_action) * config.inverse_alpha
    ).exp()
    prob_offline = prob_offline / (prob_offline + prob_online + 1e-6)
    if torch.rand(1).item() < prob_offline:
        return offline_action
    else:
        return online_action


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class PEX:
    def __init__(
        self,
        max_action: float,
        actor_offline: nn.Module,
        actor_online: nn.Module,
        actor_offline_optimizer: torch.optim.Optimizer,
        actor_online_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor_offline = actor_offline
        self.actor_online = actor_online
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_offline_optimizer = actor_offline_optimizer
        self.actor_online_optimizer = actor_online_optimizer
        self.actor_offline_lr_schedule = CosineAnnealingLR(
            self.actor_offline_optimizer, max_steps
        )
        self.actor_online_lr_schedule = CosineAnnealingLR(
            self.actor_online_optimizer, max_steps
        )
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy_offline(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor_offline(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_offline_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_offline_optimizer.step()
        self.actor_offline_lr_schedule.step()
        # Copy offline actor online policy
        self.actor_online.load_state_dict(self.actor_offline.state_dict())

    def _update_policy_online(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor_online(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_online_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_online_optimizer.step()
        self.actor_online_lr_schedule.step()

    def train_offline(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy_offline(adv, observations, actions, log_dict)

        return log_dict

    def train_online(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update offline actor
        self._update_policy_online(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor_offline": self.actor_offline.state_dict(),
            "actor_offline_optimizer": self.actor_offline_optimizer.state_dict(),
            "actor_offline_lr_schedule": self.actor_offline_lr_schedule.state_dict(),
            "actor_online": self.actor_online.state_dict(),
            "actor_online_optimizer": self.actor_online_optimizer.state_dict(),
            "actor_online_lr_schedule": self.actor_online_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor_offline.load_state_dict(state_dict["actor_offline"])
        self.actor_offline_optimizer.load_state_dict(
            state_dict["actor_offline_optimizer"]
        )
        self.actor_offline_lr_schedule.load_state_dict(
            state_dict["actor_offline_lr_schedule"]
        )

        self.actor_online.load_state_dict(state_dict["actor_online"])
        self.actor_online_optimizer.load_state_dict(
            state_dict["actor_online_optimizer"]
        )
        self.actor_online_lr_schedule.load_state_dict(
            state_dict["actor_online_lr_schedule"]
        )

        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: TrainConfig):
    env = gym.make(config.env)
    eval_env = gym.make(config.env)

    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    max_steps = env._max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    if config.replay_buffer == "naive":
        replay_buffer = ReplayBuffer(
            state_dim, action_dim, config.buffer_size, config.device
        )
    elif config.replay_buffer == "parallel":
        replay_buffer = ReplayBufferParallel(
            state_dim,
            action_dim,
            config.buffer_size,
            config.mixing_ratio,
            config.device,
        )
    elif config.replay_buffer == "top_n":
        replay_buffer = ReplayBufferTopN(
            state_dim,
            action_dim,
            config.buffer_size,
            config.top_n,
            config.device,
        )
    elif config.replay_buffer == "adaptive":
        replay_buffer = ReplayBufferAdaptive(
            state_dim,
            action_dim,
            config.buffer_size,
            config.weight_alpha,
            config.log_weight_min,
            config.log_weight_max,
            config.device,
        )
    elif config.replay_buffer == "adaptive_traj":
        replay_buffer = ReplayBufferAdaptiveTrajectory(
            state_dim,
            action_dim,
            config.buffer_size,
            config.weight_alpha,
            config.log_weight_min,
            config.log_weight_max,
            config.device,
        )

    else:
        raise ValueError(f"Unknown replay buffer type: {config.replay_buffer}")
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor_offline = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    actor_online = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_offline_optimizer = torch.optim.Adam(
        actor_offline.parameters(), lr=config.actor_lr
    )
    actor_online_optimizer = torch.optim.Adam(
        actor_online.parameters(), lr=config.actor_lr
    )

    kwargs = {
        "max_action": max_action,
        "actor_offline": actor_offline,
        "actor_offline_optimizer": actor_offline_optimizer,
        "actor_online": actor_online,
        "actor_online_optimizer": actor_online_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.online_iterations,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = PEX(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file, map_location=config.device))
        actor_offline = trainer.actor_offline
        actor_online = trainer.actor_online

    wandb_init(asdict(config))

    evaluations = []

    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    print("Online tuning")
    # trainer.switch_calibration()
    # trainer.cql_alpha = config.cql_alpha_online
    replay_buffer.switch_online()  # Switch to online mode
    for t in range(int(config.online_iterations)):
        online_log = {}

        if t % config.model_update_freq == 0:
            for _ in range(config.model_update_freq):
                episode_step += 1
                action = composite_actor(
                    state,
                    q_network,
                    actor_offline,
                    actor_online,
                    config=config,
                    max_action=max_action,
                )
                # action = action.cpu().data.numpy().flatten()
                next_state, reward, done, env_infos = env.step(action)

                if not goal_achieved:
                    goal_achieved = is_goal_reached(reward, env_infos)
                episode_return += reward
                real_done = False  # Episode can timeout which is different from done
                if done and episode_step < max_steps:
                    real_done = True

                if config.normalize_reward:
                    reward = modify_reward_online(reward, config.env, **reward_mod_dict)

                replay_buffer.add_transition(
                    state, action, reward, next_state, real_done
                )

                # online_buffer.add_transition(state, action, reward, next_state, real_done)

                state = next_state

                if done:
                    state, done = env.reset(), False
                    # Valid only for envs with goal, e.g. AntMaze, Adroit
                    if is_env_with_goal:
                        train_successes.append(goal_achieved)

                        online_log["train/regret"] = np.mean(
                            1 - np.array(train_successes)
                        )
                        online_log["train/is_success"] = float(goal_achieved)
                    online_log["train/episode_return"] = episode_return
                    normalized_return = eval_env.get_normalized_score(episode_return)
                    online_log["train/d4rl_normalized_episode_return"] = (
                        normalized_return * 100.0
                    )
                    online_log["train/episode_length"] = episode_step
                    episode_return = 0
                    episode_step = 0
                    goal_achieved = False

        if t % config.weight_update_freq == 0:
            replay_buffer.update_sample_weight(trainer.actor_online)

        batch, indices = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]

        log_dict = trainer.train_online(batch)
        log_dict["online_iter"] = t
        is_online_data = indices >= replay_buffer._offline_size
        log_dict["online_data_ratio"] = np.mean(is_online_data)
        log_dict.update(online_log)

        if (t + 1) % config.log_freq == 0:
            wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor_offline,
                actor_online,
                q_network,
                config=config,
                max_action=max_action,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            eval_log = {}
            normalized = eval_env.get_normalized_score(np.mean(eval_scores))
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(eval_successes))
                eval_log["eval/success_rate"] = success_rate
            normalized_eval_score = normalized * 100.0
            eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            """if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )"""
            wandb.log(eval_log, step=trainer.total_it)
    if config.checkpoints_path:
        torch.save(
            trainer.state_dict(), os.path.join(config.checkpoints_path, "model.pt")
        )


if __name__ == "__main__":
    train()
