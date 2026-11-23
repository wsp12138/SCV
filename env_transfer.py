"""SCV environment adapter for the archived transfer tasks."""

from __future__ import annotations
from dataclasses import dataclass, replace
import importlib
from typing import Dict, List, Mapping, Sequence, Tuple
import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    family: str
    title: str
    gym_id: str | None = None
    module: str | None = None
    default_max_steps: int = 25
    controlled_scenario: str | None = None


STANDARD_BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "lbf/8x8-2p-2f-2s-coop": BenchmarkSpec(
        name="lbf/8x8-2p-2f-2s-coop",
        family="lbf",
        title="Level-Based Foraging 8x8 2 players 2 foods sight 2 coop",
        gym_id="Foraging-2s-8x8-2p-2f-coop-v3",
        default_max_steps=50,
    ),
    "lbf/10x10-3p-3f-2s": BenchmarkSpec(
        name="lbf/10x10-3p-3f-2s",
        family="lbf",
        title="Level-Based Foraging 10x10 3 players 3 foods sight 2",
        gym_id="Foraging-2s-10x10-3p-3f-v3",
        default_max_steps=50,
    ),
    "lbf/15x15-4p-5f": BenchmarkSpec(
        name="lbf/15x15-4p-5f",
        family="lbf",
        title="Level-Based Foraging 15x15 4 players 5 foods",
        gym_id="Foraging-15x15-4p-5f-v3",
        default_max_steps=50,
    ),
    "rware/tiny-4ag": BenchmarkSpec(
        name="rware/tiny-4ag",
        family="rware",
        title="RWARE Tiny 4 agents",
        gym_id="rware-tiny-4ag-v2",
        default_max_steps=500,
    ),
}


@dataclass(frozen=True)
class StepData:
    observations: List[np.ndarray]
    rewards: List[float]
    team_reward: float
    raw_team_reward: float
    done: bool
    infos: Mapping[str, object]


class StandardMARLEnv:
    """Uniform wrapper around a homogeneous discrete-action MARL benchmark."""

    def __init__(self, name: str, max_steps: int | None = None) -> None:
        if name not in STANDARD_BENCHMARKS:
            known = ", ".join(ordered_benchmark_names())
            raise ValueError(f"unknown benchmark {name!r}; known benchmarks: {known}")
        base_spec = STANDARD_BENCHMARKS[name]
        if max_steps is not None:
            base_spec = replace(base_spec, default_max_steps=max_steps)
        self.spec = base_spec
        self.name = base_spec.name
        self.family = base_spec.family
        self.title = base_spec.title
        self.max_steps = base_spec.default_max_steps
        self._env = None
        self.agents: List[str] = []
        self.n_agents = 0
        self.obs_dim = 0
        self.n_actions = 0
        self.t = 0

    def reset(self, seed: int) -> List[np.ndarray]:
        if self._env is None:
            self._env = self._make_env()
        self.t = 0
        (raw_obs, _info) = self._env.reset(seed=seed)
        raw_obs = self._apply_controlled_scenario(raw_obs, seed)
        observations = [as_float_array(obs) for obs in list(raw_obs)]
        self.agents = [f"agent_{idx}" for idx in range(len(observations))]
        action_sizes = [int(space.n) for space in self._env.action_space.spaces]
        self._set_spaces(observations, action_sizes)
        return observations

    def step(self, actions: Sequence[int]) -> StepData:
        if self._env is None:
            raise RuntimeError("environment must be reset before step")
        self.t += 1
        (raw_obs, rewards, terminated, truncated, info) = self._env.step(
            tuple((int(a) for a in actions))
        )
        observations = [as_float_array(obs) for obs in list(raw_obs)]
        reward_values = [float(value) for value in list(rewards)]
        done = truthy_done(terminated) or truthy_done(truncated) or self.t >= self.max_steps
        return StepData(
            observations=observations,
            rewards=reward_values,
            team_reward=mean(reward_values),
            raw_team_reward=float(sum(reward_values)),
            done=done,
            infos={"raw_info": info},
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _make_env(self):
        if self.spec.gym_id is None:
            raise ValueError(f"Gym benchmark {self.name} has no gym id")
        if self.spec.family == "lbf":
            importlib.import_module("lbforaging")
            ensure_custom_lbf_env(self.spec.gym_id)
        elif self.spec.family == "rware":
            importlib.import_module("rware")
        return gym.make(self.spec.gym_id)

    def _apply_controlled_scenario(self, raw_obs, seed: int):
        return raw_obs

    def _set_spaces(self, observations: Sequence[np.ndarray], action_sizes: Sequence[int]) -> None:
        if not observations:
            raise ValueError(f"{self.name} produced no agents")
        obs_dims = [int(obs.reshape(-1).shape[0]) for obs in observations]
        if len(set(obs_dims)) != 1:
            raise ValueError(
                f"{self.name} has heterogeneous observations {obs_dims}; this runner currently expects homogeneous agents"
            )
        if len(set(action_sizes)) != 1:
            raise ValueError(
                f"{self.name} has heterogeneous action spaces {action_sizes}; this runner currently expects homogeneous discrete actions"
            )
        self.n_agents = len(observations)
        self.obs_dim = obs_dims[0]
        self.n_actions = int(action_sizes[0])


def make_standard_marl_env(name: str, max_steps: int | None = None) -> StandardMARLEnv:
    return StandardMARLEnv(name=name, max_steps=max_steps)


def ordered_benchmark_names() -> List[str]:
    return list(STANDARD_BENCHMARKS)


def ensure_custom_lbf_env(gym_id: str | None) -> None:
    if gym_id == "Foraging-8x8-3p-2f-4l-v3":
        if gym_id not in gym.envs.registry:
            gym.register(
                id=gym_id,
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "players": 3,
                    "min_player_level": 1,
                    "max_player_level": 4,
                    "field_size": (8, 8),
                    "min_food_level": 1,
                    "max_food_level": 4,
                    "max_num_food": 2,
                    "sight": 8,
                    "max_episode_steps": 50,
                    "force_coop": False,
                    "grid_observation": False,
                    "penalty": 0.0,
                },
            )
        return
    custom_player_food = {"Foraging-15x15-3p-5f-v3": (3, 5), "Foraging-15x15-4p-5f-v3": (4, 5)}
    if gym_id not in custom_player_food:
        return
    if gym_id in gym.envs.registry:
        return
    (players, foods) = custom_player_food[gym_id]
    gym.register(
        id=gym_id,
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": players,
            "min_player_level": 1,
            "max_player_level": 2,
            "field_size": (15, 15),
            "min_food_level": 1,
            "max_food_level": None,
            "max_num_food": foods,
            "sight": 15,
            "max_episode_steps": 50,
            "force_coop": False,
            "grid_observation": False,
            "penalty": 0.0,
        },
    )


def as_float_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def truthy_done(value: object) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = list(value)
        return bool(vals) and all((bool(item) for item in vals))
    if isinstance(value, dict):
        vals = list(value.values())
        return bool(vals) and all((bool(item) for item in vals))
    return bool(value)
