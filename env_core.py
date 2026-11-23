"""SCV environment adapter for the archived core tasks."""

from __future__ import annotations
import copy
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
    layout: str | None = None
    default_max_steps: int = 25
    controlled_scenario: str | None = None
    pursuit_n_catch: int | None = None
    pursuit_n_pursuers: int = 8
    pursuit_n_evaders: int = 30
    magent_map_size: int | None = None
    magent_opponent_policy: str = "seeded_random"
    magent_zero_sum_events: bool = False


STANDARD_BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "pursuit/8p-30e-2catch": BenchmarkSpec(
        name="pursuit/8p-30e-2catch",
        family="pursuit",
        title="PettingZoo Pursuit 8 pursuers 30 evaders cooperative-2",
        module="pettingzoo.sisl.pursuit_v4",
        default_max_steps=500,
        pursuit_n_catch=2,
    ),
    "pursuit/16p-30e-2catch": BenchmarkSpec(
        name="pursuit/16p-30e-2catch",
        family="pursuit",
        title="PettingZoo Pursuit 16 pursuers 30 evaders cooperative-2",
        module="pettingzoo.sisl.pursuit_v4",
        default_max_steps=500,
        pursuit_n_catch=2,
        pursuit_n_pursuers=16,
        pursuit_n_evaders=30,
    ),
    "lbf/15x15-4p-3f": BenchmarkSpec(
        name="lbf/15x15-4p-3f",
        family="lbf",
        title="Level-Based Foraging 15x15 4 players 3 foods",
        gym_id="Foraging-15x15-4p-3f-v3",
        default_max_steps=50,
    ),
    "rware/small-4ag-easy": BenchmarkSpec(
        name="rware/small-4ag-easy",
        family="rware",
        title="RWARE Small 4 agents easy",
        gym_id="rware-small-4ag-easy-v2",
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
        self._episode_seed = 0

    def reset(self, seed: int) -> List[np.ndarray]:
        if self._env is None:
            self._env = self._make_env()
        self.t = 0
        if self.spec.family == "pursuit":
            (raw_obs, _infos) = self._env.reset(seed=seed)
            self.agents = list(self._env.agents)
            observations = [as_float_array(raw_obs[agent]) for agent in self.agents]
            action_sizes = [int(self._env.action_space(agent).n) for agent in self.agents]
        else:
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
        if self.spec.family == "pursuit":
            action_dict = {
                agent: int(actions[idx])
                for (idx, agent) in enumerate(self.agents)
                if agent in self._env.agents
            }
            (raw_obs, rewards, terminations, truncations, infos) = self._env.step(action_dict)
            observations = [
                as_float_array(raw_obs.get(agent, np.zeros(self.obs_dim, dtype=np.float32)))
                for agent in self.agents
            ]
            reward_values = [float(rewards.get(agent, 0.0)) for agent in self.agents]
            done = (
                all_bool(terminations.values())
                or all_bool(truncations.values())
                or (not self._env.agents)
            )
            done = done or self.t >= self.max_steps
            team_reward = mean(reward_values)
            return StepData(
                observations=observations,
                rewards=reward_values,
                team_reward=team_reward,
                raw_team_reward=float(sum(reward_values)),
                done=done,
                infos={"raw_infos": dict(infos)},
            )
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
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
            self._env = None

    def available_actions(self) -> List[np.ndarray]:
        if self._env is None:
            raise RuntimeError("environment must be reset before reading action availability")
        return [np.ones(self.n_actions, dtype=np.float32) for _ in range(self.n_agents)]

    def project_actions_to_available(self, actions: Sequence[int]) -> List[int]:
        projected: List[int] = []
        for action, available in zip(actions, self.available_actions()):
            action = int(action)
            if 0 <= action < available.size and float(available[action]) > 0.0:
                projected.append(action)
                continue
            valid = np.flatnonzero(available > 0.0)
            if valid.size == 0:
                raise RuntimeError("environment exposed an empty available-action set")
            projected.append(int(valid[0]))
        return projected

    def supports_state_snapshot(self) -> bool:
        return True

    def snapshot_state(self) -> Dict[str, object]:
        """Capture the dynamic simulator state needed to branch one action."""
        if self._env is None:
            raise RuntimeError("environment must be reset before snapshot")
        common = {
            "family": self.spec.family,
            "t": self.t,
            "agents": copy.deepcopy(self.agents),
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
        }
        if self.spec.family == "pursuit":
            (parallel, order, bounds, raw, base) = pursuit_env_chain(self._env)
            common["parallel"] = snapshot_object_dict(parallel, {"aec_env"})
            common["order"] = snapshot_object_dict(order, {"env"})
            common["bounds"] = snapshot_object_dict(bounds, {"env"})
            common["raw"] = snapshot_object_dict(raw, {"env"})
            common["base"] = snapshot_object_dict(base, set())
            return common
        if self.spec.family in {"lbf", "rware"}:
            common["env"] = copy.deepcopy(self._env)
            return common
        raise NotImplementedError(f"state snapshots are unavailable for {self.spec.family}")

    def restore_state(self, snapshot: Mapping[str, object]) -> None:
        """Restore a snapshot without rebuilding static planners or wrappers."""
        if self._env is None:
            raise RuntimeError("environment must be reset before restore")
        family = str(snapshot.get("family", ""))
        if family != self.spec.family:
            raise ValueError(f"snapshot family {family!r} does not match {self.spec.family!r}")
        self.t = int(snapshot["t"])
        self.agents = copy.deepcopy(snapshot["agents"])
        self.n_agents = int(snapshot["n_agents"])
        self.obs_dim = int(snapshot["obs_dim"])
        self.n_actions = int(snapshot["n_actions"])
        if family == "pursuit":
            (parallel, order, bounds, raw, base) = pursuit_env_chain(self._env)
            restore_object_dict(base, snapshot["base"])
            restore_object_dict(raw, snapshot["raw"], link_name="env", link_value=base)
            restore_object_dict(bounds, snapshot["bounds"], link_name="env", link_value=raw)
            restore_object_dict(order, snapshot["order"], link_name="env", link_value=bounds)
            restore_object_dict(
                parallel, snapshot["parallel"], link_name="aec_env", link_value=order
            )
            return
        if family in {"lbf", "rware"}:
            self._env = copy.deepcopy(snapshot["env"])
            return
        raise NotImplementedError(f"state snapshots are unavailable for {family}")

    def _make_env(self):
        if self.spec.family == "pursuit":
            if self.spec.module is None:
                raise ValueError(f"Pursuit benchmark {self.name} has no module")
            module = importlib.import_module(self.spec.module)
            return module.parallel_env(
                max_cycles=self.max_steps,
                x_size=16,
                y_size=16,
                shared_reward=True,
                n_evaders=int(self.spec.pursuit_n_evaders),
                n_pursuers=int(self.spec.pursuit_n_pursuers),
                obs_range=7,
                n_catch=int(self.spec.pursuit_n_catch or 2),
                freeze_evaders=False,
                tag_reward=0.01,
                catch_reward=5.0,
                urgency_reward=-0.1,
                surround=False,
                constraint_window=1.0,
            )
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


MAGENT_BATTLE_MOVE_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (0, -2),
    1: (-1, -1),
    2: (0, -1),
    3: (1, -1),
    4: (-2, 0),
    5: (-1, 0),
    6: (0, 0),
    7: (1, 0),
    8: (2, 0),
    9: (-1, 1),
    10: (0, 1),
    11: (1, 1),
    12: (0, 2),
}


def ordered_benchmark_names() -> List[str]:
    return list(STANDARD_BENCHMARKS)


def pursuit_env_chain(parallel_env: object) -> Tuple[object, object, object, object, object]:
    parallel = parallel_env
    order = getattr(parallel, "aec_env")
    bounds = getattr(order, "env")
    raw = getattr(bounds, "env")
    base = getattr(raw, "env")
    return (parallel, order, bounds, raw, base)


def snapshot_object_dict(value: object, excluded: set[str]) -> Dict[str, object]:
    return copy.deepcopy({key: item for (key, item) in vars(value).items() if key not in excluded})


def restore_object_dict(
    value: object,
    snapshot: object,
    *,
    link_name: str | None = None,
    link_value: object | None = None,
) -> None:
    if not isinstance(snapshot, Mapping):
        raise TypeError("object snapshot must be a mapping")
    value.__dict__.clear()
    value.__dict__.update(copy.deepcopy(dict(snapshot)))
    if link_name is not None:
        value.__dict__[link_name] = link_value


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


def all_bool(values: Sequence[object]) -> bool:
    vals = list(values)
    return bool(vals) and all((bool(value) for value in vals))


def truthy_done(value: object) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        vals = list(value)
        return bool(vals) and all((bool(item) for item in vals))
    if isinstance(value, dict):
        vals = list(value.values())
        return bool(vals) and all((bool(item) for item in vals))
    return bool(value)
