"""SCV training and evaluation for the four staged-transfer benchmark tasks.

Specialized from the archived trainer (SHA-256: 3610ca7156fb3fbee91f8cf604e05fb5ac14a30dc242b8760de0aa7fe350df1e).
The SCV credit program, counterfactual query, trust gates, predictor, and
constrained PPO advantage update are evaluated in their original order.
"""

from __future__ import annotations
import argparse, itertools, json, math, os, random, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from env_transfer import StandardMARLEnv, make_standard_marl_env
from scv_credit_programs import get_program

ROOT = Path(__file__).resolve().parent
_SCV_OPTIONS = {
    "learner": "mappo",
    "scheme": "cf_certified_raw_guard_centered",
    "credit_usage": "advantage_safe_residual",
    "credit_proposer": "qwen_credit_program",
    "actor_pretrain_source": "none",
    "behavior_proposal_source": "none",
    "proposal_action_source": "none",
    "qwen_proposal_cache": "",
    "qwen_action_proposal_cache": "",
    "task_hint_source": "none",
    "credit_model_type": "ridge",
    "safe_reward_credit_conversion": False,
    "credit_dataset_file": "",
    "potential_shaping_source": "none",
    "credit_program_temporal_only": False,
    "trajectory_scv_rware_mode_gate": False,
}


def _check_scv_options(args):
    for key, expected in _SCV_OPTIONS.items():
        if getattr(args, key) != expected:
            raise ValueError(f"SCV profile requires {key}={expected!r}")


class Actor(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class CreditCritic(nn.Module):
    """Action-conditioned critic head for verified credit corrections."""

    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        return self.net(obs_action).squeeze(-1)


@dataclass
class CreditModel:
    weights: List[float]

    def predict(self, features: Sequence[float]) -> float:
        return float(
            np.dot(
                np.asarray(self.weights, dtype=np.float64), np.asarray(features, dtype=np.float64)
            )
        )


QWEN_LBF_PROPOSAL_CACHE: Dict[str, Dict[str, object]] = {}


QWEN_LBF_PROPOSAL_ITEMS: List[Dict[str, object]] = []


QWEN_LBF_PROPOSAL_OBS_MATRIX: np.ndarray | None = None


QWEN_LBF_ACTION_PROPOSAL_CACHE: Dict[str, Dict[str, object]] = {}


QWEN_LBF_ACTION_PROPOSAL_ITEMS: List[Dict[str, object]] = []


QWEN_LBF_ACTION_PROPOSAL_OBS_MATRIX: np.ndarray | None = None


LBF_JOINT_GREEDY_ACTION_CACHE: Dict[str, List[int]] = {}


LBF_JOINT_GREEDY_ACTION_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
}


QWEN_LBF_PROPOSAL_CACHE_STATS = {
    "action_hits": 0,
    "action_exact_hits": 0,
    "action_knn_hits": 0,
    "action_misses": 0,
    "credit_hits": 0,
    "credit_exact_hits": 0,
    "credit_knn_hits": 0,
    "credit_misses": 0,
}


CREDIT_PROGRAM: Dict[str, object] = {}


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for raw in str(text or "").split(","):
        item = raw.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def train(args: argparse.Namespace) -> None:
    """Run a published SCV training configuration."""
    _check_scv_options(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    start_time = time.perf_counter()
    probe = make_standard_marl_env(args.env, max_steps=args.max_steps)
    probe.reset(seed=args.seed)
    env_info = {
        "name": probe.name,
        "family": probe.family,
        "title": probe.title,
        "n_agents": probe.n_agents,
        "obs_dim": probe.obs_dim,
        "n_actions": probe.n_actions,
        "max_steps": probe.max_steps,
    }
    args.n_actions_hint_dim = int(env_info["n_actions"])
    probe.close()
    global QWEN_LBF_PROPOSAL_CACHE
    global QWEN_LBF_PROPOSAL_ITEMS
    global QWEN_LBF_PROPOSAL_OBS_MATRIX
    global QWEN_LBF_ACTION_PROPOSAL_CACHE
    global QWEN_LBF_ACTION_PROPOSAL_ITEMS
    global QWEN_LBF_ACTION_PROPOSAL_OBS_MATRIX
    global QWEN_LBF_PROPOSAL_CACHE_STATS
    global CREDIT_PROGRAM
    CREDIT_PROGRAM = load_credit_program(ROOT / args.credit_program) if args.credit_program else {}
    expected_program = {"lbf": "lbf", "rware": "rware_global", "pursuit": "pursuit_global"}[
        env_info["family"]
    ]
    if CREDIT_PROGRAM.get("environment") != expected_program:
        raise ValueError(f"This task requires the published {expected_program!r} credit program")
    if not CREDIT_PROGRAM:
        raise ValueError("--credit-program is required for credit_proposer=qwen_credit_program")
    QWEN_LBF_PROPOSAL_CACHE = {}
    QWEN_LBF_PROPOSAL_ITEMS = list(QWEN_LBF_PROPOSAL_CACHE.values())
    obs_rows = [
        item["obs"] for item in QWEN_LBF_PROPOSAL_ITEMS if isinstance(item.get("obs"), list)
    ]
    QWEN_LBF_PROPOSAL_OBS_MATRIX = np.asarray(obs_rows, dtype=np.float32) if obs_rows else None
    QWEN_LBF_ACTION_PROPOSAL_CACHE = {}
    QWEN_LBF_ACTION_PROPOSAL_ITEMS = list(QWEN_LBF_ACTION_PROPOSAL_CACHE.values())
    action_obs_rows = [
        item["obs"] for item in QWEN_LBF_ACTION_PROPOSAL_ITEMS if isinstance(item.get("obs"), list)
    ]
    QWEN_LBF_ACTION_PROPOSAL_OBS_MATRIX = (
        np.asarray(action_obs_rows, dtype=np.float32) if action_obs_rows else None
    )
    QWEN_LBF_PROPOSAL_CACHE_STATS = {
        "action_hits": 0,
        "action_exact_hits": 0,
        "action_knn_hits": 0,
        "action_misses": 0,
        "credit_hits": 0,
        "credit_exact_hits": 0,
        "credit_knn_hits": 0,
        "credit_misses": 0,
    }
    if QWEN_LBF_PROPOSAL_CACHE:
        print(f"Loaded {len(QWEN_LBF_PROPOSAL_CACHE)} Qwen LBF proposal cache entries", flush=True)
    trajectory_credit = None
    online_qwen = None
    stacked_obs_dim = int(env_info["obs_dim"]) * obs_frame_stack(args)
    actor_dim = (
        stacked_obs_dim
        + (env_info["n_agents"] if args.agent_id else 0)
        + task_hint_dim(args, env_info)
    )
    critic_dim = stacked_obs_dim * env_info["n_agents"]
    if args.agent_id:
        critic_dim += env_info["n_agents"]
    actor = Actor(actor_dim, env_info["n_actions"], args.hidden).to(device)
    critic = Critic(critic_dim, args.hidden).to(device)
    credit_critic = CreditCritic(critic_dim + env_info["n_actions"], args.hidden).to(device)
    actor_pretrain_stats = pretrain_actor_with_proposals(actor, args, device, env_info)
    episode_seed_curriculum = load_episode_seed_curriculum(ROOT / args.episode_seed_curriculum_file)
    if episode_seed_curriculum:
        print(f"Loaded {len(episode_seed_curriculum)} episode curriculum seeds", flush=True)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    credit_optim = torch.optim.Adam(credit_critic.parameters(), lr=args.critic_lr)
    train_env = make_standard_marl_env(args.env, max_steps=args.max_steps)
    resume_payload: Dict[str, object] | None = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        actor.load_state_dict(resume_payload["actor_state_dict"])
        critic.load_state_dict(resume_payload["critic_state_dict"])
        credit_state = resume_payload.get("credit_critic_state_dict")
        if credit_state is not None:
            credit_critic.load_state_dict(credit_state)
        actor_optim_state = resume_payload.get("actor_optim_state_dict")
        critic_optim_state = resume_payload.get("critic_optim_state_dict")
        credit_optim_state = resume_payload.get("credit_optim_state_dict")
        if actor_optim_state is not None:
            actor_optim.load_state_dict(actor_optim_state)
        if critic_optim_state is not None:
            critic_optim.load_state_dict(critic_optim_state)
        if credit_optim_state is not None:
            credit_optim.load_state_dict(credit_optim_state)
        if args.resume_override_lr:
            for group in actor_optim.param_groups:
                group["lr"] = float(args.actor_lr)
            for group in critic_optim.param_groups:
                group["lr"] = float(args.critic_lr)
            for group in credit_optim.param_groups:
                group["lr"] = float(args.critic_lr)
        print(f"Loaded resume checkpoint {resume_path}", flush=True)
    dataset: List[Dict[str, object]] = []
    external_credit_dataset_files: List[str] = []
    external_credit_dataset_examples = 0
    positive_replay_episodes: List[List[Dict[str, object]]] = []
    credit_model: object | None = None
    credit_model_last_fit_examples = 0
    novelty_counts: Dict[str, int] = {}
    surrogate_mse_ema: float | None = None
    surrogate_checks = 0
    surrogate_reliability = 0.0
    credit_activation_ema = (
        1.0 if args.credit_activation_full_rate <= args.credit_activation_min_rate else 0.0
    )
    credit_gradient_auto_nonzero_rewards = 0
    credit_gradient_auto_total_rewards = 0
    credit_gradient_auto_gate_active: bool | None = None
    credit_gradient_auto_reward_density: float | None = None
    credit_gradient_auto_causal_lower_bound: float | None = None
    credit_gradient_effective_gate_active: bool | None = None
    learning_curve: List[Dict[str, float]] = []
    best_eval_return: float | None = None
    best_eval_record: Dict[str, float] | None = None
    episode_log: List[Dict[str, object]] = []
    cf_stats = {
        "credit_calls": 0,
        "replay_calls": 0,
        "simulated_steps": 0,
        "dense_skips": 0,
        "pair_credit_calls": 0,
        "pair_replay_calls": 0,
        "pair_simulated_steps": 0,
        "pair_informative": 0,
        "event_horizon_extensions": 0,
        "event_horizon_sum": 0,
        "event_horizon_max": 0,
    }
    online_qwen_stats: Dict[str, object] = {
        "queries": 0,
        "failures": 0,
        "events": 0,
        "wall_time_sec": 0.0,
    }
    trajectory_scv_stats: Dict[str, object] = {
        "audits": 0,
        "queries": 0,
        "uninformative": 0,
        "accepted": 0,
        "active_episodes": 0,
        "probation_episodes": 0,
        "rejected_episodes": 0,
        "rejected_fallback_episodes": 0,
        "probation_scale_sum": 0.0,
        "max_probation_scale": 0.0,
        "probation_escalation_triggered": False,
        "agreement_sum": 0.0,
        "scale_sum": 0.0,
        "accepted_scale_sum": 0.0,
        "recent_acceptance_history": [],
        "sparse_high_assurance_certified": False,
    }
    output_path = ROOT / args.output
    total_episodes = args.updates * args.batch_episodes
    episodes_seen = 0
    env_steps_seen = 0
    agent_steps_seen = 0
    update_idx = 0
    last_cf_reward_trigger_episode = -(10**9)
    if resume_payload is not None and args.resume_counters:
        learning_curve = [dict(item) for item in resume_payload.get("learning_curve", [])]
        best_eval_return = resume_payload.get("best_eval_return")
        best_eval_record = resume_payload.get("best_eval_record")
        episodes_seen = int(resume_payload.get("episodes_seen", 0) or 0)
        env_steps_seen = int(resume_payload.get("env_steps_seen", 0) or 0)
        agent_steps_seen = int(resume_payload.get("agent_steps_seen", 0) or 0)
        update_idx = int(resume_payload.get("update_idx", 0) or 0)
        dataset = [dict(item) for item in resume_payload.get("dataset", [])]
        external_credit_dataset_files = [
            str(item) for item in resume_payload.get("external_credit_dataset_files", [])
        ]
        external_credit_dataset_examples = int(
            resume_payload.get("external_credit_dataset_examples", 0) or 0
        )
        episode_log = [dict(item) for item in resume_payload.get("episode_log", [])]
        novelty_counts = dict(resume_payload.get("novelty_counts", {}))
        surrogate_mse_ema = resume_payload.get("surrogate_mse_ema")
        surrogate_checks = int(resume_payload.get("surrogate_checks", 0) or 0)
        surrogate_reliability = float(resume_payload.get("surrogate_reliability", 0.0) or 0.0)
        credit_activation_ema = float(
            resume_payload.get("credit_activation_ema", credit_activation_ema)
        )
        credit_gradient_auto_nonzero_rewards = int(
            resume_payload.get(
                "credit_gradient_auto_nonzero_rewards", credit_gradient_auto_nonzero_rewards
            )
            or 0
        )
        credit_gradient_auto_total_rewards = int(
            resume_payload.get(
                "credit_gradient_auto_total_rewards", credit_gradient_auto_total_rewards
            )
            or 0
        )
        saved_auto_gate = resume_payload.get("credit_gradient_auto_gate_active")
        credit_gradient_auto_gate_active = (
            None if saved_auto_gate is None else bool(saved_auto_gate)
        )
        saved_auto_density = resume_payload.get("credit_gradient_auto_reward_density")
        credit_gradient_auto_reward_density = (
            None if saved_auto_density is None else float(saved_auto_density)
        )
        last_cf_reward_trigger_episode = int(
            resume_payload.get("last_cf_reward_trigger_episode", -(10**9))
        )
        saved_cf_stats = resume_payload.get("cf_stats")
        if isinstance(saved_cf_stats, dict):
            cf_stats.update(saved_cf_stats)
        saved_online_qwen_stats = resume_payload.get("online_qwen_stats")
        if isinstance(saved_online_qwen_stats, dict):
            online_qwen_stats.update(saved_online_qwen_stats)
        saved_trajectory_scv_stats = resume_payload.get("trajectory_scv_stats")
        if isinstance(saved_trajectory_scv_stats, dict):
            trajectory_scv_stats.update(saved_trajectory_scv_stats)
        python_random_state = resume_payload.get("python_random_state")
        numpy_random_state = resume_payload.get("numpy_random_state")
        torch_random_state = resume_payload.get("torch_random_state")
        if python_random_state is not None:
            random.setstate(python_random_state)
        if numpy_random_state is not None:
            np.random.set_state(numpy_random_state)
        if torch_random_state is not None:
            torch.set_rng_state(torch_random_state.cpu())
        print(f"Resumed counters: env_steps={env_steps_seen} episodes={episodes_seen}", flush=True)
    if args.reset_credit_dataset_on_resume:
        dataset = []
        external_credit_dataset_files = []
        external_credit_dataset_examples = 0
        print(
            "Reset resumed credit-surrogate dataset while retaining policy and SCV state",
            flush=True,
        )
    if args.credit_surrogate and dataset:
        credit_model = fit_credit_model_for_args(dataset, args)
        credit_model_last_fit_examples = len(dataset)
    next_eval_episode = args.eval_every if args.eval_every > 0 else None
    next_eval_step = (
        args.eval_every_steps if args.eval_every_steps and args.eval_every_steps > 0 else None
    )
    if next_eval_episode is not None and episodes_seen > 0:
        next_eval_episode = (episodes_seen // args.eval_every + 1) * args.eval_every
    if next_eval_step is not None and env_steps_seen > 0:
        next_eval_step = (env_steps_seen // args.eval_every_steps + 1) * args.eval_every_steps

    def build_output(status: str) -> Dict[str, object]:
        query_count = sum((int(item["source_counts"].get("cf_query", 0)) for item in episode_log))
        return {
            "status": status,
            "args": vars(args),
            "env": env_info,
            "updates_seen": update_idx,
            "episodes_seen": episodes_seen,
            "train_env_steps": env_steps_seen,
            "train_agent_steps": agent_steps_seen,
            "cf_queries": query_count,
            "query_rate": query_count / max(episodes_seen, 1),
            "query_rate_per_env_step": query_count / max(env_steps_seen, 1),
            "dataset_examples": len(dataset),
            "credit_model_last_fit_examples": credit_model_last_fit_examples,
            "external_credit_dataset_examples": external_credit_dataset_examples,
            "external_credit_dataset_files": list(external_credit_dataset_files),
            "cf_credit_calls": cf_stats["credit_calls"],
            "cf_replay_calls": cf_stats["replay_calls"],
            "cf_simulated_steps": cf_stats["simulated_steps"],
            "cf_dense_episode_skips": cf_stats["dense_skips"],
            "cf_pair_credit_calls": cf_stats.get("pair_credit_calls", 0),
            "cf_pair_replay_calls": cf_stats.get("pair_replay_calls", 0),
            "cf_pair_simulated_steps": cf_stats.get("pair_simulated_steps", 0),
            "cf_pair_informative": cf_stats.get("pair_informative", 0),
            "cf_event_horizon_extensions": cf_stats.get("event_horizon_extensions", 0),
            "cf_event_horizon_sum": cf_stats.get("event_horizon_sum", 0),
            "cf_event_horizon_max": cf_stats.get("event_horizon_max", 0),
            "online_qwen_stats": dict(online_qwen_stats),
            "trajectory_scv_stats": summarize_trajectory_scv_stats(
                trajectory_scv_stats, args, reward_density=credit_gradient_auto_reward_density
            ),
            "credit_gradient_auto_stats": {
                "nonzero_rewards": credit_gradient_auto_nonzero_rewards,
                "total_rewards": credit_gradient_auto_total_rewards,
                "reward_density": credit_gradient_auto_reward_density,
                "gate_active": credit_gradient_auto_gate_active,
                "causal_lower_bound": credit_gradient_auto_causal_lower_bound,
                "effective_gate_active": credit_gradient_effective_gate_active,
            },
            "qwen_lbf_proposal_cache_entries": len(QWEN_LBF_PROPOSAL_CACHE),
            "qwen_lbf_action_proposal_cache_entries": len(QWEN_LBF_ACTION_PROPOSAL_CACHE),
            "qwen_lbf_proposal_cache_stats": dict(QWEN_LBF_PROPOSAL_CACHE_STATS),
            "lbf_joint_greedy_action_cache_entries": len(LBF_JOINT_GREEDY_ACTION_CACHE),
            "lbf_joint_greedy_action_cache_stats": dict(LBF_JOINT_GREEDY_ACTION_CACHE_STATS),
            "actor_pretrain_stats": actor_pretrain_stats,
            "episode_seed_curriculum_entries": len(episode_seed_curriculum),
            "wall_time_sec": round(time.perf_counter() - start_time, 8),
            "learning_curve": learning_curve,
            "episode_log": episode_log,
            "final_eval_return": learning_curve[-1]["eval_return"] if learning_curve else None,
            "best_eval_return": best_eval_return,
            "best_eval_record": best_eval_record,
        }

    def checkpoint_payload(status: str, is_best: bool = False) -> Dict[str, object]:
        return {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "credit_critic_state_dict": (
                credit_critic.state_dict() if credit_critic is not None else None
            ),
            "actor_optim_state_dict": actor_optim.state_dict(),
            "critic_optim_state_dict": critic_optim.state_dict(),
            "credit_optim_state_dict": credit_optim.state_dict(),
            "args": vars(args),
            "env_info": dict(env_info),
            "status": status,
            "is_best": bool(is_best),
            "learning_curve": list(learning_curve),
            "best_eval_return": best_eval_return,
            "best_eval_record": best_eval_record,
            "episodes_seen": episodes_seen,
            "env_steps_seen": env_steps_seen,
            "agent_steps_seen": agent_steps_seen,
            "update_idx": update_idx,
            "dataset": list(dataset),
            "external_credit_dataset_examples": external_credit_dataset_examples,
            "external_credit_dataset_files": list(external_credit_dataset_files),
            "episode_log": list(episode_log),
            "novelty_counts": dict(novelty_counts),
            "surrogate_mse_ema": surrogate_mse_ema,
            "surrogate_checks": surrogate_checks,
            "surrogate_reliability": surrogate_reliability,
            "credit_activation_ema": credit_activation_ema,
            "credit_gradient_auto_nonzero_rewards": credit_gradient_auto_nonzero_rewards,
            "credit_gradient_auto_total_rewards": credit_gradient_auto_total_rewards,
            "credit_gradient_auto_gate_active": credit_gradient_auto_gate_active,
            "credit_gradient_auto_reward_density": credit_gradient_auto_reward_density,
            "last_cf_reward_trigger_episode": last_cf_reward_trigger_episode,
            "cf_stats": dict(cf_stats),
            "online_qwen_stats": dict(online_qwen_stats),
            "trajectory_scv_stats": dict(trajectory_scv_stats),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        }

    def save_checkpoint(path: Path, status: str, is_best: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload(status, is_best=is_best), path)

    def derived_best_checkpoint_path() -> Path | None:
        if args.best_checkpoint_path:
            return Path(args.best_checkpoint_path)
        if not args.checkpoint_path:
            return None
        checkpoint_path = Path(args.checkpoint_path)
        suffix = checkpoint_path.suffix or ".pt"
        stem = (
            checkpoint_path.name[: -len(suffix)]
            if checkpoint_path.name.endswith(suffix)
            else checkpoint_path.name
        )
        return checkpoint_path.with_name(f"{stem}.best{suffix}")

    def derived_eval_checkpoint_path(step: int) -> Path | None:
        if not args.checkpoint_path:
            return None
        checkpoint_path = Path(args.checkpoint_path)
        suffix = checkpoint_path.suffix or ".pt"
        stem = (
            checkpoint_path.name[: -len(suffix)]
            if checkpoint_path.name.endswith(suffix)
            else checkpoint_path.name
        )
        return checkpoint_path.with_name(f"{stem}.eval{int(step)}{suffix}")

    def evaluate_for_logging() -> float:
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            return evaluate(actor, args, device, env_info)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def evaluate_with_overrides_for_logging(**kwargs: object) -> float:
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            return evaluate_with_overrides(actor, args, device, env_info, **kwargs)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def write_output(status: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(build_output(status), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.checkpoint_path:
            save_checkpoint(Path(args.checkpoint_path), status, is_best=False)

    while args.total_env_steps is None or env_steps_seen < args.total_env_steps:
        if args.total_env_steps is None and update_idx >= args.updates:
            break
        update_idx += 1
        batch_samples: List[Dict[str, object]] = []
        train_returns = []
        train_env_steps = []
        source_counts = {"team": 0, "raw": 0, "cf_query": 0, "surrogate": 0}
        replay_samples_added = 0
        activation_scale = credit_activation_scale(args, credit_activation_ema)
        for batch_offset in range(args.batch_episodes):
            if args.total_env_steps is not None and env_steps_seen >= args.total_env_steps:
                break
            episode_idx = episodes_seen + 1
            episode_seed = select_episode_seed(
                args=args,
                episode_idx=episode_idx,
                env_steps_seen=env_steps_seen,
                curriculum_seeds=episode_seed_curriculum,
            )
            (samples, transitions, train_return) = collect_episode(
                actor=actor,
                args=args,
                device=device,
                episode_seed=episode_seed,
                env_info=env_info,
                env=train_env,
                task_hint_weight=scheduled_task_hint_weight(args, env_steps_seen),
                task_hint_dropout=scheduled_task_hint_dropout(args, env_steps_seen),
                behavior_proposal_prob=scheduled_behavior_proposal_prob(args, env_steps_seen),
            )
            for sample in samples:
                sample["episode_idx"] = episode_idx
            train_returns.append(train_return)
            episode_steps = len(transitions)
            train_env_steps.append(episode_steps)
            episodes_seen += 1
            env_steps_seen += episode_steps
            agent_steps_seen += episode_steps * int(env_info["n_agents"])
            online_credit_events = 0
            if str(env_info["family"]) == "rware" and str(
                CREDIT_PROGRAM.get("environment", "")
            ) in {"rware", "rware_global"}:
                apply_rware_temporal_program_credit(transitions, args)
            raw_credits = [list(transition["raw_credit"]) for transition in transitions]
            features = feature_rows(transitions, raw_credits, env_info["n_actions"], args)
            cf_credits_by_t: Dict[int, List[float]] = {}
            source = "team"
            exact_query = False
            credit_schedule_active = scheduled_credit_decay_weight(args, env_steps_seen) > 1e-12
            scheduled_query = credit_schedule_active and should_query(episode_idx, args)
            scv_schedule_resolved = (
                trajectory_scv_schedule_resolved(
                    trajectory_scv_stats, args, reward_density=credit_gradient_auto_reward_density
                )
                or False
            )
            if int(args.query_every_certified) >= 0 and scv_schedule_resolved:
                scheduled_query = (
                    int(args.query_every_certified) > 0
                    and episode_idx % int(args.query_every_certified) == 0
                )
            if int(args.query_stop_env_steps) > 0 and env_steps_seen > int(
                args.query_stop_env_steps
            ):
                scheduled_query = False
            online_credit_query = credit_schedule_active and False
            reward_trigger_query = bool(
                credit_schedule_active
                and args.cf_reward_trigger_query
                and any(
                    (
                        abs(float(transition["team_reward"])) > float(args.cf_reward_threshold)
                        for transition in transitions
                    )
                )
            )
            if (
                reward_trigger_query
                and args.cf_reward_trigger_until_certified
                and scv_schedule_resolved
            ):
                reward_trigger_query = False
            if (
                reward_trigger_query
                and int(args.cf_reward_trigger_stop_after_informative) > 0
                and (
                    int(trajectory_scv_stats.get("queries", 0))
                    >= int(args.cf_reward_trigger_stop_after_informative)
                )
            ):
                reward_trigger_query = False
            if reward_trigger_query and int(args.cf_reward_trigger_min_gap_episodes) > 0:
                reward_trigger_query = episode_idx - last_cf_reward_trigger_episode >= int(
                    args.cf_reward_trigger_min_gap_episodes
                )
            reward_density = mean(
                [
                    (
                        1.0
                        if abs(float(transition["team_reward"])) > float(args.cf_reward_threshold)
                        else 0.0
                    )
                    for transition in transitions
                ]
            )
            skip_dense_query = (
                not exact_query
                and scheduled_query
                and (float(args.cf_skip_reward_density_above) <= 1.0)
                and (reward_density >= float(args.cf_skip_reward_density_above))
            )
            if skip_dense_query:
                cf_stats["dense_skips"] += 1
            query = credit_schedule_active and (
                exact_query
                or (
                    (scheduled_query or online_credit_query or reward_trigger_query)
                    and (not skip_dense_query)
                )
            )
            if query:
                if reward_trigger_query:
                    last_cf_reward_trigger_episode = episode_idx
                query_indexes = (
                    [
                        t
                        for (t, transition) in enumerate(transitions)
                        if bool(transition.get("online_qwen_credit_event", False))
                    ][: max(1, int(args.cf_transitions))]
                    if online_credit_query
                    else selected_query_indexes_from_transitions(
                        transitions,
                        args,
                        episode_idx=episode_idx,
                        features=features,
                        credit_model=credit_model,
                    )
                )
                cf_credits_by_t = counterfactual_trajectory_credits(
                    args=args,
                    episode_seed=episode_seed,
                    transitions=transitions,
                    env_info=env_info,
                    query_indexes=query_indexes,
                    stats=cf_stats,
                    actor=actor,
                    device=device,
                )
                batch_surrogate_mse = None
                if credit_model is not None and args.surrogate_confidence_gate:
                    batch_surrogate_mse = surrogate_verification_mse(
                        features=features,
                        cf_credits_by_t=cf_credits_by_t,
                        credit_model=credit_model,
                    )
                    if batch_surrogate_mse is not None:
                        surrogate_checks += 1
                        if surrogate_mse_ema is None:
                            surrogate_mse_ema = batch_surrogate_mse
                        else:
                            beta = float(args.surrogate_confidence_ema)
                            surrogate_mse_ema = (
                                beta * surrogate_mse_ema + (1.0 - beta) * batch_surrogate_mse
                            )
                        surrogate_reliability = reliability_from_mse(
                            surrogate_mse_ema, args.surrogate_confidence_mse_gate
                        )
                dataset.extend(dataset_examples(features, cf_credits_by_t))
                refit_examples = max(1, int(args.credit_model_refit_examples))
                if (
                    credit_model is None
                    or len(dataset) - credit_model_last_fit_examples >= refit_examples
                ):
                    credit_model = fit_credit_model_for_args(dataset, args)
                    credit_model_last_fit_examples = len(dataset)
                source = "cf_query"
            elif args.credit_surrogate and credit_model is not None:
                source = "surrogate"
            trajectory_global_reliable = True
            trajectory_global_scale = 1.0
            trajectory_global_state = "certified"
            trajectory_uncertified_scale = 0.0
            if cf_credits_by_t:
                (agreement, scale, accepted) = trajectory_scv_metrics(
                    raw_credits, cf_credits_by_t, args
                )
                trajectory_scv_stats["audits"] = int(trajectory_scv_stats["audits"]) + 1
                informative = any(
                    (abs(float(value)) > 1e-08 for row in cf_credits_by_t.values() for value in row)
                )
                if informative:
                    trajectory_scv_stats["queries"] = int(trajectory_scv_stats["queries"]) + 1
                    trajectory_scv_stats["accepted"] = int(trajectory_scv_stats["accepted"]) + int(
                        accepted
                    )
                    rolling_window = max(0, int(args.trajectory_scv_rolling_window))
                    if rolling_window > 0:
                        recent_history = trajectory_scv_stats.setdefault(
                            "recent_acceptance_history", []
                        )
                        if not isinstance(recent_history, list):
                            recent_history = []
                            trajectory_scv_stats["recent_acceptance_history"] = recent_history
                        recent_history.append(int(accepted))
                        if len(recent_history) > rolling_window:
                            del recent_history[:-rolling_window]
                    trajectory_scv_stats["agreement_sum"] = (
                        float(trajectory_scv_stats["agreement_sum"]) + agreement
                    )
                    trajectory_scv_stats["scale_sum"] = (
                        float(trajectory_scv_stats["scale_sum"]) + scale
                    )
                    if accepted:
                        trajectory_scv_stats["accepted_scale_sum"] = (
                            float(trajectory_scv_stats["accepted_scale_sum"]) + scale
                        )
                else:
                    trajectory_scv_stats["uninformative"] = (
                        int(trajectory_scv_stats["uninformative"]) + 1
                    )
            trajectory_global_state = trajectory_scv_trust_state(
                trajectory_scv_stats, args, reward_density=credit_gradient_auto_reward_density
            )
            if trajectory_global_state == "certified" and trajectory_scv_sparse_reward_regime(
                args, credit_gradient_auto_reward_density
            ):
                trajectory_scv_stats["sparse_high_assurance_certified"] = True
            trajectory_global_reliable = trajectory_global_state == "certified"
            if trajectory_global_state == "probation":
                trajectory_uncertified_scale = probation_credit_scale(
                    trajectory_scv_stats, args, int(env_info["n_agents"])
                )
                if (
                    trajectory_uncertified_scale
                    > (
                        1.0 / max(1, int(env_info["n_agents"]))
                        if args.uncertified_credit_scale_by_agents
                        else float(args.uncertified_credit_scale)
                    )
                    + 1e-12
                ):
                    trajectory_scv_stats["probation_escalation_triggered"] = True
            trajectory_global_scale = float(trajectory_scv_stats["accepted_scale_sum"]) / max(
                1, int(trajectory_scv_stats["accepted"])
            )
            rware_mode_gate_active = False
            used_credits = choose_credits(
                scheme="cf_certified_raw_guard_centered",
                raw_credits=raw_credits,
                features=features,
                cf_credits_by_t=cf_credits_by_t,
                credit_model=credit_model if args.credit_surrogate else None,
                args=args,
                global_calibration_reliable=trajectory_global_reliable or rware_mode_gate_active,
                global_calibration_state=trajectory_global_state,
                global_calibration_scale=trajectory_global_scale,
                uncertified_credit_scale=trajectory_uncertified_scale,
            )
            if rware_mode_gate_active:
                used_credits = apply_rware_mode_credit_gate(
                    used_credits, transitions, trajectory_scv_stats, args, int(env_info["n_agents"])
                )
            if trajectory_global_reliable:
                trajectory_scv_stats["active_episodes"] = (
                    int(trajectory_scv_stats["active_episodes"]) + 1
                )
            elif trajectory_global_state == "probation":
                trajectory_scv_stats["probation_episodes"] = (
                    int(trajectory_scv_stats["probation_episodes"]) + 1
                )
                trajectory_scv_stats["probation_scale_sum"] = (
                    float(trajectory_scv_stats["probation_scale_sum"])
                    + trajectory_uncertified_scale
                )
                trajectory_scv_stats["max_probation_scale"] = max(
                    float(trajectory_scv_stats["max_probation_scale"]), trajectory_uncertified_scale
                )
            elif trajectory_global_state == "rejected":
                trajectory_scv_stats["rejected_episodes"] = (
                    int(trajectory_scv_stats["rejected_episodes"]) + 1
                )
                if args.rejected_credit_fallback == "verified_surrogate" and any(
                    (any((abs(float(value)) > 1e-08 for value in row)) for row in used_credits)
                ):
                    trajectory_scv_stats["rejected_fallback_episodes"] = (
                        int(trajectory_scv_stats["rejected_fallback_episodes"]) + 1
                    )
                    trajectory_scv_stats["active_episodes"] = (
                        int(trajectory_scv_stats["active_episodes"]) + 1
                    )
            credit_sources = credit_source_rows(
                scheme="cf_certified_raw_guard_centered",
                n_steps=len(transitions),
                cf_credits_by_t=cf_credits_by_t,
                credit_model=credit_model if args.credit_surrogate else None,
                args=args,
            )
            warm_credit_weight = 0.0 if env_steps_seen <= args.credit_warmup_env_steps else 1.0
            actor_credit_weight = (
                warm_credit_weight
                * activation_scale
                * scheduled_credit_decay_weight(args, env_steps_seen)
            )
            if args.trajectory_scv_confidence_weighting and trajectory_global_state == "certified":
                actor_credit_weight *= trajectory_scv_certified_confidence_weight(
                    trajectory_scv_stats, args
                )
            if args.surrogate_confidence_gate:
                if source == "cf_query" and args.verified_credit_always_on:
                    confidence_weight = 1.0
                elif surrogate_checks >= args.surrogate_confidence_min_checks:
                    confidence_weight = surrogate_reliability
                else:
                    confidence_weight = 0.0
                actor_credit_weight *= confidence_weight
            novelty_weight = scheduled_novelty_weight(args, env_steps_seen)
            novelty_bonuses = novelty_bonus_rows(transitions, novelty_counts, args)
            potential_shaping_weight = scheduled_potential_shaping_weight(args, env_steps_seen)
            event_reward_phase = event_curriculum_reward_phase_active(
                args, env_steps_seen=env_steps_seen, credit_certified=trajectory_global_reliable
            )
            assign_returns(
                samples,
                transitions,
                used_credits,
                args,
                actor_credit_weight,
                novelty_bonuses,
                novelty_weight,
                credit_sources,
                warm_credit_weight,
                warm_credit_weight
                * (
                    surrogate_reliability
                    if surrogate_checks >= args.surrogate_confidence_min_checks
                    else 0.0
                ),
                event_curriculum_reward_phase=event_reward_phase,
                proposal_action_weight=scheduled_proposal_action_weight(args, env_steps_seen),
                advantage_credit_weight=0.0 if event_reward_phase else actor_credit_weight,
                potential_shaping_weight=potential_shaping_weight,
            )
            replay_credit_certified = not args.positive_replay_require_certified or (
                trajectory_global_reliable
                and (
                    not rware_mode_gate_active
                    or rware_mode_scv_all_certified(trajectory_scv_stats, args)
                )
            )
            if train_return > args.positive_replay_min_return and replay_credit_certified:
                positive_replay_episodes.append([dict(sample) for sample in samples])
                if len(positive_replay_episodes) > args.positive_replay_max_episodes:
                    positive_replay_episodes = positive_replay_episodes[
                        -args.positive_replay_max_episodes :
                    ]
            batch_samples.extend(samples)
            source_counts[source] = source_counts.get(source, 0) + 1
        if not batch_samples:
            break
        batch_activity = mean(
            [
                1.0 if abs(float(value)) > float(args.credit_activation_reward_threshold) else 0.0
                for value in train_returns
            ]
        )
        if args.credit_activation_full_rate > args.credit_activation_min_rate:
            beta = float(np.clip(args.credit_activation_ema, 0.0, 0.9999))
            credit_activation_ema = beta * credit_activation_ema + (1.0 - beta) * batch_activity
        auto_gradient_mode = bool(
            args.credit_gradient_safety
        ) and args.credit_gradient_conflict_mode in {
            "auto",
            "auto_causal",
            "auto_reject",
            "auto_sparse_trust",
        }
        if auto_gradient_mode and credit_gradient_auto_gate_active is None:
            for sample in batch_samples:
                if float(sample.get("replay_sample", 0.0)) > 0.0:
                    continue
                credit_gradient_auto_total_rewards += 1
                if abs(float(sample.get("team_reward", 0.0))) > float(
                    args.credit_activation_reward_threshold
                ):
                    credit_gradient_auto_nonzero_rewards += 1
            if env_steps_seen >= int(args.credit_gradient_auto_warmup_env_steps):
                (credit_gradient_auto_gate_active, credit_gradient_auto_reward_density) = (
                    reward_density_gate_decision(
                        credit_gradient_auto_nonzero_rewards,
                        credit_gradient_auto_total_rewards,
                        args.credit_gradient_auto_density_threshold,
                    )
                )
        credit_gradient_effective_gate_active = bool(credit_gradient_auto_gate_active)
        if auto_gradient_mode and args.credit_gradient_conflict_mode == "auto_causal":
            (causal_gate_active, credit_gradient_auto_causal_lower_bound) = (
                causal_confidence_gate_decision(
                    trajectory_scv_stats,
                    min_queries=args.credit_gradient_causal_min_queries,
                    lower_bound_threshold=args.credit_gradient_causal_lower_bound_threshold,
                    confidence_z=args.trajectory_scv_global_confidence_z,
                )
            )
            credit_gradient_effective_gate_active = (
                bool(credit_gradient_auto_gate_active) or causal_gate_active
            )
        elif auto_gradient_mode and args.credit_gradient_conflict_mode == "auto_reject":
            credit_gradient_effective_gate_active = (
                bool(credit_gradient_auto_gate_active) or trajectory_global_state == "rejected"
            )
        elif auto_gradient_mode and args.credit_gradient_conflict_mode == "auto_sparse_trust":
            (sparse_trust_gate_active, _sparse_trust_state) = sparse_trust_gradient_gate_decision(
                trajectory_scv_stats, args, reward_density=credit_gradient_auto_reward_density
            )
            credit_gradient_effective_gate_active = (
                bool(credit_gradient_auto_gate_active) or sparse_trust_gate_active
            )
        if args.positive_replay_ratio > 0.0 and positive_replay_episodes:
            target_replay_samples = int(
                round(len(batch_samples) * float(args.positive_replay_ratio))
            )
            replay_batches: List[Dict[str, object]] = []
            replay_counter = 0
            while len(replay_batches) < target_replay_samples:
                episode = random.choice(positive_replay_episodes)
                replay_episode_idx = -(update_idx * 100000 + replay_counter + 1)
                replay_counter += 1
                for sample in episode:
                    copied = dict(sample)
                    copied["episode_idx"] = replay_episode_idx
                    copied["replay_sample"] = 1.0
                    replay_batches.append(copied)
                    if len(replay_batches) >= target_replay_samples:
                        break
            batch_samples.extend(replay_batches)
            replay_samples_added = len(replay_batches)
        entropy_coef = scheduled_entropy_coef(args, env_steps_seen, surrogate_reliability)
        update_stats = update_ppo(
            actor,
            critic,
            credit_critic,
            actor_optim,
            critic_optim,
            credit_optim,
            batch_samples,
            args,
            device,
            int(env_info["n_actions"]),
            entropy_coef,
            credit_gradient_safety_override=(
                bool(credit_gradient_effective_gate_active) if auto_gradient_mode else None
            ),
            credit_gradient_conflict_mode_override="gate" if auto_gradient_mode else None,
        )
        update_stats["credit_gradient_auto_mode"] = 1.0 if auto_gradient_mode else 0.0
        update_stats["credit_gradient_auto_gate_active"] = (
            1.0 if credit_gradient_effective_gate_active else 0.0
        )
        update_stats["credit_gradient_auto_reward_density"] = (
            float(credit_gradient_auto_reward_density)
            if credit_gradient_auto_reward_density is not None
            else 0.0
        )
        update_stats["credit_gradient_auto_causal_lower_bound"] = (
            float(credit_gradient_auto_causal_lower_bound)
            if credit_gradient_auto_causal_lower_bound is not None
            else 0.0
        )
        episode_log.append(
            {
                "update": update_idx,
                "episodes_seen": episodes_seen,
                "env_steps_seen": env_steps_seen,
                "agent_steps_seen": agent_steps_seen,
                "train_return": round(mean(train_returns), 8),
                "train_env_steps": int(sum(train_env_steps)),
                "source_counts": source_counts,
                "dataset_examples": len(dataset),
                "surrogate_checks": surrogate_checks,
                "surrogate_mse_ema": (
                    round(surrogate_mse_ema, 8) if surrogate_mse_ema is not None else None
                ),
                "surrogate_reliability": round(surrogate_reliability, 8),
                "novelty_states": len(novelty_counts),
                "novelty_weight": round(novelty_weight, 8),
                "potential_shaping_weight": round(float(potential_shaping_weight), 8),
                "positive_replay_episodes": len(positive_replay_episodes),
                "positive_replay_samples_added": replay_samples_added,
                "credit_activation_ema": round(float(credit_activation_ema), 8),
                "credit_activation_scale": round(float(activation_scale), 8),
                "credit_activation_batch_activity": round(float(batch_activity), 8),
                "credit_gradient_auto_gate_active": credit_gradient_auto_gate_active,
                "credit_gradient_auto_reward_density": (
                    round(float(credit_gradient_auto_reward_density), 8)
                    if credit_gradient_auto_reward_density is not None
                    else None
                ),
                "proposal_action_weight": round(
                    float(scheduled_proposal_action_weight(args, env_steps_seen)), 8
                ),
                "task_hint_weight": round(
                    float(scheduled_task_hint_weight(args, env_steps_seen)), 8
                ),
                "task_hint_dropout": round(
                    float(scheduled_task_hint_dropout(args, env_steps_seen)), 8
                ),
                "behavior_proposal_prob": round(
                    float(scheduled_behavior_proposal_prob(args, env_steps_seen)), 8
                ),
                "update_stats": update_stats,
            }
        )
        final_update = (
            args.total_env_steps is None
            and update_idx >= args.updates
            or (args.total_env_steps is not None and env_steps_seen >= args.total_env_steps)
        )
        eval_due = update_idx == 1 or final_update
        if next_eval_episode is not None and episodes_seen >= next_eval_episode:
            eval_due = True
            while next_eval_episode is not None and episodes_seen >= next_eval_episode:
                next_eval_episode += args.eval_every
        if next_eval_step is not None and env_steps_seen >= next_eval_step:
            eval_due = True
            while next_eval_step is not None and env_steps_seen >= next_eval_step:
                next_eval_step += args.eval_every_steps
        if eval_due:
            eval_return = evaluate_for_logging()
            extra_eval_returns: Dict[str, float] = {}
            if args.eval_report_greedy:
                extra_eval_returns["eval_return_greedy"] = round(
                    evaluate_with_overrides_for_logging(mode="greedy"), 8
                )
            for temp in parse_float_list(args.eval_report_temperatures):
                key = f"eval_return_temp_{str(temp).replace('.', 'p')}"
                extra_eval_returns[key] = round(
                    evaluate_with_overrides_for_logging(mode="stochastic", temperature=temp), 8
                )
            learning_curve.append(
                {
                    "episode": episodes_seen,
                    "env_steps": env_steps_seen,
                    "agent_steps": agent_steps_seen,
                    "train_return": round(mean(train_returns), 8),
                    "eval_return": round(eval_return, 8),
                    **extra_eval_returns,
                }
            )
            latest_eval_record = dict(learning_curve[-1])
            if args.checkpoint_every_eval:
                eval_checkpoint_path = derived_eval_checkpoint_path(env_steps_seen)
                if eval_checkpoint_path is not None:
                    save_checkpoint(eval_checkpoint_path, "eval", is_best=False)
            if best_eval_return is None or float(eval_return) > float(best_eval_return):
                best_eval_return = round(float(eval_return), 8)
                best_eval_record = latest_eval_record
                best_checkpoint_path = derived_best_checkpoint_path()
                if best_checkpoint_path is not None:
                    save_checkpoint(best_checkpoint_path, "best", is_best=True)
            if args.total_env_steps is None:
                progress = f"episode={episodes_seen}/{total_episodes}"
            else:
                progress = (
                    f"env_steps={env_steps_seen}/{args.total_env_steps} episode={episodes_seen}"
                )
            print(
                f"env={args.env} learner={'mappo'} scheme={'cf_certified_raw_guard_centered'} {progress} eval_return={eval_return:.4f}",
                flush=True,
            )
            write_output("running")
    train_env.close()
    write_output("complete")
    print(f"Wrote standard MARL result to {output_path}")


def pretrain_actor_with_proposals(
    actor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    env_info: Mapping[str, int | str],
) -> Dict[str, object]:
    return {"enabled": False}


def collect_episode(
    actor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    episode_seed: int,
    env_info: Mapping[str, int | str],
    env: StandardMARLEnv,
    task_hint_weight: float = 0.0,
    task_hint_dropout: float = 0.0,
    behavior_proposal_prob: float = 0.0,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], float]:
    observations = env.reset(seed=episode_seed)
    histories = init_obs_histories(observations, obs_frame_stack(args))
    samples: List[Dict[str, object]] = []
    transitions: List[Dict[str, object]] = []
    total_return = 0.0
    done = False
    rng = np.random.default_rng(episode_seed + 123457)
    while not done:
        actor_observations = stacked_observations_from_histories(histories)
        hint_actions = task_hint_actions_for_observations(
            observations=observations,
            env_family=str(env_info["family"]),
            n_agents=int(env_info["n_agents"]),
            args=args,
            env=env,
        )
        hint_targets = task_hint_targets_for_observations(
            observations=observations,
            env_family=str(env_info["family"]),
            n_agents=int(env_info["n_agents"]),
            args=args,
            env=env,
        )
        actor_inputs = actor_inputs_from_obs(
            actor_observations,
            env.n_agents,
            args.agent_id,
            args=args,
            env_family=str(env_info["family"]),
            task_hint_weight=task_hint_weight,
            task_hint_dropout=task_hint_dropout,
            hint_observations=observations,
            hint_actions=hint_actions,
            hint_targets=hint_targets,
        )
        obs_tensor = torch.as_tensor(np.stack(actor_inputs), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = actor(obs_tensor)
            dist = Categorical(logits=logits)
            action_tensor = dist.sample()
        actions = [int(action) for action in action_tensor.cpu().numpy().tolist()]
        behavior_proposal_actions = proposal_actions_for_observations(
            observations=observations,
            env_family=str(env_info["family"]),
            n_agents=int(env_info["n_agents"]),
            args=args,
            source="none",
            env=env,
        )
        behavior_proposal_used = [0 for _ in actions]
        proposal_actions = proposal_actions_for_observations(
            observations=observations,
            env_family=str(env_info["family"]),
            n_agents=int(env_info["n_agents"]),
            args=args,
            env=env,
        )
        actual_action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
        with torch.no_grad():
            old_logps = dist.log_prob(actual_action_tensor)
        credit_program_environment = str(CREDIT_PROGRAM.get("environment", ""))
        global_credit_before = None
        if str(env_info["family"]) == "rware" and credit_program_environment == "rware_global":
            global_credit_before = rware_global_credit_snapshot(env)
        step = env.step(actions)
        if global_credit_before is not None and credit_program_environment == "rware_global":
            global_credit_after = rware_global_credit_snapshot(env)
        else:
            global_credit_after = None
        raw_credit = proposed_credit(
            observations=observations,
            next_observations=step.observations,
            rewards=step.rewards,
            team_reward=step.team_reward,
            actions=actions,
            n_actions=env.n_actions,
            env_family=str(env_info["family"]),
            n_agents=int(env_info["n_agents"]),
            args=args,
            use_lbf_struct_credit=args.lbf_struct_credit,
            use_rware_struct_credit=args.rware_struct_credit,
        )
        if (
            global_credit_before is not None
            and global_credit_after is not None
            and (credit_program_environment == "rware_global")
        ):
            raw_credit = rware_global_program_credit(
                global_credit_before,
                global_credit_after,
                step.rewards,
                step.team_reward,
                actions,
                CREDIT_PROGRAM,
                scale=float(args.credit_program_global_scale),
            )
        central = central_input_from_obs(
            actor_observations,
            env.n_agents,
            args.agent_id,
            args=args,
            env_family=str(env_info["family"]),
        )
        transition = {
            "t": env.t - 1,
            "obs": [obs.astype(np.float32).tolist() for obs in observations],
            "next_obs": [obs.astype(np.float32).tolist() for obs in step.observations],
            "actions": list(actions),
            "action_probs": dist.probs.detach().cpu().numpy().astype(np.float32).tolist(),
            "rewards": list(step.rewards),
            "team_reward": float(step.team_reward),
            "raw_credit": raw_credit,
            "proposal_actions": proposal_actions,
            "behavior_proposal_actions": behavior_proposal_actions,
            "behavior_proposal_used": behavior_proposal_used,
        }
        if global_credit_before is not None:
            transition["credit_global_before"] = global_credit_before
            transition["credit_global_after"] = global_credit_after
        transitions.append(transition)
        for agent_idx, action in enumerate(actions):
            samples.append(
                {
                    "obs": actor_inputs[agent_idx].astype(np.float32),
                    "critic_obs": critic_input_for_agent(
                        central, agent_idx, env.n_agents, args.agent_id
                    ),
                    "action": action,
                    "old_logp": float(old_logps[agent_idx].cpu().item()),
                    "agent_idx": agent_idx,
                    "t": env.t - 1,
                    "proposal_action": int(proposal_actions[agent_idx]) if proposal_actions else -1,
                    "behavior_proposal_action": (
                        int(behavior_proposal_actions[agent_idx])
                        if behavior_proposal_actions
                        else -1
                    ),
                    "behavior_proposal_used": float(behavior_proposal_used[agent_idx]),
                }
            )
        total_return += step.team_reward
        observations = step.observations
        update_obs_histories(histories, observations, obs_frame_stack(args))
        done = step.done
    return (samples, transitions, total_return)


def trajectory_scv_metrics(
    raw_credits: Sequence[Sequence[float]],
    cf_credits_by_t: Mapping[int, Sequence[float]],
    args: argparse.Namespace,
) -> Tuple[float, float, bool]:
    raw_values = []
    verified_values = []
    for t, verified_row in cf_credits_by_t.items():
        if t >= len(raw_credits):
            continue
        raw_row = centered_scaled(raw_credits[t])
        for raw_value, verified_value in zip(raw_row, verified_row):
            if abs(float(raw_value)) <= 1e-8 and abs(float(verified_value)) <= 1e-8:
                continue
            raw_values.append(float(raw_value))
            verified_values.append(float(verified_value))
    if not raw_values:
        return 0.0, 0.0, False
    raw = np.asarray(raw_values, dtype=np.float64)
    verified = np.asarray(verified_values, dtype=np.float64)
    nonzero = (np.abs(raw) > 1e-8) & (np.abs(verified) > 1e-8)
    agreement = (
        float(np.mean(raw[nonzero] * verified[nonzero] > 0.0)) if bool(np.any(nonzero)) else 0.0
    )
    denominator = float(np.dot(raw, raw))
    scale = (
        0.0
        if denominator <= 1e-12
        else float(np.clip(np.dot(raw, verified) / denominator, 0.0, 1.0))
    )
    accepted = (
        int(np.count_nonzero(nonzero)) >= int(args.trajectory_scv_min_pairs)
        and agreement >= float(args.trajectory_scv_min_agreement)
        and scale > 0.0
    )
    return agreement, scale, accepted


def wilson_lower_bound(successes: int, trials: int, z: float) -> float:
    if trials <= 0:
        return 0.0
    z = max(0.0, float(z))
    rate = float(successes) / float(trials)
    if z <= 0.0:
        return rate
    z_squared = z * z
    denominator = 1.0 + z_squared / float(trials)
    center = rate + z_squared / (2.0 * float(trials))
    radius = z * math.sqrt(
        rate * (1.0 - rate) / float(trials) + z_squared / (4.0 * float(trials) * float(trials))
    )
    return float(np.clip((center - radius) / denominator, 0.0, 1.0))


def wilson_upper_bound(successes: int, trials: int, z: float) -> float:
    if trials <= 0:
        return 1.0
    z = max(0.0, float(z))
    rate = float(successes) / float(trials)
    if z <= 0.0:
        return rate
    z_squared = z * z
    denominator = 1.0 + z_squared / float(trials)
    center = rate + z_squared / (2.0 * float(trials))
    radius = z * math.sqrt(
        rate * (1.0 - rate) / float(trials) + z_squared / (4.0 * float(trials) * float(trials))
    )
    return float(np.clip((center + radius) / denominator, 0.0, 1.0))


def summarize_trajectory_scv_stats(
    stats: Mapping[str, object],
    args: argparse.Namespace | None = None,
    reward_density: float | None = None,
) -> Dict[str, object]:
    queries = int(stats.get("queries", 0))
    accepted = int(stats.get("accepted", 0))
    decision_accepted, decision_queries = trajectory_scv_decision_counts(stats, args)
    confidence_z = float(getattr(args, "trajectory_scv_global_confidence_z", 0.0))
    summary = {
        **dict(stats),
        "acceptance_rate": accepted / max(1, queries),
        "acceptance_lower_bound": wilson_lower_bound(accepted, queries, confidence_z),
        "acceptance_upper_bound": wilson_upper_bound(accepted, queries, confidence_z),
        "decision_queries": decision_queries,
        "decision_accepted": decision_accepted,
        "decision_acceptance_rate": decision_accepted / max(1, decision_queries),
        "decision_acceptance_lower_bound": wilson_lower_bound(
            decision_accepted,
            decision_queries,
            confidence_z,
        ),
        "decision_acceptance_upper_bound": wilson_upper_bound(
            decision_accepted,
            decision_queries,
            confidence_z,
        ),
        "confidence_z": confidence_z,
        "mean_agreement": float(stats.get("agreement_sum", 0.0)) / max(1, queries),
        "mean_scale": float(stats.get("scale_sum", 0.0)) / max(1, queries),
    }
    if args is not None:
        summary["base_calibration_state"] = trajectory_scv_global_state(stats, args)
        summary["calibration_state"] = trajectory_scv_trust_state(
            stats,
            args,
            reward_density=reward_density,
        )
        mode_stats = stats.get("rware_mode_stats")
        if isinstance(mode_stats, Mapping):
            summary["rware_mode_stats"] = summarize_rware_mode_scv_stats(mode_stats, args)
    return summary


def trajectory_scv_decision_counts(
    stats: Mapping[str, object],
    args: argparse.Namespace | None,
) -> Tuple[int, int]:
    rolling_window = max(0, int(getattr(args, "trajectory_scv_rolling_window", 0)))
    history = stats.get("recent_acceptance_history", [])
    if (
        rolling_window > 0
        and isinstance(history, Sequence)
        and not isinstance(history, (str, bytes))
    ):
        recent = [int(bool(value)) for value in history[-rolling_window:]]
        return sum(recent), len(recent)
    return int(stats.get("accepted", 0)), int(stats.get("queries", 0))


def trajectory_scv_global_state(
    stats: Mapping[str, object],
    args: argparse.Namespace,
    *,
    min_queries_override: int | None = None,
    min_acceptance_override: float | None = None,
    reject_upper_threshold_override: float | None = None,
) -> str:
    accepted, queries = trajectory_scv_decision_counts(stats, args)
    min_queries = (
        int(args.trajectory_scv_global_min_queries)
        if min_queries_override is None
        else int(min_queries_override)
    )
    min_acceptance = (
        float(args.trajectory_scv_global_min_acceptance)
        if min_acceptance_override is None
        else float(min_acceptance_override)
    )
    if queries >= min_queries:
        acceptance_lower_bound = wilson_lower_bound(
            accepted,
            queries,
            float(args.trajectory_scv_global_confidence_z),
        )
        if acceptance_lower_bound >= min_acceptance:
            return "certified"
        rejection_threshold = (
            float(args.trajectory_scv_global_reject_upper_threshold)
            if reject_upper_threshold_override is None
            else float(reject_upper_threshold_override)
        )
        if rejection_threshold >= 0.0:
            acceptance_upper_bound = wilson_upper_bound(
                accepted,
                queries,
                float(args.trajectory_scv_global_confidence_z),
            )
            if acceptance_upper_bound <= rejection_threshold:
                return "rejected"
    max_probation_audits = int(args.trajectory_scv_global_max_probation_audits)
    if max_probation_audits > 0 and int(stats.get("audits", 0)) >= max_probation_audits:
        if (
            bool(getattr(args, "trajectory_scv_global_cap_use_point_estimate", False))
            and queries >= min_queries
        ):
            acceptance_rate = accepted / max(1, queries)
            if acceptance_rate >= min_acceptance:
                return "certified"
        return "rejected"
    return "probation"


def trajectory_scv_trust_state(
    stats: Mapping[str, object],
    args: argparse.Namespace,
    reward_density: float | None,
) -> str:
    if not trajectory_scv_sparse_reward_regime(args, reward_density):
        return trajectory_scv_global_state(stats, args)
    if bool(stats.get("sparse_high_assurance_certified", False)):
        return trajectory_scv_global_state(stats, args)
    return trajectory_scv_global_state(
        stats,
        args,
        min_queries_override=int(getattr(args, "trajectory_scv_sparse_global_min_queries", 20)),
        min_acceptance_override=float(
            getattr(args, "trajectory_scv_sparse_global_min_acceptance", 0.8)
        ),
        reject_upper_threshold_override=float(
            getattr(
                args,
                "trajectory_scv_sparse_global_reject_upper_threshold",
                0.8,
            )
        ),
    )


def trajectory_scv_sparse_reward_regime(
    args: argparse.Namespace,
    reward_density: float | None,
) -> bool:
    sparse_density_threshold = float(
        getattr(args, "trajectory_scv_sparse_reward_density_threshold", -1.0)
    )
    return bool(
        reward_density is not None
        and sparse_density_threshold >= 0.0
        and float(reward_density) <= sparse_density_threshold
    )


def trajectory_scv_certified_confidence_weight(
    stats: Mapping[str, object],
    args: argparse.Namespace,
) -> float:
    """Scale certified credit by its causal-evidence margin above threshold."""
    queries = int(stats.get("queries", 0))
    if queries <= 0:
        return 0.0
    threshold = float(args.trajectory_scv_global_min_acceptance)
    lower_bound = wilson_lower_bound(
        int(stats.get("accepted", 0)),
        queries,
        float(args.trajectory_scv_global_confidence_z),
    )
    normalized_margin = float(
        np.clip((lower_bound - threshold) / max(1.0 - threshold, 1e-8), 0.0, 1.0)
    )
    power = max(float(args.trajectory_scv_confidence_weight_power), 1e-8)
    minimum = float(np.clip(args.trajectory_scv_confidence_weight_min, 0.0, 1.0))
    return float(minimum + (1.0 - minimum) * normalized_margin**power)


def trajectory_scv_schedule_resolved(
    stats: Mapping[str, object],
    args: argparse.Namespace,
    reward_density: float | None = None,
) -> bool:
    return trajectory_scv_trust_state(
        stats,
        args,
        reward_density=reward_density,
    ) in {"certified", "rejected"}


def probation_credit_scale(
    stats: Mapping[str, object],
    args: argparse.Namespace,
    n_agents: int,
) -> float:
    base_scale = (
        1.0 / max(1, int(n_agents))
        if args.uncertified_credit_scale_by_agents
        else float(args.uncertified_credit_scale)
    )
    audits = int(stats.get("audits", 0))
    queries = int(stats.get("queries", 0))
    accepted = int(stats.get("accepted", 0))
    if args.trajectory_scv_probation_wilson_scale_by_agents and queries > 0:
        min_queries = max(1, int(args.trajectory_scv_global_min_queries))
        threshold = max(1e-8, float(args.trajectory_scv_global_min_acceptance))
        lower_bound = wilson_lower_bound(
            accepted,
            queries,
            float(args.trajectory_scv_global_confidence_z),
        )
        evidence_fraction = min(1.0, queries / float(min_queries))
        confidence_fraction = min(1.0, lower_bound / threshold)
        evidence_scale = evidence_fraction * confidence_fraction / max(1, int(n_agents))
        base_scale = max(base_scale, evidence_scale)
    if (
        args.trajectory_scv_probation_supported_scale_by_agents
        and queries >= int(args.trajectory_scv_global_min_queries)
        and accepted / max(1, queries) >= float(args.trajectory_scv_global_min_acceptance)
        and queries / max(1, audits)
        <= float(args.trajectory_scv_probation_supported_max_informative_rate)
    ):
        base_scale = max(base_scale, 1.0 / max(1, int(n_agents)))
    escalation_start = int(args.trajectory_scv_probation_escalate_audits)
    if escalation_start <= 0:
        return base_scale
    if audits < escalation_start:
        return base_scale
    informative_rate = queries / max(1, audits)
    escalation_triggered = bool(stats.get("probation_escalation_triggered", False))
    if not escalation_triggered and informative_rate > float(
        args.trajectory_scv_probation_max_informative_rate
    ):
        return base_scale
    ramp_audits = max(1, int(args.trajectory_scv_probation_escalate_ramp_audits))
    progress = float(np.clip((audits - escalation_start + 1) / ramp_audits, 0.0, 1.0))
    configured_max_scale = (
        min(1.0, 2.0 / max(1, int(n_agents)))
        if args.trajectory_scv_probation_max_scale_by_agents
        else float(args.trajectory_scv_probation_max_scale)
    )
    max_scale = max(base_scale, configured_max_scale)
    return float(base_scale + progress * (max_scale - base_scale))


def rware_mode_scv_state(entry: Mapping[str, object], args: argparse.Namespace) -> str:
    pairs = int(entry.get("pairs", 0))
    accepted = int(entry.get("accepted", 0))
    min_pairs = max(1, int(args.trajectory_scv_rware_mode_min_pairs))
    if pairs >= min_pairs:
        lower = wilson_lower_bound(
            accepted,
            pairs,
            float(args.trajectory_scv_global_confidence_z),
        )
        if lower >= float(args.trajectory_scv_global_min_acceptance):
            return "certified"
        rejection_threshold = float(args.trajectory_scv_global_reject_upper_threshold)
        if rejection_threshold >= 0.0:
            upper = wilson_upper_bound(
                accepted,
                pairs,
                float(args.trajectory_scv_global_confidence_z),
            )
            if upper <= rejection_threshold:
                return "rejected"
    return "probation"


def rware_mode_scv_all_certified(stats: Mapping[str, object], args: argparse.Namespace) -> bool:
    mode_stats = stats.get("rware_mode_stats", {})
    if not isinstance(mode_stats, Mapping):
        return False
    return all(
        isinstance(mode_stats.get(mode), Mapping)
        and rware_mode_scv_state(mode_stats[mode], args) == "certified"  # type: ignore[index]
        for mode in ("search", "carry", "return")
    )


def rware_mode_credit_scale(
    entry: Mapping[str, object],
    args: argparse.Namespace,
    n_agents: int,
) -> float:
    state = rware_mode_scv_state(entry, args)
    if state == "certified":
        return 1.0
    if state == "rejected" or not args.trajectory_scv_rware_mode_wilson_scale_by_agents:
        return 0.0
    pairs = int(entry.get("pairs", 0))
    accepted = int(entry.get("accepted", 0))
    if pairs <= 0:
        return 0.0
    min_pairs = max(1, int(args.trajectory_scv_rware_mode_min_pairs))
    threshold = max(1e-8, float(args.trajectory_scv_global_min_acceptance))
    lower = wilson_lower_bound(
        accepted,
        pairs,
        float(args.trajectory_scv_global_confidence_z),
    )
    evidence_fraction = min(1.0, pairs / float(min_pairs))
    confidence_fraction = min(1.0, lower / threshold)
    return float(evidence_fraction * confidence_fraction / max(1, int(n_agents)))


def apply_rware_mode_credit_gate(
    credits: Sequence[Sequence[float]],
    transitions: Sequence[Mapping[str, object]],
    stats: Mapping[str, object],
    args: argparse.Namespace,
    n_agents: int,
) -> List[List[float]]:
    mode_stats = stats.get("rware_mode_stats", {})
    if not isinstance(mode_stats, Mapping):
        mode_stats = {}
    out: List[List[float]] = []
    for t, row in enumerate(credits):
        before = transitions[t].get("credit_global_before") if t < len(transitions) else None
        scaled = []
        for agent_idx, value in enumerate(row):
            mode = "unknown"
            if (
                isinstance(before, list)
                and agent_idx < len(before)
                and isinstance(before[agent_idx], Mapping)
            ):
                mode = str(before[agent_idx].get("mode", "unknown"))
            entry = mode_stats.get(mode, {})
            if not isinstance(entry, Mapping):
                entry = {}
            scale = rware_mode_credit_scale(entry, args, n_agents)
            scaled.append(scale * float(value))
        out.append(centered_clipped(scaled))
    return out


def summarize_rware_mode_scv_stats(
    mode_stats: Mapping[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for mode, raw_entry in sorted(mode_stats.items()):
        if not isinstance(raw_entry, Mapping):
            continue
        pairs = int(raw_entry.get("pairs", 0))
        accepted = int(raw_entry.get("accepted", 0))
        out[str(mode)] = {
            "pairs": pairs,
            "accepted": accepted,
            "acceptance_rate": accepted / max(1, pairs),
            "acceptance_lower_bound": wilson_lower_bound(
                accepted,
                pairs,
                float(args.trajectory_scv_global_confidence_z),
            ),
            "acceptance_upper_bound": wilson_upper_bound(
                accepted,
                pairs,
                float(args.trajectory_scv_global_confidence_z),
            ),
            "state": rware_mode_scv_state(raw_entry, args),
        }
    return out


def proposed_credit(
    observations: Sequence[np.ndarray],
    next_observations: Sequence[np.ndarray],
    rewards: Sequence[float],
    team_reward: float,
    actions: Sequence[int],
    n_actions: int,
    env_family: str,
    n_agents: int,
    args: argparse.Namespace,
    use_lbf_struct_credit: bool = True,
    use_rware_struct_credit: bool = True,
) -> List[float]:
    if env_family == "lbf":
        return lbf_program_credit(
            observations, next_observations, rewards, team_reward, actions, n_agents, CREDIT_PROGRAM
        )
    if env_family in {"rware", "pursuit"}:
        return [0.0 for _ in range(n_agents)]
    raise ValueError(f"Unsupported SCV environment family {env_family!r}")


def proposal_actions_for_observations(
    observations: Sequence[np.ndarray],
    env_family: str,
    n_agents: int,
    args: argparse.Namespace,
    source: str | None = None,
    env: StandardMARLEnv | None = None,
) -> List[int]:
    action_source = "none" if source is None else source
    return [-1 for _ in observations]


def rware_struct_features(obs: Sequence[float], action: int) -> Dict[str, float]:
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if arr.size < 8:
        return empty_rware_struct_features()
    carrying = float(arr[2])
    on_highway = float(arr[7])
    sensor = arr[8:]
    sensor_width = 7
    n_sensor = sensor.size // sensor_width
    visible_shelves = 0.0
    requested_shelves = 0.0
    for idx in range(n_sensor):
        base = idx * sensor_width
        visible_shelves += float(sensor[base + 5])
        requested_shelves += float(sensor[base + 6])
    is_forward = 1.0 if int(action) == 1 else 0.0
    is_toggle = 1.0 if int(action) == 4 else 0.0
    has_request = 1.0 if requested_shelves > 0.0 else 0.0
    has_shelf = 1.0 if visible_shelves > 0.0 else 0.0
    return {
        "carrying": carrying,
        "on_highway": on_highway,
        "visible_shelves": visible_shelves,
        "requested_shelves": requested_shelves,
        "forward_with_request": is_forward * has_request,
        "forward_with_shelf": is_forward * has_shelf,
        "toggle_near_request": is_toggle * has_request,
        "toggle_near_shelf": is_toggle * has_shelf,
        "invalid_toggle": is_toggle * (1.0 - max(has_request, has_shelf, carrying)),
        "carrying_forward": carrying * is_forward,
    }


def empty_rware_struct_features() -> Dict[str, float]:
    return {
        "carrying": 0.0,
        "on_highway": 0.0,
        "visible_shelves": 0.0,
        "requested_shelves": 0.0,
        "forward_with_request": 0.0,
        "forward_with_shelf": 0.0,
        "toggle_near_request": 0.0,
        "toggle_near_shelf": 0.0,
        "invalid_toggle": 0.0,
        "carrying_forward": 0.0,
    }


LBF_CREDIT_PROGRAM_FEATURES = (
    "bias",
    "reward_residual",
    "actual_progress_positive",
    "actual_progress_negative",
    "intended_progress_positive",
    "intended_progress_negative",
    "action_match",
    "valid_load",
    "invalid_load",
    "help_progress",
    "help_adjacent",
)


RWARE_GLOBAL_CREDIT_PROGRAM_FEATURES = (
    "bias",
    "reward_residual",
    "search_progress_positive",
    "search_progress_negative",
    "carry_progress_positive",
    "carry_progress_negative",
    "return_progress_positive",
    "return_progress_negative",
    "pickup_requested",
    "pickup_unrequested",
    "delivery",
    "return_drop",
    "premature_drop",
    "blocked_forward",
    "invalid_toggle",
    "idle",
)


def load_credit_program(path: Path) -> Dict[str, object]:
    payload = get_program(Path(path).name)
    if not isinstance(payload, dict):
        raise ValueError("credit program must be a JSON object")
    program = payload.get("program", payload)
    if not isinstance(program, dict):
        raise ValueError("credit program field must be a JSON object")
    weights = program.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("credit program must contain a weights object")
    weights = dict(weights)
    environment = str(program.get("environment", "lbf")).lower()
    if environment == "rware_global":
        feature_names = RWARE_GLOBAL_CREDIT_PROGRAM_FEATURES
        for name in (
            "return_progress_positive",
            "return_progress_negative",
            "pickup_unrequested",
            "return_drop",
            "premature_drop",
        ):
            weights.setdefault(name, 0.0)
    else:
        feature_names = LBF_CREDIT_PROGRAM_FEATURES
    missing = [name for name in feature_names if name not in weights]
    if missing:
        raise ValueError(f"credit program is missing weights: {missing}")
    numeric_weights: Dict[str, float] = {}
    for name in feature_names:
        value = float(weights[name])
        if not math.isfinite(value):
            raise ValueError(f"credit program weight {name} is not finite")
        numeric_weights[name] = float(np.clip(value, -3.0, 3.0))
    clip = float(program.get("clip", 3.0))
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError("credit program clip must be a positive finite value")
    return {
        **program,
        "weights": numeric_weights,
        "environment": environment,
        "clip": float(np.clip(clip, 0.1, 10.0)),
    }


def lbf_program_credit(
    observations: Sequence[np.ndarray],
    next_observations: Sequence[np.ndarray],
    rewards: Sequence[float],
    team_reward: float,
    actions: Sequence[int],
    n_agents: int,
    program: Mapping[str, object],
) -> List[float]:
    weights = program.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("loaded LBF credit program has no weights")
    values = []
    for obs, next_obs, reward, action in zip(observations, next_observations, rewards, actions):
        parsed = parse_lbf_obs(obs, n_agents)
        next_parsed = parse_lbf_obs(next_obs, n_agents)
        if parsed is None or next_parsed is None or not parsed["foods"]:
            values.append(float(reward - team_reward))
            continue
        target = choose_lbf_task_target(parsed)
        if target is None:
            values.append(float(reward - team_reward))
            continue

        target_row, target_col, target_level = target
        self_row, self_col, self_level = parsed["self"]
        next_row, next_col, _next_level = next_parsed["self"]
        before_dist = abs(target_row - self_row) + abs(target_col - self_col)
        after_dist = abs(target_row - next_row) + abs(target_col - next_col)
        actual_progress = float(np.clip(before_dist - after_dist, -1.0, 1.0))
        recommended = lbf_task_prior_action(obs, n_agents)
        adjacent = 1.0 if before_dist == 1.0 else 0.0
        load = 1.0 if int(action) == 5 else 0.0
        action_match = 1.0 if int(action) == int(recommended) and recommended >= 0 else 0.0
        action_delta = lbf_action_delta(action)
        proposed_dist = abs(target_row - (self_row + action_delta[0])) + abs(
            target_col - (self_col + action_delta[1])
        )
        intended_progress = float(np.clip(before_dist - proposed_dist, -1.0, 1.0))
        adjacent_level = self_level if adjacent else 0.0
        for row, col, level in parsed["others"]:
            if abs(target_row - row) + abs(target_col - col) == 1.0:
                adjacent_level += level
        enough_to_load = 1.0 if adjacent_level >= target_level else 0.0
        needs_help = 1.0 if target_level > self_level else 0.0
        features = {
            "bias": 1.0,
            "reward_residual": float(reward - team_reward),
            "actual_progress_positive": max(0.0, actual_progress),
            "actual_progress_negative": max(0.0, -actual_progress),
            "intended_progress_positive": max(0.0, intended_progress),
            "intended_progress_negative": max(0.0, -intended_progress),
            "action_match": action_match,
            "valid_load": load * adjacent * max(0.5, enough_to_load),
            "invalid_load": load * (1.0 - adjacent),
            "help_progress": needs_help * max(0.0, intended_progress),
            "help_adjacent": needs_help * adjacent,
        }
        values.append(
            sum(
                float(weights[name]) * float(features[name]) for name in LBF_CREDIT_PROGRAM_FEATURES
            )
        )
    return clip_row(values, limit=float(program.get("clip", 3.0)))


def rware_global_credit_snapshot(env: StandardMARLEnv) -> List[Dict[str, object]]:
    base = getattr(env._env, "unwrapped", env._env)
    if base is None or not hasattr(base, "agents"):
        return []
    requested_ids = {int(shelf.id) for shelf in getattr(base, "request_queue", [])}
    carrying_ids = {
        int(agent.carrying_shelf.id)
        for agent in base.agents
        if getattr(agent, "carrying_shelf", None) is not None
    }
    requested = [
        shelf for shelf in getattr(base, "request_queue", []) if int(shelf.id) not in carrying_ids
    ]
    shelf_by_id = {int(shelf.id): shelf for shelf in requested}
    unique_assignment = bool(CREDIT_PROGRAM.get("unique_assignment", False))
    assignments: Dict[int, int] = {}
    if unique_assignment:
        if int(getattr(base, "_cur_steps", 0)) == 0:
            setattr(base, "_aamas_credit_shelf_assignments", {})
        cached_assignments = dict(getattr(base, "_aamas_credit_shelf_assignments", {}))
        used_shelf_ids = set()
        for agent_idx, shelf_id in sorted(cached_assignments.items()):
            if not 0 <= int(agent_idx) < len(base.agents):
                continue
            agent = base.agents[int(agent_idx)]
            if (
                getattr(agent, "carrying_shelf", None) is not None
                or int(shelf_id) not in shelf_by_id
            ):
                continue
            assignments[int(agent_idx)] = int(shelf_id)
            used_shelf_ids.add(int(shelf_id))
        free_agent_indexes = [
            agent_idx
            for agent_idx, agent in enumerate(base.agents)
            if getattr(agent, "carrying_shelf", None) is None and agent_idx not in assignments
        ]
        free_shelves = [
            shelf
            for shelf in sorted(requested, key=lambda item: int(item.id))
            if int(shelf.id) not in used_shelf_ids
        ]
        assignments.update(
            minimum_cost_rware_shelf_assignments(base.agents, free_agent_indexes, free_shelves)
        )
        setattr(base, "_aamas_credit_shelf_assignments", dict(assignments))
    goals = [(float(goal[0]), float(goal[1])) for goal in getattr(base, "goals", [])]
    occupied_storage = {
        (int(shelf.x), int(shelf.y))
        for shelf in getattr(base, "shelfs", [])
        if int(shelf.id) not in carrying_ids
    }
    highways = np.asarray(getattr(base, "highways", np.ones((0, 0))), dtype=np.int8)
    empty_storage: List[Tuple[float, float]] = []
    if highways.ndim == 2:
        for y in range(highways.shape[0]):
            for x in range(highways.shape[1]):
                if highways[y, x] == 0 and (x, y) not in occupied_storage:
                    empty_storage.append((float(x), float(y)))
    rows: List[Dict[str, object]] = []
    for agent_idx, agent in enumerate(base.agents):
        x, y = float(agent.x), float(agent.y)
        carrying_shelf = getattr(agent, "carrying_shelf", None)
        carrying_shelf_id = int(carrying_shelf.id) if carrying_shelf is not None else -1
        carrying_requested = carrying_shelf_id in requested_ids
        if carrying_shelf is not None and carrying_requested:
            targets = [(gx, gy, f"goal:{int(gx)}:{int(gy)}") for gx, gy in goals]
            mode = "carry"
        elif carrying_shelf is None:
            if unique_assignment:
                assigned_shelf = shelf_by_id.get(assignments.get(agent_idx, -1))
                targets = (
                    [
                        (
                            float(assigned_shelf.x),
                            float(assigned_shelf.y),
                            f"shelf:{int(assigned_shelf.id)}",
                        )
                    ]
                    if assigned_shelf is not None
                    else []
                )
                mode = "search" if targets else "support"
            else:
                targets = [
                    (float(shelf.x), float(shelf.y), f"shelf:{int(shelf.id)}")
                    for shelf in requested
                ]
                mode = "search"
        else:
            targets = [(sx, sy, f"storage:{int(sx)}:{int(sy)}") for sx, sy in empty_storage]
            mode = "return"
        if targets:
            tx, ty, target_id = min(targets, key=lambda item: abs(item[0] - x) + abs(item[1] - y))
            distance = abs(tx - x) + abs(ty - y)
        else:
            target_id = "none"
            distance = -1.0
        rows.append(
            {
                "x": x,
                "y": y,
                "mode": mode,
                "target_id": target_id,
                "distance": float(distance),
                "carrying": carrying_shelf is not None,
                "carrying_shelf_id": carrying_shelf_id,
                "carrying_requested": carrying_requested,
            }
        )
    return rows


def minimum_cost_rware_shelf_assignments(
    agents: Sequence[object],
    agent_indexes: Sequence[int],
    shelves: Sequence[object],
) -> Dict[int, int]:
    """Assign each available requested shelf to at most one searching agent."""

    if not agent_indexes or not shelves:
        return {}
    n_pairs = min(len(agent_indexes), len(shelves))
    best_key: Tuple[float, Tuple[Tuple[int, int], ...]] | None = None
    best: Dict[int, int] = {}
    for chosen_agents in itertools.combinations(sorted(agent_indexes), n_pairs):
        for chosen_shelves in itertools.combinations(
            sorted(shelves, key=lambda item: int(getattr(item, "id"))),
            n_pairs,
        ):
            for shelf_order in itertools.permutations(chosen_shelves):
                pairs = tuple(
                    (int(agent_idx), int(getattr(shelf, "id")))
                    for agent_idx, shelf in zip(chosen_agents, shelf_order)
                )
                cost = sum(
                    abs(float(getattr(agents[agent_idx], "x")) - float(getattr(shelf, "x")))
                    + abs(float(getattr(agents[agent_idx], "y")) - float(getattr(shelf, "y")))
                    for agent_idx, shelf in zip(chosen_agents, shelf_order)
                )
                key = (float(cost), pairs)
                if best_key is None or key < best_key:
                    best_key = key
                    best = {agent_idx: shelf_id for agent_idx, shelf_id in pairs}
    return best


def rware_global_program_credit(
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
    rewards: Sequence[float],
    team_reward: float,
    actions: Sequence[int],
    program: Mapping[str, object],
    scale: float = 1.0,
) -> List[float]:
    weights = program.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("loaded global RWARE credit program has no weights")
    values = []
    for current, nxt, reward, action in zip(before, after, rewards, actions):
        same_target = (
            str(current.get("target_id")) == str(nxt.get("target_id"))
            and str(current.get("target_id")) != "none"
        )
        before_dist = float(current.get("distance", -1.0))
        after_dist = float(nxt.get("distance", -1.0))
        progress = (
            before_dist - after_dist
            if same_target and before_dist >= 0.0 and after_dist >= 0.0
            else 0.0
        )
        moved = (
            abs(float(current.get("x", 0.0)) - float(nxt.get("x", 0.0)))
            + abs(float(current.get("y", 0.0)) - float(nxt.get("y", 0.0)))
        ) > 0.0
        carrying_before = bool(current.get("carrying", False))
        carrying_after = bool(nxt.get("carrying", False))
        carrying_requested_after = bool(nxt.get("carrying_requested", False))
        mode = str(current.get("mode", ""))
        picked_up = not carrying_before and carrying_after
        dropped = carrying_before and not carrying_after
        features = {
            "bias": 1.0,
            "reward_residual": float(reward - team_reward),
            "search_progress_positive": max(0.0, progress) if mode == "search" else 0.0,
            "search_progress_negative": max(0.0, -progress) if mode == "search" else 0.0,
            "carry_progress_positive": max(0.0, progress) if mode == "carry" else 0.0,
            "carry_progress_negative": max(0.0, -progress) if mode == "carry" else 0.0,
            "return_progress_positive": max(0.0, progress) if mode == "return" else 0.0,
            "return_progress_negative": max(0.0, -progress) if mode == "return" else 0.0,
            "pickup_requested": 1.0 if picked_up and carrying_requested_after else 0.0,
            "pickup_unrequested": 1.0 if picked_up and not carrying_requested_after else 0.0,
            "delivery": 1.0 if float(reward) > 0.0 else 0.0,
            "return_drop": 1.0 if dropped and mode == "return" else 0.0,
            "premature_drop": 1.0 if dropped and mode == "carry" and float(reward) <= 0.0 else 0.0,
            "blocked_forward": 1.0 if int(action) == 1 and not moved else 0.0,
            "invalid_toggle": (
                1.0 if int(action) == 4 and carrying_before == carrying_after else 0.0
            ),
            "idle": 1.0 if int(action) == 0 else 0.0,
        }
        values.append(
            float(scale)
            * sum(
                float(weights[name]) * float(features[name])
                for name in RWARE_GLOBAL_CREDIT_PROGRAM_FEATURES
            )
        )
    return clip_row(values, limit=float(program.get("clip", 3.0)))


def apply_rware_temporal_program_credit(
    transitions: Sequence[Dict[str, object]], args: argparse.Namespace
) -> None:
    horizon = max(0, int(args.credit_program_temporal_horizon))
    if horizon <= 0:
        return
    discount = float(np.clip(args.credit_program_temporal_discount, 0.0, 1.0))
    scale = max(0.0, float(args.credit_program_temporal_scale))
    for event_t, event in enumerate(transitions):
        rewards = [float(value) for value in event["rewards"]]
        for agent_idx, reward in enumerate(rewards):
            if reward <= float(args.cf_reward_threshold):
                continue
            earliest = max(0, event_t - horizon + 1)
            pickup_t = event_t
            for t in range(event_t, earliest - 1, -1):
                transition = transitions[t]
                action = int(transition["actions"][agent_idx])
                current = rware_struct_features(transition["obs"][agent_idx], action)
                nxt = rware_struct_features(transition["next_obs"][agent_idx], action)
                if nxt["carrying"] - current["carrying"] > 0.5:
                    pickup_t = t
                    break
            start_t = earliest
            for t in range(start_t, event_t + 1):
                transition = transitions[t]
                action = int(transition["actions"][agent_idx])
                current = rware_struct_features(transition["obs"][agent_idx], action)
                nxt = rware_struct_features(transition["next_obs"][agent_idx], action)
                carrying_path = max(current["carrying"], nxt["carrying"])
                before_pickup = t < pickup_t
                if carrying_path <= 0.5 and (not before_pickup) and (t not in {pickup_t, event_t}):
                    continue
                phase_scale = (
                    float(args.credit_program_temporal_search_scale) if before_pickup else 1.0
                )
                if phase_scale <= 0.0:
                    continue
                action_factor = 0.5 if action == 0 else 1.0
                lag = event_t - t
                addition = scale * phase_scale * reward * discount**lag * action_factor
                row = [float(value) for value in transition["raw_credit"]]
                row[agent_idx] += float(addition)
                transition["raw_credit"] = clip_row(
                    row, limit=float(CREDIT_PROGRAM.get("clip", 3.0))
                )


def parse_lbf_obs(obs: Sequence[float], n_agents: int) -> Dict[str, object] | None:
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if n_agents <= 0 or arr.size < 3 * n_agents:
        return None
    n_food = max(0, (arr.size - 3 * n_agents) // 3)
    foods: List[Tuple[float, float, float]] = []
    for idx in range(n_food):
        row, col, level = arr[3 * idx : 3 * idx + 3]
        if row >= 0.0 and col >= 0.0 and level > 0.0:
            foods.append((float(row), float(col), float(level)))
    player_offset = 3 * n_food
    self_row, self_col, self_level = arr[player_offset : player_offset + 3]
    others: List[Tuple[float, float, float]] = []
    for idx in range(1, n_agents):
        start = player_offset + 3 * idx
        if start + 2 >= arr.size:
            continue
        row, col, level = arr[start : start + 3]
        if row >= 0.0 and col >= 0.0 and level > 0.0:
            others.append((float(row), float(col), float(level)))
    return {
        "foods": foods,
        "self": (float(self_row), float(self_col), float(self_level)),
        "others": others,
    }


def choose_lbf_task_target(parsed: Mapping[str, object]) -> Tuple[float, float, float] | None:
    foods = list(parsed["foods"])
    if not foods:
        return None
    (self_row, self_col, self_level) = parsed["self"]
    others = list(parsed["others"])
    best = None
    best_score = 1e309
    for food in foods:
        (row, col, level) = food
        dist = abs(row - self_row) + abs(col - self_col)
        teammate_potential = sum(
            (
                other_level / (1.0 + abs(row - other_row) + abs(col - other_col))
                for (other_row, other_col, other_level) in others
            )
        )
        enough_eventually = 1.0 if self_level + teammate_potential >= 0.55 * level else 0.0
        score = (
            dist
            + 0.35 * max(0.0, level - self_level)
            - 0.35 * teammate_potential
            - 0.4 * enough_eventually
        )
        if score < best_score:
            best_score = float(score)
            best = food
    return best


def lbf_task_prior_action(obs: Sequence[float], n_agents: int) -> int:
    parsed = parse_lbf_obs(obs, n_agents)
    if parsed is None:
        return -1
    target = choose_lbf_task_target(parsed)
    if target is None:
        return 0
    target_row, target_col, _target_level = target
    self_row, self_col, _self_level = parsed["self"]  # type: ignore[misc]
    dist = abs(target_row - self_row) + abs(target_col - self_col)
    if dist == 1.0:
        return 5
    row_delta = target_row - self_row
    col_delta = target_col - self_col
    if abs(row_delta) >= abs(col_delta) and abs(row_delta) > 0.0:
        return 2 if row_delta > 0.0 else 1
    if abs(col_delta) > 0.0:
        return 4 if col_delta > 0.0 else 3
    return 0


def task_hint_dim(args: argparse.Namespace, env_info: Mapping[str, int | str]) -> int:
    return 0


def scheduled_task_hint_weight(args: argparse.Namespace, env_steps_seen: int) -> float:
    return 0.0


def scheduled_task_hint_dropout(args: argparse.Namespace, env_steps_seen: int) -> float:
    return 0.0


def task_hints_for_observations(
    observations: Sequence[np.ndarray],
    env_family: str,
    n_agents: int,
    args: argparse.Namespace | None,
    task_hint_weight: float,
    task_hint_dropout: float = 0.0,
    hint_actions: Sequence[int] | None = None,
    hint_targets: Sequence[object] | None = None,
) -> List[np.ndarray]:
    return []


def task_hint_actions_for_observations(
    observations: Sequence[np.ndarray],
    env_family: str,
    n_agents: int,
    args: argparse.Namespace | None,
    env: StandardMARLEnv | None = None,
) -> List[int] | None:
    if args is None:
        return None
    return None


def task_hint_targets_for_observations(
    observations: Sequence[np.ndarray],
    env_family: str,
    n_agents: int,
    args: argparse.Namespace | None,
    env: StandardMARLEnv | None = None,
) -> List[object] | None:
    if args is None:
        return None
    return None


def lbf_struct_features(obs: Sequence[float], action: int, n_agents: int) -> Dict[str, float]:
    arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    if n_agents <= 0 or arr.size < 3 * n_agents:
        return empty_lbf_struct_features()
    n_food = max(0, (arr.size - 3 * n_agents) // 3)
    foods = []
    for idx in range(n_food):
        row, col, level = arr[3 * idx : 3 * idx + 3]
        if row >= 0.0 and col >= 0.0 and level > 0.0:
            foods.append((float(row), float(col), float(level)))
    player_offset = 3 * n_food
    self_row, self_col, self_level = arr[player_offset : player_offset + 3]
    other_players = []
    for idx in range(1, n_agents):
        start = player_offset + 3 * idx
        if start + 2 >= arr.size:
            continue
        row, col, level = arr[start : start + 3]
        if row >= 0.0 and col >= 0.0 and level > 0.0:
            other_players.append((float(row), float(col), float(level)))

    nearest = None
    nearest_dist = -1.0
    if foods:
        nearest = min(foods, key=lambda item: abs(item[0] - self_row) + abs(item[1] - self_col))
        nearest_dist = float(abs(nearest[0] - self_row) + abs(nearest[1] - self_col))

    move_row, move_col = lbf_action_delta(action)
    proposed_row = float(self_row + move_row)
    proposed_col = float(self_col + move_col)
    proposed_dist = nearest_dist
    if nearest is not None:
        proposed_dist = float(abs(nearest[0] - proposed_row) + abs(nearest[1] - proposed_col))
    action_progress = 0.0
    if nearest_dist >= 0.0 and proposed_dist >= 0.0:
        action_progress = float(np.clip(nearest_dist - proposed_dist, -1.0, 1.0))

    adjacent_nearest = 1.0 if nearest_dist == 1.0 else 0.0
    is_load = 1.0 if int(action) == 5 else 0.0
    teammate_adjacent = 0.0
    teammate_level_adjacent = 0.0
    if nearest is not None:
        for row, col, level in other_players:
            if abs(row - nearest[0]) + abs(col - nearest[1]) == 1.0:
                teammate_adjacent = 1.0
                teammate_level_adjacent += level
    can_solo = 1.0 if nearest is not None and self_level >= nearest[2] else 0.0
    can_group = (
        1.0 if nearest is not None and self_level + teammate_level_adjacent >= nearest[2] else 0.0
    )
    return {
        "food_count": float(len(foods)),
        "nearest_food_dist": nearest_dist,
        "nearest_food_level": float(nearest[2]) if nearest is not None else 0.0,
        "self_level": float(self_level),
        "load_adjacent_food": is_load * adjacent_nearest,
        "invalid_load": is_load * (1.0 - adjacent_nearest),
        "action_toward_food": max(0.0, action_progress),
        "action_away_from_food": max(0.0, -action_progress),
        "adjacent_nearest": adjacent_nearest,
        "can_solo_nearest": can_solo,
        "can_group_nearest": can_group,
        "teammate_adjacent_nearest": teammate_adjacent,
        "teammate_level_adjacent": float(teammate_level_adjacent),
    }


def empty_lbf_struct_features() -> Dict[str, float]:
    return {
        "food_count": 0.0,
        "nearest_food_dist": -1.0,
        "nearest_food_level": 0.0,
        "self_level": 0.0,
        "load_adjacent_food": 0.0,
        "invalid_load": 0.0,
        "action_toward_food": 0.0,
        "action_away_from_food": 0.0,
        "adjacent_nearest": 0.0,
        "can_solo_nearest": 0.0,
        "can_group_nearest": 0.0,
        "teammate_adjacent_nearest": 0.0,
        "teammate_level_adjacent": 0.0,
    }


def lbf_action_delta(action: int) -> Tuple[float, float]:
    if int(action) == 1:
        return -1.0, 0.0
    if int(action) == 2:
        return 1.0, 0.0
    if int(action) == 3:
        return 0.0, -1.0
    if int(action) == 4:
        return 0.0, 1.0
    return 0.0, 0.0


def feature_rows(
    transitions: Sequence[Dict[str, object]],
    raw_credits: Sequence[Sequence[float]],
    n_actions: int,
    args: argparse.Namespace | None = None,
) -> List[List[List[float]]]:
    rows = []
    for step_idx, (transition, raw_row) in enumerate(zip(transitions, raw_credits)):
        row = []
        rewards = [float(value) for value in transition["rewards"]]  # type: ignore[index]
        obs_list = [np.asarray(obs, dtype=np.float64) for obs in transition["obs"]]  # type: ignore[index]
        next_obs_list = [np.asarray(obs, dtype=np.float64) for obs in transition["next_obs"]]  # type: ignore[index]
        actions = [int(value) for value in transition["actions"]]  # type: ignore[index]
        n_agents = len(actions)
        for agent_idx, raw_value in enumerate(raw_row):
            action = actions[agent_idx]
            one_hot = [1.0 if action == idx else 0.0 for idx in range(n_actions)]
            base_features = [
                1.0,
                float(raw_value),
                float(transition["team_reward"]),
                rewards[agent_idx],
                float(np.linalg.norm(obs_list[agent_idx])),
                float(np.linalg.norm(next_obs_list[agent_idx] - obs_list[agent_idx])),
                float(action) / max(n_actions - 1, 1),
                *one_hot,
            ]
            if args is not None and args.lbf_struct_credit and str(args.env).startswith("lbf/"):
                current = lbf_struct_features(obs_list[agent_idx], action, n_agents)
                nxt = lbf_struct_features(next_obs_list[agent_idx], action, n_agents)
                before = current["nearest_food_dist"]
                after = nxt["nearest_food_dist"]
                progress = 0.0 if before < 0.0 or after < 0.0 else float(before - after)
                base_features.extend(
                    [
                        current["food_count"],
                        current["nearest_food_dist"],
                        current["nearest_food_level"],
                        current["self_level"],
                        current["load_adjacent_food"],
                        current["invalid_load"],
                        current["action_toward_food"],
                        current["action_away_from_food"],
                        current["adjacent_nearest"],
                        current["can_solo_nearest"],
                        current["can_group_nearest"],
                        current["teammate_adjacent_nearest"],
                        current["teammate_level_adjacent"],
                        progress,
                    ]
                )
            if args is not None and args.rware_struct_credit and str(args.env).startswith("rware/"):
                current = rware_struct_features(obs_list[agent_idx], action)
                nxt = rware_struct_features(next_obs_list[agent_idx], action)
                base_features.extend(
                    [
                        current["carrying"],
                        nxt["carrying"],
                        nxt["carrying"] - current["carrying"],
                        current["on_highway"],
                        nxt["on_highway"],
                        current["visible_shelves"],
                        nxt["visible_shelves"],
                        nxt["visible_shelves"] - current["visible_shelves"],
                        current["requested_shelves"],
                        nxt["requested_shelves"],
                        nxt["requested_shelves"] - current["requested_shelves"],
                        current["forward_with_request"],
                        current["forward_with_shelf"],
                        current["toggle_near_request"],
                        current["toggle_near_shelf"],
                        current["invalid_toggle"],
                        current["carrying_forward"],
                    ]
                )
            if args is not None and bool(getattr(args, "credit_feature_global_summary", False)):
                base_features.extend(
                    credit_global_summary_features(
                        transition=transition,
                        focal_agent_idx=agent_idx,
                        n_agents=n_agents,
                    )
                )
            if args is not None and int(getattr(args, "credit_feature_context_radius", 0)) > 0:
                base_features.extend(
                    credit_temporal_context_features(
                        transitions=transitions,
                        raw_credits=raw_credits,
                        step_idx=step_idx,
                        agent_idx=agent_idx,
                        n_actions=n_actions,
                        radius=int(args.credit_feature_context_radius),
                    )
                )
            row.append(base_features)
        rows.append(row)
    return rows


def credit_global_summary_features(
    transition: Mapping[str, object],
    focal_agent_idx: int,
    n_agents: int,
) -> List[float]:
    """Permutation-relative centralized features used only by the credit student."""
    before = transition.get("credit_global_before")
    after = transition.get("credit_global_after")
    if not isinstance(before, list) or not isinstance(after, list):
        return [0.0 for _ in range(13 * n_agents + 4)]
    if len(before) < n_agents or len(after) < n_agents:
        return [0.0 for _ in range(13 * n_agents + 4)]

    focal = before[focal_agent_idx]
    focal_target = str(focal.get("target_id", "none")) if isinstance(focal, Mapping) else "none"
    values: List[float] = []
    distances = []
    carrying = 0.0
    carrying_requested = 0.0
    for offset in range(n_agents):
        agent_idx = (focal_agent_idx + offset) % n_agents
        current = before[agent_idx]
        nxt = after[agent_idx]
        if not isinstance(current, Mapping) or not isinstance(nxt, Mapping):
            values.extend([0.0 for _ in range(13)])
            continue
        distance = float(current.get("distance", -1.0))
        next_distance = float(nxt.get("distance", -1.0))
        valid_distance = distance >= 0.0
        progress = distance - next_distance if valid_distance and next_distance >= 0.0 else 0.0
        current_carrying = 1.0 if bool(current.get("carrying", False)) else 0.0
        current_requested = 1.0 if bool(current.get("carrying_requested", False)) else 0.0
        carrying += current_carrying
        carrying_requested += current_requested
        if valid_distance:
            distances.append(distance)
        mode = str(current.get("mode", ""))
        values.extend(
            [
                current_carrying,
                current_requested,
                float(np.tanh(max(distance, 0.0) / 10.0)) if valid_distance else 0.0,
                float(np.tanh(progress / 5.0)),
                float(np.tanh(float(current.get("x", 0.0)) / 10.0)),
                float(np.tanh(float(current.get("y", 0.0)) / 10.0)),
                float(np.tanh((float(nxt.get("x", 0.0)) - float(current.get("x", 0.0))) / 2.0)),
                float(np.tanh((float(nxt.get("y", 0.0)) - float(current.get("y", 0.0))) / 2.0)),
                1.0 if mode == "search" else 0.0,
                1.0 if mode == "carry" else 0.0,
                1.0 if mode == "return" else 0.0,
                1.0 if mode == "support" else 0.0,
                1.0 if str(current.get("target_id", "none")) == focal_target else 0.0,
            ]
        )
    values.extend(
        [
            carrying / max(1, n_agents),
            carrying_requested / max(1, n_agents),
            float(np.tanh(mean(distances) / 10.0)) if distances else 0.0,
            float(np.tanh(min(distances) / 10.0)) if distances else 0.0,
        ]
    )
    return values


def credit_temporal_context_features(
    transitions: Sequence[Mapping[str, object]],
    raw_credits: Sequence[Sequence[float]],
    step_idx: int,
    agent_idx: int,
    n_actions: int,
    radius: int,
) -> List[float]:
    values: List[float] = []
    radius = max(0, int(radius))
    for offset in [*range(-radius, 0), *range(1, radius + 1)]:
        index = step_idx + offset
        if index < 0 or index >= len(transitions):
            values.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        transition = transitions[index]
        centered = centered_scaled(raw_credits[index])
        actions = [int(value) for value in transition["actions"]]  # type: ignore[index]
        observations = transition["obs"]  # type: ignore[index]
        next_observations = transition["next_obs"]  # type: ignore[index]
        obs_delta = float(
            np.linalg.norm(
                np.asarray(next_observations[agent_idx], dtype=np.float64)
                - np.asarray(observations[agent_idx], dtype=np.float64)
            )
        )
        values.extend(
            [
                1.0,
                float(centered[agent_idx]),
                float(transition["team_reward"]),
                float(actions[agent_idx]) / max(1, int(n_actions) - 1),
                float(np.tanh(obs_delta)),
            ]
        )
    return values


def counterfactual_trajectory_credits(
    args: argparse.Namespace,
    episode_seed: int,
    transitions: Sequence[Dict[str, object]],
    env_info: Mapping[str, int | str],
    query_indexes: Sequence[int],
    stats: Dict[str, int],
    actor: nn.Module | None = None,
    device: torch.device | None = None,
) -> Dict[int, List[float]]:
    n_agents = int(env_info["n_agents"])
    n_actions = int(env_info["n_actions"])
    out: Dict[int, List[float]] = {}
    cf_env = make_standard_marl_env(args.env, max_steps=args.max_steps)
    try:
        for step_idx in query_indexes:
            if step_idx >= len(transitions):
                continue
            effective_horizon = counterfactual_horizon_for_step(
                transitions=transitions,
                step_idx=step_idx,
                base_horizon=int(args.cf_horizon),
                event_horizon_max=int(getattr(args, "cf_event_horizon_max", 0)),
                reward_threshold=float(args.cf_reward_threshold),
                post_event_steps=int(getattr(args, "cf_event_horizon_post_steps", 0)),
            )
            if effective_horizon > int(args.cf_horizon):
                stats["event_horizon_extensions"] = stats.get("event_horizon_extensions", 0) + 1
                stats["event_horizon_sum"] = stats.get("event_horizon_sum", 0) + effective_horizon
                stats["event_horizon_max"] = max(
                    stats.get("event_horizon_max", 0), effective_horizon
                )
            policy_continuation = (
                getattr(args, "cf_continuation", "logged") == "actor_greedy"
                and actor is not None
                and device is not None
            )
            if policy_continuation:
                real_return, simulated_steps = replay_policy_counterfactual_return(
                    env=cf_env,
                    episode_seed=episode_seed,
                    transitions=transitions,
                    step_idx=step_idx,
                    agent_idx=None,
                    alt_action=None,
                    horizon=effective_horizon,
                    actor=actor,
                    device=device,
                    args=args,
                )
                stats["replay_calls"] += 1
                stats["simulated_steps"] += simulated_steps
            else:
                real_return = logged_horizon_return(transitions, step_idx, effective_horizon)
            row = []
            alternative_returns_by_agent: List[Dict[int, float]] = []
            for agent_idx in range(n_agents):
                original = int(transitions[step_idx]["actions"][agent_idx])  # type: ignore[index]
                cf_returns: Dict[int, float] = {}
                for alt_action in range(n_actions):
                    if alt_action == original:
                        continue
                    if policy_continuation:
                        cf_return, simulated_steps = replay_policy_counterfactual_return(
                            env=cf_env,
                            episode_seed=episode_seed,
                            transitions=transitions,
                            step_idx=step_idx,
                            agent_idx=agent_idx,
                            alt_action=alt_action,
                            horizon=effective_horizon,
                            actor=actor,
                            device=device,
                            args=args,
                        )
                    else:
                        cf_return, simulated_steps = replay_counterfactual_return(
                            env=cf_env,
                            episode_seed=episode_seed,
                            transitions=transitions,
                            step_idx=step_idx,
                            agent_idx=agent_idx,
                            alt_action=alt_action,
                            horizon=effective_horizon,
                        )
                    cf_returns[alt_action] = cf_return
                    stats["replay_calls"] += 1
                    stats["simulated_steps"] += simulated_steps
                row.append(
                    counterfactual_action_credit(
                        real_return=real_return,
                        original_action=original,
                        alternative_returns=cf_returns,
                        action_probabilities=transitions[step_idx].get("action_probs"),
                        agent_idx=agent_idx,
                        baseline=str(args.cf_action_baseline),
                    )
                )
                alternative_returns_by_agent.append(cf_returns)
                stats["credit_calls"] += 1
            pair_allocations = [0.0 for _ in range(n_agents)]
            for left, right in selected_pairwise_credit_pairs(
                transition=transitions[step_idx],
                fallback_credit=row,
                n_agents=n_agents,
                top_k=int(args.cf_pairwise_top_k),
            ):
                joint_returns = []
                for left_action in alternative_returns_by_agent[left]:
                    for right_action in alternative_returns_by_agent[right]:
                        overrides = {left: int(left_action), right: int(right_action)}
                        if policy_continuation:
                            joint_return, simulated_steps = replay_policy_counterfactual_return(
                                env=cf_env,
                                episode_seed=episode_seed,
                                transitions=transitions,
                                step_idx=step_idx,
                                agent_idx=None,
                                alt_action=None,
                                horizon=effective_horizon,
                                actor=actor,
                                device=device,
                                args=args,
                                action_overrides=overrides,
                            )
                        else:
                            joint_return, simulated_steps = replay_counterfactual_return(
                                env=cf_env,
                                episode_seed=episode_seed,
                                transitions=transitions,
                                step_idx=step_idx,
                                agent_idx=None,
                                alt_action=None,
                                horizon=effective_horizon,
                                action_overrides=overrides,
                            )
                        joint_returns.append(float(joint_return))
                        stats["replay_calls"] += 1
                        stats["simulated_steps"] += simulated_steps
                        stats["pair_replay_calls"] = stats.get("pair_replay_calls", 0) + 1
                        stats["pair_simulated_steps"] = (
                            stats.get("pair_simulated_steps", 0) + simulated_steps
                        )
                interaction = counterfactual_pair_interaction(
                    real_return=real_return,
                    left_returns=alternative_returns_by_agent[left],
                    right_returns=alternative_returns_by_agent[right],
                    joint_returns=joint_returns,
                )
                stats["pair_credit_calls"] = stats.get("pair_credit_calls", 0) + 1
                if abs(interaction) >= float(args.cf_pairwise_min_abs):
                    stats["pair_informative"] = stats.get("pair_informative", 0) + 1
                    pair_allocations[left] += 0.5 * interaction
                    pair_allocations[right] += 0.5 * interaction
            out[step_idx] = combine_individual_pairwise_credit(
                individual_credit=row,
                pair_allocations=pair_allocations,
                pairwise_scale=float(args.cf_pairwise_scale),
            )
    finally:
        cf_env.close()
    return out


def counterfactual_action_credit(
    real_return: float,
    original_action: int,
    alternative_returns: Mapping[int, float],
    action_probabilities: object,
    agent_idx: int,
    baseline: str,
) -> float:
    if not alternative_returns:
        return 0.0
    if baseline != "policy":
        return float(real_return) - mean(list(alternative_returns.values()))
    try:
        probabilities = np.asarray(action_probabilities, dtype=np.float64)[agent_idx]
    except (IndexError, TypeError, ValueError):
        return float(real_return) - mean(list(alternative_returns.values()))
    if probabilities.ndim != 1 or original_action >= probabilities.size:
        return float(real_return) - mean(list(alternative_returns.values()))
    expected_return = float(probabilities[original_action]) * float(real_return)
    probability_mass = float(probabilities[original_action])
    for action, action_return in alternative_returns.items():
        if 0 <= int(action) < probabilities.size:
            probability = max(0.0, float(probabilities[int(action)]))
            expected_return += probability * float(action_return)
            probability_mass += probability
    if probability_mass <= 1e-12:
        return float(real_return) - mean(list(alternative_returns.values()))
    return float(real_return) - expected_return / probability_mass


def selected_pairwise_credit_pairs(
    transition: Mapping[str, object],
    fallback_credit: Sequence[float],
    n_agents: int,
    top_k: int,
) -> List[Tuple[int, int]]:
    if top_k <= 0 or n_agents < 2:
        return []
    try:
        proposed = [float(value) for value in transition.get("raw_credit", [])]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        proposed = []
    if len(proposed) != n_agents or max((abs(value) for value in proposed), default=0.0) <= 1e-12:
        proposed = [float(value) for value in fallback_credit]
    if len(proposed) != n_agents:
        proposed = [0.0 for _ in range(n_agents)]
    pairs = list(itertools.combinations(range(n_agents), 2))
    pairs.sort(
        key=lambda pair: (
            -(abs(proposed[pair[0]]) + abs(proposed[pair[1]])),
            pair[0],
            pair[1],
        )
    )
    return pairs[: min(int(top_k), len(pairs))]


def counterfactual_pair_interaction(
    real_return: float,
    left_returns: Mapping[int, float],
    right_returns: Mapping[int, float],
    joint_returns: Sequence[float],
) -> float:
    if not left_returns or not right_returns or not joint_returns:
        return 0.0
    return (
        float(real_return)
        - mean(list(left_returns.values()))
        - mean(list(right_returns.values()))
        + mean(list(joint_returns))
    )


def combine_individual_pairwise_credit(
    individual_credit: Sequence[float],
    pair_allocations: Sequence[float],
    pairwise_scale: float,
) -> List[float]:
    individual = centered_scaled(individual_credit)
    if (
        pairwise_scale <= 0.0
        or not pair_allocations
        or max((abs(float(value)) for value in pair_allocations), default=0.0) <= 1e-12
    ):
        return individual
    pairwise = centered_scaled(pair_allocations)
    return centered_scaled(
        [
            float(base) + float(pairwise_scale) * float(pair_value)
            for base, pair_value in zip(individual, pairwise)
        ]
    )


def replay_counterfactual_return(
    env: StandardMARLEnv,
    episode_seed: int,
    transitions: Sequence[Dict[str, object]],
    step_idx: int,
    agent_idx: int | None,
    alt_action: int | None,
    horizon: int,
    action_overrides: Mapping[int, int] | None = None,
) -> Tuple[float, int]:
    env.reset(seed=episode_seed)
    total = 0.0
    simulated = 0
    end = min(len(transitions), step_idx + horizon)
    for idx in range(end):
        actions = [int(value) for value in transitions[idx]["actions"]]  # type: ignore[index]
        if idx == step_idx:
            if action_overrides:
                for override_agent, override_action in action_overrides.items():
                    actions[int(override_agent)] = int(override_action)
            elif agent_idx is not None and alt_action is not None:
                actions[agent_idx] = int(alt_action)
        step = env.step(actions)
        simulated += 1
        if idx >= step_idx:
            total += step.team_reward
        if step.done:
            break
    return total, simulated


def replay_policy_counterfactual_return(
    env: StandardMARLEnv,
    episode_seed: int,
    transitions: Sequence[Dict[str, object]],
    step_idx: int,
    agent_idx: int | None,
    alt_action: int | None,
    horizon: int,
    actor: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    action_overrides: Mapping[int, int] | None = None,
) -> Tuple[float, int]:
    observations = env.reset(seed=episode_seed)
    histories = init_obs_histories(observations, obs_frame_stack(args))
    simulated = 0
    for idx in range(step_idx):
        actions = [int(value) for value in transitions[idx]["actions"]]  # type: ignore[index]
        step = env.step(actions)
        simulated += 1
        observations = step.observations
        update_obs_histories(histories, observations, obs_frame_stack(args))
        if step.done:
            return 0.0, simulated

    actions = [int(value) for value in transitions[step_idx]["actions"]]  # type: ignore[index]
    if action_overrides:
        for override_agent, override_action in action_overrides.items():
            actions[int(override_agent)] = int(override_action)
    elif agent_idx is not None and alt_action is not None:
        actions[agent_idx] = int(alt_action)
    step = env.step(actions)
    simulated += 1
    total = float(step.team_reward)
    observations = step.observations
    update_obs_histories(histories, observations, obs_frame_stack(args))
    done = step.done

    for _offset in range(1, max(1, int(horizon))):
        if done:
            break
        actor_observations = stacked_observations_from_histories(histories)
        actor_inputs = actor_inputs_from_obs(
            actor_observations,
            env.n_agents,
            args.agent_id,
            args=args,
            env_family=env.family,
            task_hint_weight=0.0,
            task_hint_dropout=0.0,
            hint_observations=observations,
        )
        obs_tensor = torch.as_tensor(np.stack(actor_inputs), dtype=torch.float32, device=device)
        with torch.no_grad():
            continuation_actions = torch.argmax(actor(obs_tensor), dim=-1).cpu().numpy().tolist()
        step = env.step(continuation_actions)
        simulated += 1
        total += float(step.team_reward)
        observations = step.observations
        update_obs_histories(histories, observations, obs_frame_stack(args))
        done = step.done
    return total, simulated


def choose_credits(
    scheme: str,
    raw_credits: Sequence[Sequence[float]],
    features: Sequence[Sequence[Sequence[float]]],
    cf_credits_by_t: Mapping[int, Sequence[float]],
    credit_model: CreditModel | None,
    args: argparse.Namespace,
    global_calibration_reliable: bool = True,
    global_calibration_state: str | None = None,
    global_calibration_scale: float = 1.0,
    uncertified_credit_scale: float | None = None,
) -> List[List[float]]:
    predicted_rows = None
    if credit_model is not None:
        if hasattr(credit_model, "predict_rows"):
            predicted_rows = credit_model.predict_rows(features)
        else:
            predicted_rows = [
                predicted_credit_row(feature_row, credit_model, args) for feature_row in features
            ]
    out = []
    for t, row in enumerate(raw_credits):
        raw_row = centered_scaled(row)
        verifier_row = None
        if t in cf_credits_by_t:
            verifier_row = list(cf_credits_by_t[t])
        elif predicted_rows is not None:
            verifier_row = predicted_rows[t]
        if not global_calibration_reliable:
            if (
                global_calibration_state == "rejected"
                and args.rejected_credit_fallback == "verified_surrogate"
                and (verifier_row is not None)
            ):
                out.append(centered_scaled(verifier_row))
                continue
            if t in cf_credits_by_t and args.verified_credit_always_on:
                verified_row = centered_scaled(cf_credits_by_t[t])
                if args.verified_credit_mode == "raw_guard":
                    out.append(raw_guard_credit(raw_row, verified_row, args))
                elif args.verified_credit_mode in {"blend", "adaptive_blend"}:
                    if args.verified_credit_mode == "adaptive_blend":
                        (agreement, _scale, _accepted) = trajectory_scv_metrics(
                            [raw_row], {0: verified_row}, args
                        )
                        blend = 1.0 - float(np.clip(agreement, 0.0, 1.0))
                    else:
                        blend = float(np.clip(args.verified_credit_blend, 0.0, 1.0))
                    out.append(
                        centered_clipped(
                            [
                                (1.0 - blend) * raw_value + blend * verified_value
                                for (raw_value, verified_value) in zip(raw_row, verified_row)
                            ]
                        )
                    )
                else:
                    out.append(verified_row)
                continue
            probation_scale = (
                float(args.uncertified_credit_scale)
                if uncertified_credit_scale is None
                else float(uncertified_credit_scale)
            )
            candidate_row = centered_scaled(cf_credits_by_t[t]) if t in cf_credits_by_t else raw_row
            out.append([probation_scale * value for value in candidate_row])
        elif verifier_row is None:
            out.append(raw_row)
        elif t in cf_credits_by_t and args.verified_credit_mode == "exact":
            out.append(centered_scaled(verifier_row))
        elif t in cf_credits_by_t and args.verified_credit_mode in {"blend", "adaptive_blend"}:
            if args.verified_credit_mode == "adaptive_blend":
                (agreement, _scale, _accepted) = trajectory_scv_metrics(
                    [raw_row], {0: verifier_row}, args
                )
                blend = 1.0 - float(np.clip(agreement, 0.0, 1.0))
            else:
                blend = float(np.clip(args.verified_credit_blend, 0.0, 1.0))
            out.append(
                centered_clipped(
                    [
                        (1.0 - blend) * raw_value + blend * verified_value
                        for (raw_value, verified_value) in zip(raw_row, verifier_row)
                    ]
                )
            )
        else:
            out.append(
                raw_guard_credit(
                    raw_row,
                    verifier_row,
                    args,
                    allow_conflict_repair=not args.raw_guard_repair_exact_only
                    or t in cf_credits_by_t,
                )
            )
    return out


def predicted_credit_row(
    feature_row: Sequence[Sequence[float]],
    credit_model: object,
    args: argparse.Namespace,
) -> List[float]:
    predictions = [float(credit_model.predict(agent_features)) for agent_features in feature_row]  # type: ignore[attr-defined]
    uncertainty_threshold = float(args.credit_ensemble_uncertainty_threshold)
    if uncertainty_threshold >= 0.0 and hasattr(credit_model, "predict_std"):
        uncertainties = [
            float(credit_model.predict_std(agent_features))  # type: ignore[attr-defined]
            for agent_features in feature_row
        ]
        if max(uncertainties, default=0.0) > uncertainty_threshold:
            return [0.0 for _ in predictions]
    return centered_scaled(predictions)


def credit_source_rows(
    scheme: str,
    n_steps: int,
    cf_credits_by_t: Mapping[int, Sequence[float]],
    credit_model: CreditModel | None,
    args: argparse.Namespace,
) -> List[str]:
    out = []
    for t in range(n_steps):
        if t in cf_credits_by_t:
            out.append("verified")
        elif args.credit_surrogate and credit_model is not None:
            out.append("surrogate")
        else:
            out.append("none")
    return out


def raw_guard_credit(
    raw_row: Sequence[float],
    verifier_row: Sequence[float],
    args: argparse.Namespace,
    allow_conflict_repair: bool = True,
) -> List[float]:
    guarded = []
    for raw_value, verifier_value in zip(raw_row, verifier_row):
        raw_value = float(raw_value)
        verifier_value = float(verifier_value)
        if abs(verifier_value) < args.raw_guard_threshold:
            guarded.append(raw_value)
        elif raw_value * verifier_value < 0.0:
            guarded.append(
                args.raw_guard_conflict_weight * raw_value
                + (
                    args.raw_guard_conflict_verifier_scale * verifier_value
                    if allow_conflict_repair
                    else 0.0
                )
            )
        else:
            guarded.append(raw_value + args.raw_guard_agree_scale * verifier_value)
    return centered_clipped(guarded)


def novelty_bonus_rows(
    transitions: Sequence[Dict[str, object]],
    novelty_counts: Dict[str, int],
    args: argparse.Namespace,
) -> List[List[float]]:
    if args.novelty_bonus_coef <= 0.0:
        return [[0.0 for _ in transition["actions"]] for transition in transitions]  # type: ignore[index]
    rows = []
    bins = max(float(args.novelty_hash_bins), 1.0)
    for transition in transitions:
        actions = [int(value) for value in transition["actions"]]  # type: ignore[index]
        obs_list = [np.asarray(obs, dtype=np.float32).reshape(-1) for obs in transition["obs"]]  # type: ignore[index]
        row = []
        for agent_idx, (obs, action) in enumerate(zip(obs_list, actions)):
            key = novelty_key(obs, action, agent_idx if args.novelty_agent_specific else -1, bins)
            count = novelty_counts.get(key, 0) + 1
            novelty_counts[key] = count
            row.append(float(args.novelty_bonus_coef) / math.sqrt(float(count)))
        rows.append(row)
    return rows


def novelty_key(obs: np.ndarray, action: int, agent_idx: int, bins: float) -> str:
    quantized = np.round(np.asarray(obs, dtype=np.float32).reshape(-1) * bins).astype(np.int32)
    prefix = f"{agent_idx}:{int(action)}"
    return prefix + ":" + ",".join(str(int(value)) for value in quantized.tolist())


def potential_shaping_rows(
    transitions: Sequence[Dict[str, object]], args: argparse.Namespace, weight: float
) -> List[float]:
    return [0.0 for _ in transitions]


def scheduled_novelty_weight(args: argparse.Namespace, env_steps_seen: int) -> float:
    if args.novelty_bonus_coef <= 0.0:
        return 0.0
    warmup = max(int(args.novelty_warmup_env_steps), 0)
    decay_steps = max(int(args.novelty_decay_env_steps), 1)
    if env_steps_seen <= warmup:
        return 1.0
    progress = float(np.clip((env_steps_seen - warmup) / decay_steps, 0.0, 1.0))
    return float(1.0 - progress)


def scheduled_potential_shaping_weight(args: argparse.Namespace, env_steps_seen: int) -> float:
    return 0.0


def scheduled_proposal_action_weight(args: argparse.Namespace, env_steps_seen: int) -> float:
    start_coef = float(args.proposal_action_coef)
    return 0.0


def scheduled_behavior_proposal_prob(args: argparse.Namespace, env_steps_seen: int) -> float:
    return 0.0


def scheduled_credit_decay_weight(args: argparse.Namespace, env_steps_seen: int) -> float:
    start = int(args.credit_decay_start_env_steps)
    if start < 0:
        return 1.0
    decay_steps = max(int(args.credit_decay_env_steps), 1)
    if env_steps_seen <= start:
        return 1.0
    progress = float(np.clip((env_steps_seen - start) / decay_steps, 0.0, 1.0))
    return float(1.0 - progress)


def credit_activation_scale(args: argparse.Namespace, activity_ema: float) -> float:
    min_rate = float(args.credit_activation_min_rate)
    full_rate = float(args.credit_activation_full_rate)
    if full_rate <= min_rate:
        return 1.0
    raw_scale = float(np.clip((float(activity_ema) - min_rate) / (full_rate - min_rate), 0.0, 1.0))
    min_scale = float(np.clip(args.credit_activation_min_scale, 0.0, 1.0))
    return float(min_scale + (1.0 - min_scale) * raw_scale)


def reward_density_gate_decision(
    nonzero_rewards: int,
    total_rewards: int,
    threshold: float,
) -> Tuple[bool, float]:
    """Trust Team-gradient conflict checks only when warm-up reward is informative."""
    density = float(nonzero_rewards) / max(int(total_rewards), 1)
    return density >= float(threshold), density


def causal_confidence_gate_decision(
    stats: Mapping[str, object],
    min_queries: int,
    lower_bound_threshold: float,
    confidence_z: float,
) -> Tuple[bool, float]:
    """Gate credit while sparse counterfactual evidence is weak."""
    queries = int(stats.get("queries", 0))
    lower_bound = wilson_lower_bound(
        int(stats.get("accepted", 0)),
        queries,
        confidence_z,
    )
    insufficient = queries < max(1, int(min_queries))
    return insufficient or lower_bound < float(lower_bound_threshold), lower_bound


def sparse_trust_gradient_gate_decision(
    stats: Mapping[str, object],
    args: argparse.Namespace,
    reward_density: float | None,
) -> Tuple[bool, str]:
    trust_state = trajectory_scv_trust_state(
        stats,
        args,
        reward_density=reward_density,
    )
    sparse_reward = trajectory_scv_sparse_reward_regime(args, reward_density)
    return sparse_reward and trust_state != "certified", trust_state


def event_curriculum_reward_phase_active(
    args: argparse.Namespace, env_steps_seen: int, credit_certified: bool
) -> bool:
    return False


def assign_returns(
    samples: List[Dict[str, object]],
    transitions: Sequence[Dict[str, object]],
    credits: Sequence[Sequence[float]],
    args: argparse.Namespace,
    actor_credit_weight: float = 1.0,
    novelty_bonuses: Sequence[Sequence[float]] | None = None,
    novelty_weight: float = 0.0,
    credit_sources: Sequence[str] | None = None,
    verified_credit_weight: float = 1.0,
    surrogate_credit_weight: float = 0.0,
    event_curriculum_reward_phase: bool = False,
    proposal_action_weight: float = 0.0,
    advantage_credit_weight: float | None = None,
    potential_shaping_weight: float = 0.0,
) -> None:
    if advantage_credit_weight is None:
        advantage_credit_weight = actor_credit_weight
    n_agents = len(credits[0]) if credits else 0
    rewards_by_agent: List[List[float]] = [[] for _ in range(n_agents)]
    return_corrections_by_agent: List[List[float]] = [[] for _ in range(n_agents)]
    potential_shaping = potential_shaping_rows(transitions, args, potential_shaping_weight)
    team_returns = discounted_returns(
        [float(transition["team_reward"]) for transition in transitions], gamma=args.gamma
    )
    for t, transition in enumerate(transitions):
        base = float(transition["team_reward"]) + float(potential_shaping[t])
        step_return_corrections = []
        credit_source = (
            credit_sources[t] if credit_sources is not None and t < len(credit_sources) else "none"
        )
        safe_reward_shaping = None
        for agent_idx in range(n_agents):
            credit = float(credits[t][agent_idx])
            novelty = (
                float(novelty_bonuses[t][agent_idx])
                if novelty_bonuses is not None and t < len(novelty_bonuses)
                else 0.0
            )
            shaping = 0.0
            rewards_by_agent[agent_idx].append(base + shaping + novelty_weight * novelty)
            correction = 0.0
            step_return_corrections.append(correction)
        for agent_idx, correction in enumerate(step_return_corrections):
            return_corrections_by_agent[agent_idx].append(float(correction))
    returns_by_agent = [
        discounted_returns(rewards, gamma=args.gamma) for rewards in rewards_by_agent
    ]
    for sample in samples:
        agent_idx = int(sample["agent_idx"])
        t = int(sample["t"])
        sample["return"] = returns_by_agent[agent_idx][t]
        sample["team_return"] = team_returns[t]
        sample["credit_target"] = float(credits[t][agent_idx])
        sample["credit_weight"] = float(advantage_credit_weight)
        sample["novelty_bonus"] = (
            float(novelty_bonuses[t][agent_idx])
            if novelty_bonuses is not None and t < len(novelty_bonuses)
            else 0.0
        )
        sample["novelty_weight"] = float(novelty_weight)
        sample["team_reward"] = float(transitions[t]["team_reward"])
        sample["potential_shaping"] = float(potential_shaping[t])
        sample["event_curriculum_reward_phase"] = 1.0 if event_curriculum_reward_phase else 0.0
        sample["proposal_action_weight"] = float(proposal_action_weight)
        sample["credit"] = advantage_credit_weight * float(credits[t][agent_idx])


def add_safe_credit_residual(
    base_advantages: torch.Tensor,
    credit_correction: torch.Tensor,
    credit_weights: torch.Tensor,
    group_keys: Sequence[Tuple[int, int]],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Add bounded, zero-sum credit without changing the team-advantage budget."""
    out = base_advantages.clone()
    groups: Dict[Tuple[int, int], List[int]] = {}
    for idx, key in enumerate(group_keys):
        groups.setdefault(key, []).append(idx)

    active_groups = 0
    drift_values = []
    residual_rms_values = []
    blend_values = []
    temperature = float(args.advantage_redistribute_temperature)
    for indexes in groups.values():
        if len(indexes) <= 1:
            continue
        idx = torch.as_tensor(indexes, dtype=torch.long, device=base_advantages.device)
        group_weight = torch.clamp(
            torch.mean(credit_weights[idx]),
            min=0.0,
            max=float(args.advantage_redistribute_max_blend),
        )
        if float(group_weight.detach().cpu().item()) <= 1e-8:
            continue
        centered_credit = credit_correction[idx] - torch.mean(credit_correction[idx])
        bounded_credit = torch.tanh(temperature * centered_credit)
        bounded_credit = bounded_credit - torch.mean(bounded_credit)
        credit_rms = torch.sqrt(torch.mean(bounded_credit**2))
        if float(credit_rms.detach().cpu().item()) <= 1e-8:
            continue
        direction = bounded_credit / (credit_rms + 1e-8)
        residual = group_weight * direction
        before_sum = torch.sum(base_advantages[idx])
        out[idx] = base_advantages[idx] + residual
        after_sum = torch.sum(out[idx])
        active_groups += 1
        drift_values.append(float(torch.abs(after_sum - before_sum).detach().cpu().item()))
        residual_rms_values.append(float(torch.sqrt(torch.mean(residual**2)).detach().cpu().item()))
        blend_values.append(float(group_weight.detach().cpu().item()))

    return out, {
        "advantage_redistribution_groups": float(active_groups),
        "advantage_redistribution_positive_groups": 0.0,
        "advantage_redistribution_negative_groups": 0.0,
        "advantage_redistribution_zero_groups": 0.0,
        "advantage_redistribution_drift": mean(drift_values),
        "advantage_redistribution_blend": mean(blend_values),
        "advantage_safe_residual_rms": mean(residual_rms_values),
    }


def combine_credit_safe_gradients(
    base_gradients: Sequence[torch.Tensor | None],
    residual_gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    conflict_mode: str,
    max_residual_norm_ratio: float,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """Combine team and credit gradients without opposing the team gradient."""
    base = [
        gradient.detach() if gradient is not None else torch.zeros_like(parameter)
        for gradient, parameter in zip(base_gradients, parameters)
    ]
    residual = [
        gradient.detach() if gradient is not None else torch.zeros_like(parameter)
        for gradient, parameter in zip(residual_gradients, parameters)
    ]
    if not base:
        return [], {
            "credit_gradient_conflict": 0.0,
            "credit_gradient_cosine": 0.0,
            "credit_gradient_residual_scale": 0.0,
            "credit_gradient_post_dot": 0.0,
        }

    base_sq = sum(torch.sum(gradient * gradient) for gradient in base)
    residual_sq = sum(torch.sum(gradient * gradient) for gradient in residual)
    dot = sum(
        torch.sum(base_gradient * residual_gradient)
        for base_gradient, residual_gradient in zip(base, residual)
    )
    base_norm = torch.sqrt(torch.clamp(base_sq, min=0.0))
    residual_norm = torch.sqrt(torch.clamp(residual_sq, min=0.0))
    denominator = base_norm * residual_norm
    cosine = (
        dot / (denominator + 1e-12)
        if float(denominator.detach().cpu().item()) > 1e-12
        else torch.zeros((), dtype=dot.dtype, device=dot.device)
    )
    conflict = (
        float(base_norm.detach().cpu().item()) > 1e-12
        and float(residual_norm.detach().cpu().item()) > 1e-12
        and float(dot.detach().cpu().item()) < 0.0
    )

    safe_residual = list(residual)
    if float(base_norm.detach().cpu().item()) <= 1e-12:
        safe_residual = [torch.zeros_like(gradient) for gradient in residual]
    elif conflict and conflict_mode == "gate":
        safe_residual = [torch.zeros_like(gradient) for gradient in residual]
    elif conflict and conflict_mode == "project":
        coefficient = dot / (base_sq + 1e-12)
        safe_residual = [
            residual_gradient - coefficient * base_gradient
            for base_gradient, residual_gradient in zip(base, residual)
        ]

    safe_residual_sq = sum(torch.sum(gradient * gradient) for gradient in safe_residual)
    safe_residual_norm = torch.sqrt(torch.clamp(safe_residual_sq, min=0.0))
    residual_scale = torch.ones((), dtype=dot.dtype, device=dot.device)
    if max_residual_norm_ratio > 0.0:
        residual_scale = torch.clamp(
            float(max_residual_norm_ratio) * base_norm / (safe_residual_norm + 1e-12),
            min=0.0,
            max=1.0,
        )
    safe_residual = [residual_scale * gradient for gradient in safe_residual]
    post_dot = sum(
        torch.sum(base_gradient * residual_gradient)
        for base_gradient, residual_gradient in zip(base, safe_residual)
    )
    combined = [
        base_gradient + residual_gradient
        for base_gradient, residual_gradient in zip(base, safe_residual)
    ]
    return combined, {
        "credit_gradient_conflict": 1.0 if conflict else 0.0,
        "credit_gradient_cosine": float(cosine.detach().cpu().item()),
        "credit_gradient_residual_scale": float(residual_scale.detach().cpu().item()),
        "credit_gradient_post_dot": float(post_dot.detach().cpu().item()),
    }


def combine_grouped_credit_safe_gradients(
    base_gradients: Sequence[torch.Tensor | None],
    grouped_residual_gradients: Sequence[Sequence[torch.Tensor | None]],
    parameters: Sequence[torch.nn.Parameter],
    conflict_mode: str,
    max_residual_norm_ratio: float,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """Gate agent-level credit residuals before applying one global norm cap."""
    base = [
        gradient.detach() if gradient is not None else torch.zeros_like(parameter)
        for gradient, parameter in zip(base_gradients, parameters)
    ]
    groups = [
        [
            gradient.detach() if gradient is not None else torch.zeros_like(parameter)
            for gradient, parameter in zip(group, parameters)
        ]
        for group in grouped_residual_gradients
    ]
    if not base:
        return [], {
            "credit_gradient_conflict": 0.0,
            "credit_gradient_cosine": 0.0,
            "credit_gradient_residual_scale": 0.0,
            "credit_gradient_post_dot": 0.0,
        }

    base_sq = sum(torch.sum(gradient * gradient) for gradient in base)
    base_norm = torch.sqrt(torch.clamp(base_sq, min=0.0))
    raw_residual = [torch.zeros_like(parameter) for parameter in parameters]
    for group in groups:
        raw_residual = [total + value for total, value in zip(raw_residual, group)]
    raw_sq = sum(torch.sum(gradient * gradient) for gradient in raw_residual)
    raw_norm = torch.sqrt(torch.clamp(raw_sq, min=0.0))
    raw_dot = sum(
        torch.sum(base_gradient * residual_gradient)
        for base_gradient, residual_gradient in zip(base, raw_residual)
    )
    denominator = base_norm * raw_norm
    cosine = (
        raw_dot / (denominator + 1e-12)
        if float(denominator.detach().cpu().item()) > 1e-12
        else torch.zeros((), dtype=raw_dot.dtype, device=raw_dot.device)
    )

    safe_residual = [torch.zeros_like(parameter) for parameter in parameters]
    conflict_count = 0
    nonzero_groups = 0
    for group in groups:
        group_sq = sum(torch.sum(gradient * gradient) for gradient in group)
        group_norm = torch.sqrt(torch.clamp(group_sq, min=0.0))
        if float(group_norm.detach().cpu().item()) <= 1e-12:
            continue
        nonzero_groups += 1
        group_dot = sum(
            torch.sum(base_gradient * residual_gradient)
            for base_gradient, residual_gradient in zip(base, group)
        )
        conflict = (
            float(base_norm.detach().cpu().item()) > 1e-12
            and float(group_dot.detach().cpu().item()) < 0.0
        )
        conflict_count += int(conflict)
        safe_group = list(group)
        if float(base_norm.detach().cpu().item()) <= 1e-12:
            safe_group = [torch.zeros_like(gradient) for gradient in group]
        elif conflict and conflict_mode == "gate":
            safe_group = [torch.zeros_like(gradient) for gradient in group]
        elif conflict and conflict_mode == "project":
            coefficient = group_dot / (base_sq + 1e-12)
            safe_group = [
                residual_gradient - coefficient * base_gradient
                for base_gradient, residual_gradient in zip(base, group)
            ]
        safe_residual = [total + value for total, value in zip(safe_residual, safe_group)]

    safe_sq = sum(torch.sum(gradient * gradient) for gradient in safe_residual)
    safe_norm = torch.sqrt(torch.clamp(safe_sq, min=0.0))
    residual_scale = torch.ones((), dtype=raw_dot.dtype, device=raw_dot.device)
    if max_residual_norm_ratio > 0.0:
        residual_scale = torch.clamp(
            float(max_residual_norm_ratio) * base_norm / (safe_norm + 1e-12),
            min=0.0,
            max=1.0,
        )
    safe_residual = [residual_scale * gradient for gradient in safe_residual]
    post_dot = sum(
        torch.sum(base_gradient * residual_gradient)
        for base_gradient, residual_gradient in zip(base, safe_residual)
    )
    combined = [
        base_gradient + residual_gradient
        for base_gradient, residual_gradient in zip(base, safe_residual)
    ]
    return combined, {
        "credit_gradient_conflict": float(conflict_count) / max(nonzero_groups, 1),
        "credit_gradient_cosine": float(cosine.detach().cpu().item()),
        "credit_gradient_residual_scale": float(residual_scale.detach().cpu().item()),
        "credit_gradient_post_dot": float(post_dot.detach().cpu().item()),
    }


def generalized_advantage_estimates(
    rewards: torch.Tensor,
    values: torch.Tensor,
    group_keys: Sequence[Tuple[int, int]],
    agent_indices: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute terminal-episode GAE independently for each shared-policy agent."""
    advantages = torch.zeros_like(values)
    grouped_indexes: Dict[Tuple[int, int], List[int]] = {}
    for row_idx, ((episode_idx, _t), agent_idx) in enumerate(
        zip(group_keys, agent_indices.detach().cpu().tolist())
    ):
        grouped_indexes.setdefault((int(episode_idx), int(agent_idx)), []).append(row_idx)

    for indexes in grouped_indexes.values():
        indexes.sort(key=lambda idx: int(group_keys[idx][1]))
        running = torch.zeros((), dtype=values.dtype, device=values.device)
        next_value = torch.zeros((), dtype=values.dtype, device=values.device)
        for idx in reversed(indexes):
            delta = rewards[idx] + float(gamma) * next_value - values[idx]
            running = delta + float(gamma) * float(gae_lambda) * running
            advantages[idx] = running
            next_value = values[idx]
    return advantages, advantages + values


def update_ppo(
    actor: nn.Module,
    critic: Critic,
    credit_critic: CreditCritic,
    actor_optim: torch.optim.Optimizer,
    critic_optim: torch.optim.Optimizer,
    credit_optim: torch.optim.Optimizer,
    samples: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    device: torch.device,
    n_actions: int,
    entropy_coef: float,
    credit_gradient_safety_override: bool | None = None,
    credit_gradient_conflict_mode_override: str | None = None,
) -> Dict[str, float]:
    obs = torch.as_tensor(
        np.stack([sample["obs"] for sample in samples]), dtype=torch.float32, device=device
    )
    critic_obs = torch.as_tensor(
        np.stack([sample["critic_obs"] for sample in samples]), dtype=torch.float32, device=device
    )
    actions = torch.as_tensor(
        [int(sample["action"]) for sample in samples], dtype=torch.int64, device=device
    )
    agent_indices = torch.as_tensor(
        [int(sample["agent_idx"]) for sample in samples], dtype=torch.int64, device=device
    )
    old_logps = torch.as_tensor(
        [float(sample["old_logp"]) for sample in samples], dtype=torch.float32, device=device
    )
    returns = torch.as_tensor(
        [float(sample["return"]) for sample in samples], dtype=torch.float32, device=device
    )
    team_returns = torch.as_tensor(
        [float(sample.get("team_return", sample["return"])) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    credits = torch.as_tensor(
        [float(sample["credit"]) for sample in samples], dtype=torch.float32, device=device
    )
    credit_targets = torch.as_tensor(
        [float(sample.get("credit_target", sample["credit"])) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    credit_weights = torch.as_tensor(
        [float(sample.get("credit_weight", 1.0)) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    novelty_bonuses = torch.as_tensor(
        [float(sample.get("novelty_bonus", 0.0)) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    novelty_weights = torch.as_tensor(
        [float(sample.get("novelty_weight", 0.0)) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    team_rewards = torch.as_tensor(
        [float(sample.get("team_reward", 0.0)) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    event_curriculum_reward_phases = torch.as_tensor(
        [float(sample.get("event_curriculum_reward_phase", 0.0)) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    proposal_actions = torch.as_tensor(
        [int(sample.get("proposal_action", -1)) for sample in samples],
        dtype=torch.int64,
        device=device,
    )
    proposal_action_weights = torch.as_tensor(
        [float(sample.get("proposal_action_weight", 0.0)) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    potential_shapings = torch.as_tensor(
        [float(sample.get("potential_shaping", 0.0)) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    group_keys = [(int(sample.get("episode_idx", 0)), int(sample["t"])) for sample in samples]
    action_one_hot = F.one_hot(actions, num_classes=n_actions).to(dtype=torch.float32)
    credit_inputs = torch.cat([critic_obs, action_one_hot], dim=-1)
    with torch.no_grad():
        values = critic(critic_obs)
        if float(args.gae_lambda) >= 0.0:
            (policy_base_advantages, critic_training_targets) = generalized_advantage_estimates(
                team_rewards,
                values,
                group_keys,
                agent_indices,
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
            )
        else:
            policy_base_advantages = returns - values
            critic_training_targets = returns
        base_advantages = policy_base_advantages
        credit_pred_mse = torch.zeros((), dtype=torch.float32, device=device)
        credit_reliability = torch.zeros((), dtype=torch.float32, device=device)
        effective_credit_mix = torch.zeros((), dtype=torch.float32, device=device)
        redistribution_stats = {
            "advantage_redistribution_groups": 0.0,
            "advantage_redistribution_positive_groups": 0.0,
            "advantage_redistribution_negative_groups": 0.0,
            "advantage_redistribution_zero_groups": 0.0,
            "advantage_redistribution_drift": 0.0,
            "advantage_redistribution_blend": 0.0,
            "advantage_reward_density": 0.0,
            "advantage_blend_scale": 1.0,
            "advantage_credit_delta_ratio_before": 0.0,
            "advantage_credit_delta_ratio_after": 0.0,
            "advantage_credit_delta_clip_scale": 1.0,
            "advantage_safe_residual_rms": 0.0,
        }
        reward_density = torch.mean(
            (torch.abs(team_rewards) > float(args.advantage_redistribute_reward_threshold)).to(
                torch.float32
            )
        )
        if args.advantage_redistribute_density_gate:
            blend_scale = torch.clamp(
                1.0 - reward_density,
                min=float(args.advantage_redistribute_min_blend_scale),
                max=1.0,
            )
        else:
            blend_scale = torch.ones((), dtype=torch.float32, device=device)
        redistribution_stats["advantage_reward_density"] = float(
            reward_density.detach().cpu().item()
        )
        redistribution_stats["advantage_blend_scale"] = float(blend_scale.detach().cpu().item())
        (advantages, redistribution_core_stats) = add_safe_credit_residual(
            base_advantages=policy_base_advantages,
            credit_correction=credits,
            credit_weights=credit_weights,
            group_keys=group_keys,
            args=args,
        )
        redistribution_stats.update(redistribution_core_stats)
    normalized_base = base_advantages
    if normalized_base.numel() > 1 and torch.std(normalized_base) > 1e-08:
        normalized_base = (normalized_base - normalized_base.mean()) / (
            normalized_base.std() + 1e-08
        )
    normalized_advantages = advantages
    if normalized_advantages.numel() > 1 and torch.std(normalized_advantages) > 1e-08:
        normalized_advantages = (normalized_advantages - normalized_advantages.mean()) / (
            normalized_advantages.std() + 1e-08
        )
    max_delta_ratio = float(args.advantage_credit_max_l2_ratio)
    if max_delta_ratio > 0.0:
        delta = normalized_advantages - normalized_base
        base_norm = torch.linalg.vector_norm(normalized_base)
        delta_norm = torch.linalg.vector_norm(delta)
        ratio_before = delta_norm / (base_norm + 1e-08)
        clip_scale = torch.clamp(
            max_delta_ratio * base_norm / (delta_norm + 1e-08), min=0.0, max=1.0
        )
        if float(delta_norm.detach().cpu().item()) > 1e-08:
            advantages = normalized_base + clip_scale * delta
            ratio_after = torch.linalg.vector_norm(advantages - normalized_base) / (
                base_norm + 1e-08
            )
        else:
            advantages = normalized_base
            ratio_after = torch.zeros((), dtype=torch.float32, device=device)
        redistribution_stats["advantage_credit_delta_ratio_before"] = float(
            ratio_before.detach().cpu().item()
        )
        redistribution_stats["advantage_credit_delta_ratio_after"] = float(
            ratio_after.detach().cpu().item()
        )
        redistribution_stats["advantage_credit_delta_clip_scale"] = float(
            clip_scale.detach().cpu().item()
        )
    else:
        advantages = normalized_advantages
    n_samples = obs.shape[0]
    indices = torch.arange(n_samples, device=device)
    minibatch = min(args.minibatch_size, n_samples)
    actor_losses = []
    critic_losses = []
    credit_losses = []
    proposal_action_losses = []
    proposal_action_match_rates = []
    proposal_action_weight_means = []
    credit_gradient_conflicts = []
    credit_gradient_cosines = []
    credit_gradient_residual_scales = []
    credit_gradient_post_dots = []
    actor_parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    gradient_safety_enabled = (
        bool(args.credit_gradient_safety)
        if credit_gradient_safety_override is None
        else bool(credit_gradient_safety_override)
    )
    gradient_conflict_mode = (
        str(credit_gradient_conflict_mode_override)
        if credit_gradient_conflict_mode_override is not None
        else str(args.credit_gradient_conflict_mode)
    )
    for _ in range(args.ppo_epochs):
        perm = indices[torch.randperm(n_samples, device=device)]
        for start in range(0, n_samples, minibatch):
            idx = perm[start : start + minibatch]
            logits = actor(obs[idx])
            dist = Categorical(logits=logits)
            logps = dist.log_prob(actions[idx])
            ratio = torch.exp(logps - old_logps[idx])
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * advantages[idx]
            actor_loss_rows = -torch.min(unclipped, clipped) - entropy_coef * dist.entropy()
            actor_loss = actor_loss_rows.mean()
            base_unclipped = ratio * normalized_base[idx]
            base_clipped = (
                torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * normalized_base[idx]
            )
            base_actor_loss_rows = (
                -torch.min(base_unclipped, base_clipped) - entropy_coef * dist.entropy()
            )
            base_actor_loss = base_actor_loss_rows.mean()
            proposal_valid = (proposal_actions[idx] >= 0) & (proposal_action_weights[idx] > 0.0)
            if torch.any(proposal_valid):
                valid_logits = logits[proposal_valid]
                valid_targets = proposal_actions[idx][proposal_valid]
                valid_weights = proposal_action_weights[idx][proposal_valid]
                proposal_action_loss = torch.mean(
                    F.cross_entropy(valid_logits, valid_targets, reduction="none") * valid_weights
                )
                actor_loss = actor_loss + proposal_action_loss
                base_actor_loss = base_actor_loss + proposal_action_loss
                proposal_action_losses.append(float(proposal_action_loss.detach().cpu().item()))
                with torch.no_grad():
                    proposal_action_match_rates.append(
                        float(
                            (torch.argmax(valid_logits, dim=-1) == valid_targets)
                            .to(torch.float32)
                            .mean()
                            .cpu()
                            .item()
                        )
                    )
                    proposal_action_weight_means.append(
                        float(valid_weights.mean().detach().cpu().item())
                    )
            else:
                proposal_action_loss = torch.zeros((), dtype=torch.float32, device=device)
            values = critic(critic_obs[idx])
            critic_loss = torch.mean((critic_training_targets[idx] - values) ** 2)
            credit_loss = torch.zeros((), dtype=torch.float32, device=device)
            actor_losses.append(float(actor_loss.detach().cpu().item()))
            critic_losses.append(float(critic_loss.detach().cpu().item()))
            credit_losses.append(float(credit_loss.detach().cpu().item()))
            actor_optim.zero_grad()
            if gradient_safety_enabled:
                base_gradients = torch.autograd.grad(
                    base_actor_loss, actor_parameters, retain_graph=True, allow_unused=True
                )
                if gradient_conflict_mode in {"agent_gate", "agent_project"}:
                    residual_rows = actor_loss_rows - base_actor_loss_rows
                    grouped_residual_gradients = []
                    for agent_idx in torch.unique(agent_indices[idx]).tolist():
                        mask = agent_indices[idx] == int(agent_idx)
                        group_loss = torch.sum(residual_rows[mask]) / max(int(idx.numel()), 1)
                        grouped_residual_gradients.append(
                            torch.autograd.grad(
                                group_loss, actor_parameters, retain_graph=True, allow_unused=True
                            )
                        )
                    (combined_gradients, gradient_stats) = combine_grouped_credit_safe_gradients(
                        base_gradients=base_gradients,
                        grouped_residual_gradients=grouped_residual_gradients,
                        parameters=actor_parameters,
                        conflict_mode=gradient_conflict_mode.removeprefix("agent_"),
                        max_residual_norm_ratio=args.credit_gradient_max_norm_ratio,
                    )
                else:
                    credit_residual_loss = actor_loss - base_actor_loss
                    residual_gradients = torch.autograd.grad(
                        credit_residual_loss, actor_parameters, allow_unused=True
                    )
                    (combined_gradients, gradient_stats) = combine_credit_safe_gradients(
                        base_gradients=base_gradients,
                        residual_gradients=residual_gradients,
                        parameters=actor_parameters,
                        conflict_mode=gradient_conflict_mode,
                        max_residual_norm_ratio=args.credit_gradient_max_norm_ratio,
                    )
                for parameter, gradient in zip(actor_parameters, combined_gradients):
                    parameter.grad = gradient.clone()
                credit_gradient_conflicts.append(gradient_stats["credit_gradient_conflict"])
                credit_gradient_cosines.append(gradient_stats["credit_gradient_cosine"])
                credit_gradient_residual_scales.append(
                    gradient_stats["credit_gradient_residual_scale"]
                )
                credit_gradient_post_dots.append(gradient_stats["credit_gradient_post_dot"])
            else:
                actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optim.step()
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_optim.step()
    stats = {
        "actor_loss": round(mean(actor_losses), 8),
        "critic_loss": round(mean(critic_losses), 8),
        "credit_loss": round(mean(credit_losses), 8),
        "advantage_mean": round(float(advantages.mean().detach().cpu().item()), 8),
        "advantage_std": round(
            float(advantages.std().detach().cpu().item()) if advantages.numel() > 1 else 0.0, 8
        ),
        "credit_mean": round(float(credits.mean().detach().cpu().item()), 8),
        "credit_std": round(
            float(credits.std().detach().cpu().item()) if credits.numel() > 1 else 0.0, 8
        ),
        "credit_target_mean": round(float(credit_targets.mean().detach().cpu().item()), 8),
        "credit_target_std": round(
            (
                float(credit_targets.std().detach().cpu().item())
                if credit_targets.numel() > 1
                else 0.0
            ),
            8,
        ),
        "credit_weight_mean": round(float(credit_weights.mean().detach().cpu().item()), 8),
        "credit_pred_mse": round(float(credit_pred_mse.detach().cpu().item()), 8),
        "credit_reliability": round(float(credit_reliability.detach().cpu().item()), 8),
        "effective_credit_mix": round(float(effective_credit_mix.detach().cpu().item()), 8),
        "entropy_coef": round(float(entropy_coef), 8),
        "novelty_bonus_mean": round(float(novelty_bonuses.mean().detach().cpu().item()), 8),
        "novelty_weight_mean": round(float(novelty_weights.mean().detach().cpu().item()), 8),
        "potential_shaping_mean": round(float(potential_shapings.mean().detach().cpu().item()), 8),
        "potential_shaping_std": round(
            (
                float(potential_shapings.std().detach().cpu().item())
                if potential_shapings.numel() > 1
                else 0.0
            ),
            8,
        ),
        "proposal_action_loss": round(mean(proposal_action_losses), 8),
        "proposal_action_weight_mean": round(
            float(proposal_action_weights.mean().detach().cpu().item()), 8
        ),
        "proposal_action_active_weight_mean": round(mean(proposal_action_weight_means), 8),
        "proposal_action_match_rate": round(mean(proposal_action_match_rates), 8),
        "proposal_action_valid_rate": round(
            float(
                ((proposal_actions >= 0) & (proposal_action_weights > 0.0))
                .to(torch.float32)
                .mean()
                .detach()
                .cpu()
                .item()
            ),
            8,
        ),
        "behavior_proposal_used_rate": round(
            mean([float(sample.get("behavior_proposal_used", 0.0)) for sample in samples]), 8
        ),
        "event_curriculum_reward_phase_mean": round(
            float(event_curriculum_reward_phases.mean().detach().cpu().item()), 8
        ),
        "advantage_redistribution_groups": round(
            redistribution_stats["advantage_redistribution_groups"], 8
        ),
        "advantage_redistribution_positive_groups": round(
            redistribution_stats["advantage_redistribution_positive_groups"], 8
        ),
        "advantage_redistribution_negative_groups": round(
            redistribution_stats["advantage_redistribution_negative_groups"], 8
        ),
        "advantage_redistribution_zero_groups": round(
            redistribution_stats["advantage_redistribution_zero_groups"], 8
        ),
        "advantage_redistribution_drift": round(
            redistribution_stats["advantage_redistribution_drift"], 8
        ),
        "advantage_redistribution_blend": round(
            redistribution_stats["advantage_redistribution_blend"], 8
        ),
        "advantage_reward_density": round(redistribution_stats["advantage_reward_density"], 8),
        "advantage_blend_scale": round(redistribution_stats["advantage_blend_scale"], 8),
        "advantage_credit_delta_ratio_before": round(
            redistribution_stats["advantage_credit_delta_ratio_before"], 8
        ),
        "advantage_credit_delta_ratio_after": round(
            redistribution_stats["advantage_credit_delta_ratio_after"], 8
        ),
        "advantage_credit_delta_clip_scale": round(
            redistribution_stats["advantage_credit_delta_clip_scale"], 8
        ),
        "advantage_safe_residual_rms": round(
            redistribution_stats["advantage_safe_residual_rms"], 8
        ),
        "credit_gradient_safety_enabled": 1.0 if gradient_safety_enabled else 0.0,
        "credit_gradient_conflict_rate": round(mean(credit_gradient_conflicts), 8),
        "credit_gradient_cosine": round(mean(credit_gradient_cosines), 8),
        "credit_gradient_residual_scale": round(mean(credit_gradient_residual_scales), 8),
        "credit_gradient_post_dot": round(mean(credit_gradient_post_dots), 8),
    }
    return stats


def evaluate(
    actor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    env_info: Mapping[str, int | str],
) -> float:
    returns = []
    rng = np.random.default_rng(args.eval_seed)
    for eval_idx in range(args.eval_episodes):
        env = make_standard_marl_env(args.env, max_steps=args.max_steps)
        observations = env.reset(seed=args.eval_seed + eval_idx)
        histories = init_obs_histories(observations, obs_frame_stack(args))
        total = 0.0
        done = False
        while not done:
            actor_observations = stacked_observations_from_histories(histories)
            hint_actions = task_hint_actions_for_observations(
                observations=observations,
                env_family=str(env_info["family"]),
                n_agents=int(env_info["n_agents"]),
                args=args,
                env=env,
            )
            hint_targets = task_hint_targets_for_observations(
                observations=observations,
                env_family=str(env_info["family"]),
                n_agents=int(env_info["n_agents"]),
                args=args,
                env=env,
            )
            actor_inputs = actor_inputs_from_obs(
                actor_observations,
                int(env_info["n_agents"]),
                args.agent_id,
                args=args,
                env_family=str(env_info["family"]),
                task_hint_weight=float(args.eval_task_hint_weight),
                task_hint_dropout=0.0,
                hint_observations=observations,
                hint_actions=hint_actions,
                hint_targets=hint_targets,
            )
            obs_tensor = torch.as_tensor(np.stack(actor_inputs), dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = actor(obs_tensor)
                if args.eval_mode == "greedy":
                    actions = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                else:
                    eval_temperature = max(1e-6, float(args.eval_temperature))
                    eval_logits = logits / eval_temperature
                    if args.eval_top_k > 0 and args.eval_top_k < eval_logits.shape[-1]:
                        top_values, top_indexes = torch.topk(
                            eval_logits, k=int(args.eval_top_k), dim=-1
                        )
                        filtered = torch.full_like(eval_logits, -1e9)
                        eval_logits = filtered.scatter(dim=-1, index=top_indexes, src=top_values)
                    probs = torch.softmax(eval_logits, dim=-1).cpu().numpy()
                    actions = [
                        int(rng.choice(probs.shape[1], p=probs[agent_idx]))
                        for agent_idx in range(probs.shape[0])
                    ]
            step = env.step(actions)
            total += step.team_reward
            observations = step.observations
            update_obs_histories(histories, observations, obs_frame_stack(args))
            done = step.done
        env.close()
        returns.append(total)
    return float(np.mean(returns))


def evaluate_with_overrides(
    actor: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    env_info: Mapping[str, int | str],
    *,
    mode: str | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
) -> float:
    eval_args = argparse.Namespace(**vars(args))
    if mode is not None:
        eval_args.eval_mode = mode
    if temperature is not None:
        eval_args.eval_temperature = float(temperature)
    if top_k is not None:
        eval_args.eval_top_k = int(top_k)
    return evaluate(actor, eval_args, device, env_info)


def actor_inputs_from_obs(
    observations: Sequence[np.ndarray],
    n_agents: int,
    use_agent_id: bool,
    args: argparse.Namespace | None = None,
    env_family: str = "",
    task_hint_weight: float = 0.0,
    task_hint_dropout: float = 0.0,
    hint_observations: Sequence[np.ndarray] | None = None,
    hint_actions: Sequence[int] | None = None,
    hint_targets: Sequence[object] | None = None,
) -> List[np.ndarray]:
    hints = task_hints_for_observations(
        hint_observations if hint_observations is not None else observations,
        env_family,
        n_agents,
        args,
        task_hint_weight,
        task_hint_dropout,
        hint_actions=hint_actions,
        hint_targets=hint_targets,
    )
    out = []
    for agent_idx, obs in enumerate(observations):
        base = normalize_policy_observation(obs, args=args, env_family=env_family)
        if use_agent_id:
            base = np.concatenate([base, one_hot(agent_idx, n_agents)])
        if hints:
            base = np.concatenate([base, hints[agent_idx]])
        out.append(base.astype(np.float32))
    return out


def normalize_policy_observation(
    observation: np.ndarray,
    *,
    args: argparse.Namespace | None,
    env_family: str,
) -> np.ndarray:
    """Scale raw LBF tuple fields before they enter actor/critic MLPs."""
    values = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
    mode = str(getattr(args, "obs_normalization", "none") or "none") if args is not None else "none"
    if mode == "none" or str(env_family) != "lbf":
        return values
    if mode != "lbf_static":
        raise ValueError(f"Unsupported observation normalization mode: {mode}")

    env_name = str(getattr(args, "env", "") or "")
    match = re.search(r"lbf/(\d+)x\d+", env_name)
    if match is None:
        raise ValueError(f"Cannot infer LBF grid size from environment name: {env_name}")
    coordinate_scale = float(max(1, int(match.group(1)) - 1))

    frame_stack = obs_frame_stack(args)
    if values.size % frame_stack != 0:
        raise ValueError(
            f"Observation length {values.size} is not divisible by frame stack {frame_stack}"
        )
    frame_dim = values.size // frame_stack
    if frame_dim % 3 != 0:
        raise ValueError(f"LBF observation frame length must be divisible by 3, got {frame_dim}")

    for frame_idx in range(frame_stack):
        start = frame_idx * frame_dim
        stop = start + frame_dim
        frame = values[start:stop]
        # Preserve zero-filled history frames used before enough observations exist.
        if frame_stack > 1 and not np.any(frame):
            continue
        frame.reshape(-1, 3)[:, :2] /= coordinate_scale
        frame.reshape(-1, 3)[:, 2] /= coordinate_scale
    return values


def obs_frame_stack(args: argparse.Namespace | None) -> int:
    if args is None:
        return 1
    return max(1, int(getattr(args, "obs_frame_stack", 1) or 1))


def init_obs_histories(observations: Sequence[np.ndarray], stack: int) -> List[List[np.ndarray]]:
    stack = max(1, int(stack))
    histories: List[List[np.ndarray]] = []
    for obs in observations:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        histories.append([np.zeros_like(arr) for _ in range(stack - 1)] + [arr.copy()])
    return histories


def stacked_observations_from_histories(
    histories: Sequence[Sequence[np.ndarray]],
) -> List[np.ndarray]:
    return [
        np.concatenate([np.asarray(obs, dtype=np.float32).reshape(-1) for obs in history]).astype(
            np.float32
        )
        for history in histories
    ]


def update_obs_histories(
    histories: List[List[np.ndarray]], observations: Sequence[np.ndarray], stack: int
) -> None:
    stack = max(1, int(stack))
    for history, obs in zip(histories, observations):
        history.append(np.asarray(obs, dtype=np.float32).reshape(-1).copy())
        if len(history) > stack:
            del history[: len(history) - stack]


def central_input_from_obs(
    observations: Sequence[np.ndarray],
    n_agents: int,
    use_agent_id: bool,
    *,
    args: argparse.Namespace | None = None,
    env_family: str = "",
) -> np.ndarray:
    parts = [
        normalize_policy_observation(obs, args=args, env_family=env_family) for obs in observations
    ]
    central = np.concatenate(parts).astype(np.float32)
    return central


def critic_input_for_agent(
    central: np.ndarray,
    agent_idx: int,
    n_agents: int,
    use_agent_id: bool,
) -> np.ndarray:
    if not use_agent_id:
        return central.astype(np.float32)
    return np.concatenate([central, one_hot(agent_idx, n_agents)]).astype(np.float32)


def selected_query_indexes(n_steps: int, args: argparse.Namespace) -> List[int]:
    if args.cf_transitions <= 0 or args.cf_transitions >= n_steps:
        return list(range(n_steps))
    start = max(0, n_steps - args.cf_transitions)
    return list(range(start, n_steps))


def selected_query_indexes_from_transitions(
    transitions: Sequence[Dict[str, object]],
    args: argparse.Namespace,
    episode_idx: int | None = None,
    features: Sequence[Sequence[Sequence[float]]] | None = None,
    credit_model: object | None = None,
) -> List[int]:
    n_steps = len(transitions)
    if args.cf_selection == "tail":
        return selected_query_indexes(n_steps, args)
    if args.cf_transitions <= 0 or args.cf_transitions >= n_steps:
        return list(range(n_steps))
    if args.cf_selection == "uniform":
        if args.cf_transitions == 1:
            return [n_steps - 1]
        values = np.linspace(0, n_steps - 1, num=args.cf_transitions)
        return sorted({int(round(value)) for value in values})
    if args.cf_selection == "random":
        selection_seed = int(args.seed) * 1_000_003 + int(episode_idx or 0)
        rng = np.random.default_rng(selection_seed)
        selected = rng.choice(n_steps, size=args.cf_transitions, replace=False)
        return sorted(int(value) for value in selected.tolist())
    if args.cf_selection == "credit":
        ranked = ranked_credit_indexes(transitions)
        return sorted(idx for _score, idx in ranked[: args.cf_transitions])
    if args.cf_selection == "hybrid_disagreement":
        if features is None or credit_model is None:
            ranked = ranked_credit_indexes(transitions)
            return sorted(idx for _score, idx in ranked[: args.cf_transitions])
        predicted_rows = [
            predicted_credit_row(feature_row, credit_model, args) for feature_row in features
        ]
        return selected_hybrid_disagreement_indexes(
            transitions=transitions,
            predicted_rows=predicted_rows,
            budget=int(args.cf_transitions),
            disagreement_fraction=float(args.cf_disagreement_fraction),
        )
    if args.cf_selection == "reward":
        budget = args.cf_transitions
        threshold = float(args.cf_reward_threshold)
        context = max(0, int(args.cf_reward_context))
        scored = []
        for idx, transition in enumerate(transitions):
            magnitude = abs(float(transition["team_reward"]))
            if magnitude > threshold:
                scored.append((magnitude, idx))
        selected: List[int] = []
        for _magnitude, idx in sorted(scored, key=lambda item: (-item[0], -item[1])):
            for candidate in range(max(0, idx - context), idx + 1):
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= budget:
                    return sorted(selected)
        for candidate in selected_query_indexes(n_steps, args):
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= budget:
                break
        return sorted(selected)
    raise ValueError(f"unknown cf_selection {args.cf_selection}")


def ranked_credit_indexes(
    transitions: Sequence[Mapping[str, object]],
) -> List[Tuple[float, int]]:
    scored: List[Tuple[float, int]] = []
    for idx, transition in enumerate(transitions):
        raw = np.asarray(transition.get("raw_credit", []), dtype=np.float64).reshape(-1)
        if raw.size == 0 or not bool(np.all(np.isfinite(raw))):
            score = 0.0
        else:
            centered = raw - float(np.mean(raw))
            score = float(np.max(np.abs(centered)))
        scored.append((score, idx))
    return sorted(scored, key=lambda item: (-item[0], -item[1]))


def selected_hybrid_disagreement_indexes(
    transitions: Sequence[Mapping[str, object]],
    predicted_rows: Sequence[Sequence[float]],
    budget: int,
    disagreement_fraction: float = 0.5,
) -> List[int]:
    n_steps = len(transitions)
    if budget <= 0 or budget >= n_steps:
        return list(range(n_steps))
    disagreement_budget = int(
        np.clip(round(float(budget) * float(disagreement_fraction)), 1, budget)
    )
    credit_budget = budget - disagreement_budget
    selected = [idx for _score, idx in ranked_credit_indexes(transitions)[:credit_budget]]

    disagreement_scores: List[Tuple[float, int]] = []
    for idx, transition in enumerate(transitions):
        raw = centered_scaled(transition.get("raw_credit", []))  # type: ignore[arg-type]
        predicted = centered_scaled(predicted_rows[idx]) if idx < len(predicted_rows) else []
        if len(raw) != len(predicted) or not raw:
            score = 0.0
        else:
            score = max(abs(float(left) - float(right)) for left, right in zip(raw, predicted))
        disagreement_scores.append((score, idx))
    for _score, idx in sorted(disagreement_scores, key=lambda item: (-item[0], -item[1])):
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= budget:
            break
    return sorted(selected)


def dataset_examples(
    features: Sequence[Sequence[Sequence[float]]],
    cf_credits_by_t: Mapping[int, Sequence[float]],
) -> List[Dict[str, object]]:
    examples = []
    for t, cf_row in cf_credits_by_t.items():
        for agent_idx, target in enumerate(cf_row):
            examples.append(
                {
                    "features": list(features[t][agent_idx]),
                    "target": float(target),
                }
            )
    return examples


def surrogate_verification_mse(
    features: Sequence[Sequence[Sequence[float]]],
    cf_credits_by_t: Mapping[int, Sequence[float]],
    credit_model: CreditModel,
) -> float | None:
    errors = []
    for t, cf_row in cf_credits_by_t.items():
        if t >= len(features):
            continue
        pred_row = centered_scaled(
            [credit_model.predict(agent_features) for agent_features in features[t]]
        )
        for pred_value, target_value in zip(pred_row, cf_row):
            errors.append((float(pred_value) - float(target_value)) ** 2)
    if not errors:
        return None
    return mean(errors)


def reliability_from_mse(mse: float | None, mse_gate: float) -> float:
    if mse is None:
        return 0.0
    if mse_gate <= 0.0:
        return 1.0
    return float(np.clip(1.0 - float(mse) / float(mse_gate), 0.0, 1.0))


def scheduled_entropy_coef(
    args: argparse.Namespace, env_steps_seen: int, surrogate_reliability: float
) -> float:
    if not args.entropy_curriculum:
        return float(args.entropy_coef)
    decay_steps = max(int(args.entropy_decay_env_steps), 1)
    progress = float(np.clip(float(env_steps_seen) / float(decay_steps), 0.0, 1.0))
    start = float(args.entropy_start_coef)
    end = float(args.entropy_end_coef)
    scheduled = end + (start - end) * (1.0 - progress)
    if args.entropy_reliability_decay:
        reliability = float(np.clip(surrogate_reliability, 0.0, 1.0))
        scheduled = end + (scheduled - end) * (1.0 - reliability)
    return float(max(min(start, end), min(max(start, end), scheduled)))


def fit_credit_model(dataset: Sequence[Dict[str, object]], ridge: float) -> CreditModel:
    if not dataset:
        return CreditModel(weights=[0.0])
    x = np.asarray([item["features"] for item in dataset], dtype=np.float64)
    y = np.asarray([float(item["target"]) for item in dataset], dtype=np.float64)
    reg = ridge * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    try:
        weights = np.linalg.solve(x.T @ x + reg, x.T @ y)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(x.T @ x + reg) @ x.T @ y
    return CreditModel(weights=weights.tolist())


def fit_credit_model_for_args(
    dataset: Sequence[Dict[str, object]], args: argparse.Namespace
) -> CreditModel:
    return fit_credit_model(dataset, ridge=float(args.ridge))


def logged_horizon_return(
    transitions: Sequence[Dict[str, object]],
    step_idx: int,
    horizon: int,
) -> float:
    end = min(len(transitions), step_idx + horizon)
    return sum(float(transitions[idx]["team_reward"]) for idx in range(step_idx, end))


def counterfactual_horizon_for_step(
    transitions: Sequence[Mapping[str, object]],
    step_idx: int,
    base_horizon: int,
    event_horizon_max: int,
    reward_threshold: float,
    post_event_steps: int = 0,
) -> int:
    base = max(1, int(base_horizon))
    maximum = max(0, int(event_horizon_max))
    if maximum <= base or step_idx < 0 or step_idx >= len(transitions):
        return base
    end = min(len(transitions), step_idx + maximum)
    for event_idx in range(step_idx, end):
        if abs(float(transitions[event_idx]["team_reward"])) <= float(reward_threshold):
            continue
        required = event_idx - step_idx + 1 + max(0, int(post_event_steps))
        return max(base, min(maximum, required))
    return base


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    out = [0.0 for _ in rewards]
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def centered_scaled(values: Sequence[float], eps: float = 1e-8) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    scale = float(np.max(np.abs(arr)))
    if scale > eps:
        arr = arr / scale
    return clip_row(arr.tolist())


def centered_clipped(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    return clip_row(arr.tolist())


def clip_row(values: Sequence[float], limit: float = 1.0) -> List[float]:
    return [float(np.clip(value, -limit, limit)) for value in values]


def one_hot(idx: int, n: int) -> np.ndarray:
    values = np.zeros(n, dtype=np.float32)
    values[idx] = 1.0
    return values


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def load_episode_seed_curriculum(path: Path) -> List[int]:
    if not str(path) or str(path) == ".":
        return []
    if path.is_dir():
        return []
    if not path.exists():
        raise FileNotFoundError(f"episode seed curriculum file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload
    if isinstance(payload, dict):
        rows = payload.get("seeds", payload.get("selected_seeds", []))
    seeds: List[int] = []
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, dict) and "seed" in item:
                seeds.append(int(item["seed"]))
            elif isinstance(item, (int, float, str)):
                seeds.append(int(item))
    return seeds


def default_episode_seed(args: argparse.Namespace, episode_idx: int) -> int:
    if args.episode_seed_base is not None:
        return int(args.episode_seed_base) + int(episode_idx)
    return int(args.seed) * 100000 + int(episode_idx)


def select_episode_seed(
    args: argparse.Namespace,
    episode_idx: int,
    env_steps_seen: int,
    curriculum_seeds: Sequence[int],
) -> int:
    if curriculum_seeds:
        active_steps = int(args.episode_seed_curriculum_env_steps)
        curriculum_active = active_steps <= 0 or int(env_steps_seen) < active_steps
        if curriculum_active:
            return int(curriculum_seeds[(int(episode_idx) - 1) % len(curriculum_seeds)])
    return default_episode_seed(args, episode_idx)


def should_query(episode_idx: int, args: argparse.Namespace) -> bool:
    if episode_idx <= args.bootstrap_queries:
        return True
    return args.query_every > 0 and episode_idx % args.query_every == 0
