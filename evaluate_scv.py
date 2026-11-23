"""Evaluate a trusted SCV final checkpoint under the supplied table's protocol.

Example: python evaluate_scv.py lbf8 /path/to/final.pt
Supply --mode greedy only for a separate greedy analysis, not for recreating
the supplied LBF/RWARE column. No training or checkpoint updates occur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_scv import backend_for
from scv_config import TASK_SEEDS, EVALUATION, stage_args


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", choices=tuple(TASK_SEEDS))
    parser.add_argument(
        "checkpoint", type=Path, help="trusted ordinary final checkpoint, not a best checkpoint"
    )
    parser.add_argument(
        "--mode",
        choices=("stochastic", "greedy"),
        help="override the table's default evaluation mode",
    )
    cli = parser.parse_args()
    engine = backend_for(cli.task)
    import torch

    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    info = dict(checkpoint["env_info"])
    seed = int(args.seed)
    expected = stage_args(cli.task, seed, cli.checkpoint.resolve().parent)[-1]
    if (
        checkpoint.get("status") != "complete"
        or checkpoint.get("is_best") is not False
        or args.env != expected.env
        or info["name"] != expected.env
        or args.credit_usage != "advantage_safe_residual"
        or args.scheme != expected.scheme
        or not args.credit_surrogate
        or int(checkpoint["env_steps_seen"]) < expected.total_env_steps
        or int(args.total_env_steps) != expected.total_env_steps
    ):
        parser.error("not the ordinary final SCV checkpoint for the selected task/budget")
    mode, eval_seed, episodes, metric = EVALUATION[cli.task]
    args.eval_mode = cli.mode or mode
    args.eval_seed, args.eval_episodes = eval_seed, episodes
    args.eval_task_hint_weight, args.eval_temperature, args.eval_top_k = 0.0, 1.0, 0
    args.n_actions_hint_dim = int(info["n_actions"])
    obs_dim = int(info["obs_dim"]) * engine.obs_frame_stack(args)
    actor_dim = (
        obs_dim + (int(info["n_agents"]) if args.agent_id else 0) + engine.task_hint_dim(args, info)
    )
    actor = engine.Actor(actor_dim, int(info["n_actions"]), int(args.hidden))
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    value = float(engine.evaluate(actor, args, torch.device("cpu"), info))
    scale = int(info["n_agents"]) if metric == "team_sum_return" else 1
    print(
        json.dumps(
            dict(
                task=cli.task,
                training_seed=seed,
                evaluation_seed=eval_seed,
                episodes=episodes,
                mode=args.eval_mode,
                metric=metric,
                value=value * scale,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
