"""Train SCV using the profiles behind the supplied eight-task table.

Example: python train_scv.py lbf8 --seed 31001 --run-dir ./runs/lbf8-31001
Use --list to see the supported task/seed pairs. A run directory must be new
or empty; the trainer writes its own checkpoints and measurements there.
The frozen candidate-credit programs are included as Python, not model weights.

Training/evaluation tested with Python 3.10.20, PyTorch 2.12.1, NumPy 2.2.6,
Gymnasium 1.3.0, lbforaging 2.0.0, rware 2.0.0, PettingZoo 1.25.0 and
pygame 2.6.1. No LLM installation is needed to run the frozen SCV programs.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

from scv_config import TASK_SEEDS, stage_args


def backend_for(task):
    if task not in TASK_SEEDS:
        raise ValueError(f"Unknown SCV task: {task}")
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    name = "scv_transfer" if task in ("lbf8", "lbf10", "lbf15", "rware") else "scv_core"
    return importlib.import_module(name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", nargs="?", choices=tuple(TASK_SEEDS))
    parser.add_argument(
        "--seed", type=int, help="archived training seed (default: first for this task)"
    )
    parser.add_argument("--run-dir", type=Path, help="new/empty output directory for this seed")
    parser.add_argument(
        "--list", action="store_true", help="list tasks/seeds and exit without training"
    )
    cli = parser.parse_args()
    if cli.list:
        for task, seeds in TASK_SEEDS.items():
            print(f"{task}: {', '.join(map(str, seeds))}")
        return
    if cli.task is None:
        parser.error("select a task, or use --list")
    seed = cli.seed if cli.seed is not None else TASK_SEEDS[cli.task][0]
    if seed not in TASK_SEEDS[cli.task]:
        parser.error(f"unsupported training seed for {cli.task}; see --list")
    directory = (cli.run_dir or Path("runs") / f"{cli.task}-{seed}").resolve()
    stages = stage_args(cli.task, seed, directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        parser.error(f"will not overwrite a nonempty run directory: {directory}")
    backend = backend_for(cli.task)
    directory.mkdir(parents=True, exist_ok=True)
    for index, args in enumerate(stages, 1):
        print(
            f"SCV | {cli.task} | seed {seed} | stage {index}/{len(stages)} | target {args.total_env_steps}",
            flush=True,
        )
        backend.train(args)
    print(f"Final checkpoint: {stages[-1].checkpoint_path}")


if __name__ == "__main__":
    main()
