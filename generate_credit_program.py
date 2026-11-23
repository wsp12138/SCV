"""Generate SCV's candidate-credit rules offline (optional; training uses the frozen rules).
 
Usage: python generate_credit_program.py lbf --model-path /path/to/local/qwen3-8b-4bit
Requires mlx-lm and a local MLX model on supported hardware. Writes only the
validated program to standard output; it never modifies the training presets.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict


MAX_TOKENS = {
    'lbf': 700,
    'pursuit_global': 700,
    'rware_global': 900,
}


FEATURES = {
    'lbf': (
        'bias',
        'reward_residual',
        'actual_progress_positive',
        'actual_progress_negative',
        'intended_progress_positive',
        'intended_progress_negative',
        'action_match',
        'valid_load',
        'invalid_load',
        'help_progress',
        'help_adjacent',
    ),
    'pursuit_global': (
        'bias',
        'reward_residual',
        'nearest_progress_positive',
        'nearest_progress_negative',
        'tag_contact',
        'capture_participation',
        'capture_support',
        'pair_closure',
        'redundant_overlap',
        'blocked_move',
        'idle',
    ),
    'rware_global': (
        'bias',
        'reward_residual',
        'search_progress_positive',
        'search_progress_negative',
        'carry_progress_positive',
        'carry_progress_negative',
        'return_progress_positive',
        'return_progress_negative',
        'pickup_requested',
        'pickup_unrequested',
        'delivery',
        'return_drop',
        'premature_drop',
        'blocked_forward',
        'invalid_toggle',
        'idle',
    ),
}


PROMPTS = {
    'lbf': (
        'Return exactly one JSON object and no markdown or prose.\n'
        'You are designing an individualized training-credit rule for cooperative Level-Based Foraging.\n'
        'Agents move on a grid and collect food by jointly using load from adjacent cells; combined adjacent agent levels must reach the food level.\n'
        'The rule scores only actions that were actually executed. It is used as a credit proposal, not as an observation, action recommendation, or policy override.\n'
        'Choose one numeric weight in [-3,3] for every feature below. Positive total score means helpful contribution; negative means harmful contribution.\n'
        'Prefer dense progress feedback, strongly reward effective collection/cooperation, penalize invalid load and movement away, and keep magnitudes conservative because a later counterfactual verifier will calibrate the rule.\n'
        'Set bias to zero. Use clip in [1,5].\n'
        '- bias: constant offset; normally zero because common offsets are unsafe\n'
        '- reward_residual: agent reward minus mean team reward after the executed joint action\n'
        '- actual_progress_positive: executed movement reduced Manhattan distance to a feasible food\n'
        '- actual_progress_negative: executed movement increased that distance\n'
        '- intended_progress_positive: the chosen action would move toward the selected feasible food\n'
        '- intended_progress_negative: the chosen action would move away from it\n'
        '- action_match: executed action matches the local shortest-path/load rule\n'
        '- valid_load: agent loaded adjacent food, weighted by whether visible adjacent levels suffice\n'
        '- invalid_load: agent used load while not adjacent to the selected food\n'
        '- help_progress: an under-level agent moved toward food that requires cooperation\n'
        '- help_adjacent: an under-level agent is adjacent and available to cooperate\n'
        'Required schema:\n'
        '{"program_version":1,"environment":"lbf","weights":{"bias":0.0,"reward_residual":0.0,"actual_progress_positive":0.0,"actual_progress_negative":0.0,"intended_progress_positive":0.0,"intended_progress_negative":0.0,"action_match":0.0,"valid_load":0.0,"invalid_load":0.0,"help_progress":0.0,"help_adjacent":0.0},"clip":3.0}'
    ),
    'pursuit_global': (
        'Return exactly one JSON object and no markdown or prose.\n'
        'You are designing a centralized training-only credit rule for PettingZoo Pursuit with eight pursuers and two-agent captures.\n'
        'Pursuers have local observations and must jointly occupy an evader cell; evaders move stochastically, so apparent nearest-target progress may not be causal.\n'
        'Reward verified capture participation and useful approach behavior; penalize blocked or redundant overlap while keeping ordinary idle behavior conservative.\n'
        'The rule scores only actions that were actually executed. It is used as a credit proposal, not as an observation, action recommendation, or policy override.\n'
        'Choose one numeric weight in [-3,3] for every feature below. Positive total score means helpful contribution; negative means harmful contribution.\n'
        'Keep magnitudes conservative because a later counterfactual verifier will calibrate the rule.\n'
        'Set bias to zero. Use clip in [1,5].\n'
        '- bias: constant offset; set this to zero\n'
        '- reward_residual: the agent reward minus its equal share of team reward\n'
        '- nearest_progress_positive: the executed move reduced distance to the nearest evader\n'
        '- nearest_progress_negative: the executed move increased that distance\n'
        '- tag_contact: the agent ended adjacent to an evader\n'
        '- capture_participation: the agent occupied the cell of an evader removed by the joint action\n'
        '- capture_support: the agent ended next to a captured evader without occupying its cell\n'
        '- pair_closure: the agent and a teammate jointly occupied the captured evader cell\n'
        '- redundant_overlap: the agent overlapped a teammate without completing a capture\n'
        '- blocked_move: a movement action did not change position\n'
        '- idle: the agent selected stay\n'
        'Required schema:\n'
        '{"program_version":1,"environment":"pursuit_global","weights":{"bias":0.0,"reward_residual":0.0,"nearest_progress_positive":0.0,"nearest_progress_negative":0.0,"tag_contact":0.0,"capture_participation":0.0,"capture_support":0.0,"pair_closure":0.0,"redundant_overlap":0.0,"blocked_move":0.0,"idle":0.0},"clip":3.0}'
    ),
    'rware_global': (
        'Return exactly one JSON object and no markdown or prose.\n'
        'You are designing a centralized training-only credit rule for cooperative Multi-Robot Warehouse.\n'
        'The critic may inspect global task progress during training, but its output is only an individualized credit score and is never added to actor observations or used to override actions.\n'
        'Reward the complete retrieve-deliver-return cycle: movement toward requested shelves and workstations, requested pickup, delivery, movement to an empty storage cell, and correct return drop.\n'
        'Penalize picking up unrequested shelves, premature drops, moving away from the current task target, blocked forward actions, and invalid toggles.\n'
        'No-op can be a reasonable collision-avoidance action, so give idle a near-zero magnitude rather than treating it like moving away or an invalid action.\n'
        'The rule scores only actions that were actually executed. It is used as a credit proposal, not as an observation, action recommendation, or policy override.\n'
        'Choose one numeric weight in [-3,3] for every feature below. Positive total score means helpful contribution; negative means harmful contribution.\n'
        'Keep magnitudes conservative because a later counterfactual verifier will calibrate the rule.\n'
        'Set bias to zero. Use clip in [1,5].\n'
        '- bias: constant offset; set this to zero\n'
        '- reward_residual: agent reward minus mean team reward\n'
        '- search_progress_positive: the executed action reduced global distance to the nearest unclaimed requested shelf\n'
        '- search_progress_negative: the executed action increased that distance\n'
        '- carry_progress_positive: while carrying a requested shelf, the action reduced global distance to a workstation\n'
        '- carry_progress_negative: while carrying a requested shelf, the action increased that distance\n'
        '- return_progress_positive: after delivery or an accidental pickup, the action reduced distance to an empty storage cell\n'
        '- return_progress_negative: while returning a shelf, the action increased distance to an empty storage cell\n'
        '- pickup_requested: the action picked up a requested shelf\n'
        '- pickup_unrequested: the action picked up a shelf that was not requested\n'
        '- delivery: the action completed a requested-shelf delivery\n'
        '- return_drop: after delivery, the action correctly released the shelf in an empty storage cell\n'
        '- premature_drop: the action dropped a requested shelf before delivery\n'
        '- blocked_forward: the agent selected forward but its global position did not change\n'
        '- invalid_toggle: toggle load did not change carrying state and produced no delivery\n'
        '- idle: the agent selected no-op\n'
        'Required schema:\n'
        '{"program_version":1,"environment":"rware_global","weights":{"bias":0.0,"reward_residual":0.0,"search_progress_positive":0.0,"search_progress_negative":0.0,"carry_progress_positive":0.0,"carry_progress_negative":0.0,"return_progress_positive":0.0,"return_progress_negative":0.0,"pickup_requested":0.0,"pickup_unrequested":0.0,"delivery":0.0,"return_drop":0.0,"premature_drop":0.0,"blocked_forward":0.0,"invalid_toggle":0.0,"idle":0.0},"clip":3.0}'
    ),
}


class QwenJSONGenerator:
    def __init__(self, model_path: Path, max_tokens: int, enable_thinking: bool = False) -> None:
        from mlx_lm import generate, load

        self._generate = generate
        self.model, self.tokenizer = load(str(model_path))
        self.max_tokens = int(max_tokens)
        self.enable_thinking = bool(enable_thinking)

    def generate(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None) is not None:
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        return self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )


def parse_qwen_json(text: str) -> object:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else balanced_json_object(text)
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def validate_program(parsed: object, environment: str) -> Dict[str, object] | None:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("weights"), dict):
        return None
    weights = parsed["weights"]
    feature_names = FEATURES[environment]
    try:
        numeric = {name: float(weights[name]) for name in feature_names}
        clip = float(parsed.get("clip", 3.0))
    except (KeyError, TypeError, ValueError):
        return None
    if any(not -3.0 <= value <= 3.0 for value in numeric.values()) or not 1.0 <= clip <= 5.0:
        return None
    if abs(numeric["bias"]) > 1e-8:
        return None
    return {
        "program_version": int(parsed.get("program_version", 1)),
        "environment": environment,
        "weights": numeric,
        "clip": clip,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=tuple(PROMPTS))
    parser.add_argument("--model-path", required=True, type=Path)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        parser.error("--model-path must be a local MLX model directory with config.json")
    # Set before the lazy mlx-lm import; this command does not fetch a model.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    environment = args.environment
    qwen = QwenJSONGenerator(model_path, MAX_TOKENS[environment], enable_thinking=False)
    prompt = PROMPTS[environment]
    for attempt in range(3):
        retry = "" if not attempt else "\nFormatting correction: include every required key, keep bias exactly 0, and return valid JSON only."
        raw = qwen.generate(prompt + retry)
        program = validate_program(parse_qwen_json(raw), environment)
        if program is not None:
            print(json.dumps(program, indent=2, sort_keys=True))
            return
    raise RuntimeError("Qwen did not return a valid credit program after three attempts")


if __name__ == "__main__":
    main()
