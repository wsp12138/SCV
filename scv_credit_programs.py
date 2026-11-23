"""Frozen, executable candidate-credit specifications used by SCV.

The three offline Qwen programs are reused across tasks; training consumes
these weights directly, without downloading or querying a language model.
"""

from copy import deepcopy


_PROGRAMS = {'qwen_lbf_credit_program': {'clip': 3.0,
                             'environment': 'lbf',
                             'program_version': 1,
                             'weights': {'action_match': 1.0,
                                         'actual_progress_negative': -1.0,
                                         'actual_progress_positive': 1.0,
                                         'bias': 0.0,
                                         'help_adjacent': 1.0,
                                         'help_progress': 1.0,
                                         'intended_progress_negative': -1.0,
                                         'intended_progress_positive': 1.0,
                                         'invalid_load': -1.0,
                                         'reward_residual': 1.0,
                                         'valid_load': 1.0}},
 'qwen_pursuit_global_credit_program': {'clip': 3.0,
                                        'environment': 'pursuit_global',
                                        'program_version': 1,
                                        'weights': {'bias': 0.0,
                                                    'blocked_move': -0.5,
                                                    'capture_participation': 2.0,
                                                    'capture_support': 0.5,
                                                    'idle': -0.5,
                                                    'nearest_progress_negative': -0.5,
                                                    'nearest_progress_positive': 0.5,
                                                    'pair_closure': 1.5,
                                                    'redundant_overlap': -1.0,
                                                    'reward_residual': 0.0,
                                                    'tag_contact': 1.0}},
 'qwen_rware_cycle_credit_program': {'clip': 3.0,
                                     'environment': 'rware_global',
                                     'program_version': 1,
                                     'weights': {'bias': 0.0,
                                                 'blocked_forward': -0.5,
                                                 'carry_progress_negative': -0.5,
                                                 'carry_progress_positive': 0.5,
                                                 'delivery': 1.0,
                                                 'idle': 0.1,
                                                 'invalid_toggle': -1.0,
                                                 'pickup_requested': 1.0,
                                                 'pickup_unrequested': -1.0,
                                                 'premature_drop': -1.0,
                                                 'return_drop': 1.0,
                                                 'return_progress_negative': -0.5,
                                                 'return_progress_positive': 0.5,
                                                 'reward_residual': 0.0,
                                                 'search_progress_negative': -0.5,
                                                 'search_progress_positive': 0.5}}}


def get_program(selector: str) -> dict:
    """Return a fresh copy; the trainer validates and normalizes the weights."""
    try:
        return deepcopy(_PROGRAMS[selector])
    except KeyError as exc:
        raise ValueError(f"Unknown SCV candidate-credit program: {selector!r}") from exc
