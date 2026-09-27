#!/usr/bin/env python3
"""Evaluate one Pursuit-Evasion pursuer/evader policy pair.

This is the targeted counterpart to ``run_pe_table.py``.  It imports the
table script's policy factory, CSV schemas, and policy names, and adapts its
episode runner to retain passive final-state, observation, and distance diagnostics.

An episode with T transitions has T+1 observation samples: reset at timestep 0
and every returned step observation, including the terminal one. Distance means
include all these samples. First-detection timesteps are blank if never detected.
Seeing terminates a transition, so post-step seen=True occurs at most once;
seen-count is not a repeated-observation measure. A terminal reset observation
is recorded as a zero-transition episode.
Unreachable shortest-path distances are inf and remain in distance aggregates.
For MIXED runs, timestep evader_policy/evader_level identify the sampled policy.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import intmcp.model as model_lib
from intmcp.envs.pe import PEModel
from intmcp.envs.pe import grid as pe_grid
from intmcp.envs.pe.shortest_path import all_shortest_paths
from intmcp.envs.pe.state import PEState
from intmcp.policy import BasePolicy
from run_pe_table import (
    BASE_EVADER_POLICIES,
    CHASER_IDX,
    EPISODE_FIELDS,
    INTMCP_PREFIX,
    MIXED,
    RANDOM,
    RUNNER_IDX,
    SHORTEST_PATH,
    SUMMARY_FIELDS,
    close_logger,
    csv_level,
    make_cell_logger,
    make_policy,
)


# Keep run_pe_table.py's episode columns in their original order and append
# pair-runner-only diagnostics.  Do not mutate the imported shared list.
DIAGNOSTIC_EPISODE_FIELDS = [
    "termination_reason",
    "evader_reached_goal",
    "pursuer_caught_evader",
    "timeout",
    "final_pursuer_position",
    "final_evader_position",
    "evader_goal",
    "final_runner_loc",
    "final_chaser_loc",
    "final_runner_goal_loc",
    "final_runner_dir",
    "final_chaser_dir",
    "likely_termination_reason",
    "pursuer_ever_saw_runner",
    "first_runner_seen_timestep",
    "runner_seen_timesteps",
    "pursuer_ever_heard_runner",
    "first_runner_heard_timestep",
    "runner_heard_timesteps",
    "initial_manhattan_distance",
    "final_manhattan_distance",
    "min_manhattan_distance",
    "mean_manhattan_distance",
    "initial_shortest_path_distance",
    "final_shortest_path_distance",
    "min_shortest_path_distance",
    "mean_shortest_path_distance",
]
PAIR_EPISODE_FIELDS = list(EPISODE_FIELDS) + DIAGNOSTIC_EPISODE_FIELDS
TIMESTEP_FIELDS = [
    "grid", "pursuer_policy", "pursuer_level", "evader_policy", "evader_level",
    "seed", "episode", "timestep", "pursuer_seen", "pursuer_heard",
    "runner_loc", "chaser_loc", "manhattan_distance", "shortest_path_distance",
    "grid_name",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one PE pursuer/evader matchup (pursuer win rate)."
    )
    parser.add_argument("--grid_name", default="8by8")
    parser.add_argument("--num_sims", type=int, default=512)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pursuer",
        required=True,
        choices=("random", "shortest_path", "intmcp"),
    )
    parser.add_argument(
        "--pursuer_level",
        type=int,
        default=0,
        help="Nesting level used only when --pursuer=intmcp.",
    )
    parser.add_argument(
        "--evader",
        required=True,
        choices=("random", "shortest_path", "intmcp", "mixed"),
    )
    parser.add_argument(
        "--evader_level",
        type=int,
        default=0,
        help="Nesting level used only when --evader=intmcp.",
    )
    parser.add_argument(
        "--output_prefix",
        default="results/pe_pair",
        help="Prefix for _summary.csv, _episodes.csv, and _timesteps.csv files.",
    )
    parser.add_argument(
        "--debug_episode_info",
        action="store_true",
        help=(
            "Print final state/result attributes after every episode. "
            "Useful because PE does not expose a termination-reason field."
        ),
    )

    # Copied from run_pe_table.py: these defaults reproduce the remaining
    # paper settings and are consumed by its shared make_policy helper.
    parser.add_argument("--step_limit", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--uct_c", type=float, default=230.0)

    args = parser.parse_args(argv)
    args.grid_name = args.grid_name.lower()
    if args.grid_name not in pe_grid.SUPPORTED_GRIDS:
        supported = ", ".join(sorted(pe_grid.SUPPORTED_GRIDS))
        parser.error("--grid_name must be one of: {}.".format(supported))
    if args.num_sims <= 0:
        parser.error("--num_sims must be positive.")
    if args.num_episodes <= 0:
        parser.error("--num_episodes must be positive.")
    if args.step_limit <= 0:
        parser.error("--step_limit must be positive.")
    if not 0.0 < args.gamma < 1.0:
        parser.error("--gamma must be between 0 and 1 (exclusive).")
    if not 0.0 < args.epsilon < 1.0:
        parser.error("--epsilon must be between 0 and 1 (exclusive).")
    if args.pursuer == "intmcp" and args.pursuer_level < 0:
        parser.error("--pursuer_level must be non-negative for I-NTMCP.")
    if args.evader == "intmcp" and args.evader_level < 0:
        parser.error("--evader_level must be non-negative for I-NTMCP.")
    return args


def selected_policy_name(policy: str, level: int) -> str:
    """Translate CLI spellings to the exact names used by run_pe_table.py."""
    if policy == "random":
        return RANDOM
    if policy == "shortest_path":
        return SHORTEST_PATH
    if policy == "intmcp":
        return "{}{}".format(INTMCP_PREFIX, level)
    if policy == "mixed":
        return MIXED
    raise ValueError("Unknown policy: {}".format(policy))


def run_episode_with_final_state(
    env: PEModel,
    evader: BasePolicy,
    pursuer: BasePolicy,
    step_limit: int,
    distance_lookup: Dict[int, Dict[int, float]],
) -> Tuple[model_lib.Outcomes, int, PEState, bool, List[Dict[str, object]]]:
    """Run one episode and retain passive diagnostics for t=0 through t=T.

    Adapted from run_pe_table.py's run_episode.  The reset, action ordering,
    transition loop, step limit, and outcome calculation are unchanged.
    Samples contain plain diagnostic values, never supplied to either policy.
    """
    samples = []

    def record(timestep, state, joint_obs):
        samples.append({
            "timestep": timestep,
            "pursuer_seen": int(joint_obs[CHASER_IDX].seen),
            "pursuer_heard": int(joint_obs[CHASER_IDX].heard),
            "runner_loc": state.runner_loc,
            "chaser_loc": state.chaser_loc,
            "manhattan_distance": env.grid.manhattan_dist(
                state.chaser_loc, state.runner_loc
            ),
            "shortest_path_distance": distance_lookup[state.chaser_loc].get(
                state.runner_loc, float("inf")
            ),
        })

    state, joint_obs = env.reset()
    record(0, state, joint_obs)
    evader.reset()
    pursuer.reset()

    steps = 0
    done = env.is_terminal(state)
    while not done and steps < step_limit:
        actions = model_lib.JointAction(
            (
                evader.step(joint_obs[RUNNER_IDX]),
                pursuer.step(joint_obs[CHASER_IDX]),
            )
        )
        state, joint_obs, _, done = env.step(state, actions)
        steps += 1
        record(steps, state, joint_obs)

    assert isinstance(state, PEState)
    return env.get_outcome(state)[CHASER_IDX], steps, state, done, samples


def get_observation_diagnostics(samples: List[Dict[str, object]]) -> Dict[str, object]:
    """Summarise every real observation, including reset and terminal samples."""
    diagnostics = {}
    for signal, verb in (("seen", "saw"), ("heard", "heard")):
        timesteps = [
            sample["timestep"] for sample in samples
            if sample["pursuer_" + signal]
        ]
        diagnostics["pursuer_ever_" + verb + "_runner"] = int(bool(timesteps))
        diagnostics["first_runner_" + signal + "_timestep"] = (
            timesteps[0] if timesteps else ""
        )
        diagnostics["runner_" + signal + "_timesteps"] = len(timesteps)
    for distance in ("manhattan_distance", "shortest_path_distance"):
        values = [sample[distance] for sample in samples]
        diagnostics["initial_" + distance] = values[0]
        diagnostics["final_" + distance] = values[-1]
        diagnostics["min_" + distance] = min(values)
        diagnostics["mean_" + distance] = sum(values) / len(values)
    return diagnostics


def get_episode_diagnostics(
    env: PEModel,
    outcome: model_lib.Outcomes,
    final_state: PEState,
    terminated_by_env: bool,
) -> Dict[str, object]:
    """Derive diagnostics using PEModel.get_outcome's terminal semantics."""
    timeout = not terminated_by_env
    pursuer_caught_evader = (
        terminated_by_env and outcome == model_lib.Outcomes.WIN
    )
    # This is intentionally the direct state comparison requested for the CSV,
    # independent of inferred outcome or termination labels.
    evader_reached_goal = (
        final_state.runner_loc == final_state.runner_goal_loc
    )

    if outcome == model_lib.Outcomes.WIN:
        likely_termination_reason = "PURSUER_WIN"
    elif evader_reached_goal:
        likely_termination_reason = "EVADER_REACHED_GOAL"
    else:
        likely_termination_reason = "OTHER_LOSS_OR_TIMEOUT"

    if pursuer_caught_evader:
        termination_reason = "pursuer_caught_evader"
    elif evader_reached_goal:
        termination_reason = "evader_reached_goal"
    elif timeout:
        termination_reason = "timeout"
    else:
        # This should not occur for PE, but makes a future model change visible
        # instead of incorrectly labelling it as capture, evasion, or timeout.
        termination_reason = "unknown"

    # PEState stores integer grid locations.  CSV positions use readable
    # zero-based (row, column) coordinates derived without changing the state.
    return {
        "termination_reason": termination_reason,
        "evader_reached_goal": evader_reached_goal,
        "pursuer_caught_evader": int(pursuer_caught_evader),
        "timeout": int(timeout),
        "final_pursuer_position": str(
            env.grid.loc_to_coord(final_state.chaser_loc)
        ),
        "final_evader_position": str(
            env.grid.loc_to_coord(final_state.runner_loc)
        ),
        "evader_goal": str(
            env.grid.loc_to_coord(final_state.runner_goal_loc)
        ),
        # Raw PEState values are logged alongside the readable coordinates.
        "final_runner_loc": final_state.runner_loc,
        "final_chaser_loc": final_state.chaser_loc,
        "final_runner_goal_loc": final_state.runner_goal_loc,
        "final_runner_dir": final_state.runner_dir,
        "final_chaser_dir": final_state.chaser_dir,
        "likely_termination_reason": likely_termination_reason,
    }


def print_debug_episode_info(
    episode: int,
    final_state: PEState,
    outcome: model_lib.Outcomes,
    terminated_by_env: bool,
) -> None:
    """Print available final state/result attributes on explicit request."""
    state_attributes = sorted(vars(final_state))
    state_values = {
        name: repr(value) for name, value in vars(final_state).items()
    }
    result_attributes = sorted(
        name for name in dir(outcome) if not name.startswith("__")
    )
    print("Episode {} diagnostic info:".format(episode), flush=True)
    print(
        "  final state type: {}".format(type(final_state).__name__),
        flush=True,
    )
    print("  final state attributes: {}".format(state_attributes), flush=True)
    print("  final state values: {}".format(state_values), flush=True)
    print(
        "  episode result: outcome={!s}, terminated_by_env={!r}".format(
            outcome, terminated_by_env
        ),
        flush=True,
    )
    print("  result attributes: {}".format(result_attributes), flush=True)


def run(args: argparse.Namespace) -> None:
    pursuer_name = selected_policy_name(args.pursuer, args.pursuer_level)
    evader_name = selected_policy_name(args.evader, args.evader_level)

    output_prefix = Path(args.output_prefix)
    summary_path = Path("{}_summary.csv".format(output_prefix))
    episode_path = Path("{}_episodes.csv".format(output_prefix))
    timestep_path = Path("{}_timesteps.csv".format(output_prefix))
    log_dir = Path("{}_logs".format(output_prefix))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Adapted from run_pe_table.py's innermost table-cell loop: there is no
    # policy-matrix loop here, so exactly this one requested pair is run.
    random.seed(args.seed)
    np.random.seed(args.seed)
    mixed_rng = random.Random(args.seed)
    env = PEModel(grid_name=args.grid_name)
    logger = make_cell_logger(log_dir, pursuer_name, evader_name, args.seed)
    logger.info(
        "start pursuer=%s evader=%s seed=%s episodes=%s sims=%s",
        pursuer_name,
        evader_name,
        args.seed,
        args.num_episodes,
        args.num_sims,
    )

    # Policy construction is imported from run_pe_table.py, preserving its
    # RandomPolicy, PESPPolicy, and NestedSearchTree classes and parameters.
    pursuer = make_policy(pursuer_name, env, CHASER_IDX, args, logger)
    concrete_evaders = (
        BASE_EVADER_POLICIES if evader_name == MIXED else [evader_name]
    )
    evaders = {
        name: make_policy(name, env, RUNNER_IDX, args, logger)
        for name in concrete_evaders
    }

    # Diagnostic-owned lookup: deterministic, read-only use of the grid, once
    # per run. Neither this lookup nor true-state samples enter either policy.
    distance_lookup = all_shortest_paths(sorted(env.grid.valid_locs), env.grid)

    wins = 0
    try:
        with summary_path.open(
            "w", newline="", encoding="utf-8"
        ) as summary_file, episode_path.open(
            "w", newline="", encoding="utf-8"
        ) as episode_file, timestep_path.open(
            "w", newline="", encoding="utf-8"
        ) as timestep_file:
            summary_writer = csv.DictWriter(
                summary_file, fieldnames=SUMMARY_FIELDS
            )
            episode_writer = csv.DictWriter(
                episode_file, fieldnames=PAIR_EPISODE_FIELDS
            )
            timestep_writer = csv.DictWriter(timestep_file, fieldnames=TIMESTEP_FIELDS)
            summary_writer.writeheader()
            episode_writer.writeheader()
            timestep_writer.writeheader()

            # Adapted from run_pe_table.py: MIXED is sampled once per episode,
            # then the selected evader remains fixed for that whole episode.
            for episode in range(args.num_episodes):
                sampled_name = (
                    mixed_rng.choice(BASE_EVADER_POLICIES)
                    if evader_name == MIXED
                    else evader_name
                )
                outcome, steps, final_state, terminated_by_env, samples = (
                    run_episode_with_final_state(
                        env,
                        evaders[sampled_name],
                        pursuer,
                        args.step_limit,
                        distance_lookup,
                    )
                )
                pursuer_win = outcome == model_lib.Outcomes.WIN
                wins += int(pursuer_win)
                episode_row = {
                    "grid_name": args.grid_name,
                    "pursuer_policy": pursuer_name,
                    "evader_policy": evader_name,
                    "sampled_evader_policy": sampled_name,
                    "pursuer_level": csv_level(pursuer_name),
                    "sampled_evader_level": csv_level(sampled_name),
                    "num_sims": args.num_sims,
                    "num_episodes": args.num_episodes,
                    "seed": args.seed,
                    "episode": episode,
                    "pursuer_win": int(pursuer_win),
                    "pursuer_outcome": str(outcome),
                    "episode_steps": steps,
                }
                episode_row.update(
                    get_episode_diagnostics(
                        env, outcome, final_state, terminated_by_env
                    )
                )
                episode_row.update(get_observation_diagnostics(samples))
                episode_writer.writerow(episode_row)
                episode_file.flush()
                timestep_metadata = {
                    "grid_name": args.grid_name,
                    "grid": args.grid_name,
                    "pursuer_policy": pursuer_name,
                    "pursuer_level": csv_level(pursuer_name),
                    "evader_policy": sampled_name,
                    "evader_level": csv_level(sampled_name),
                    "seed": args.seed,
                    "episode": episode,
                }
                timestep_writer.writerows(
                    {**timestep_metadata, **sample} for sample in samples
                )
                timestep_file.flush()
                if args.debug_episode_info:
                    print_debug_episode_info(
                        episode, final_state, outcome, terminated_by_env
                    )
                logger.info(
                    "episode=%s sampled_evader=%s outcome=%s steps=%s",
                    episode,
                    sampled_name,
                    outcome,
                    steps,
                )

            score = wins / args.num_episodes
            summary_writer.writerow({
                "grid_name": args.grid_name,
                "pursuer_policy": pursuer_name,
                "evader_policy": evader_name,
                "pursuer_level": csv_level(pursuer_name),
                "evader_level": csv_level(evader_name),
                "num_sims": args.num_sims,
                "num_episodes": args.num_episodes,
                "seed": args.seed,
                "score": score,
                "wins": wins,
            })
            summary_file.flush()
    finally:
        close_logger(logger)

    print(
        "{} vs {} (seed {}): {:.3f} ({}/{})".format(
            pursuer_name,
            evader_name,
            args.seed,
            score,
            wins,
            args.num_episodes,
        ),
        flush=True,
    )
    print("Summary CSV: {}".format(summary_path))
    print("Episode CSV: {}".format(episode_path))
    print("Timestep CSV: {}".format(timestep_path))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
