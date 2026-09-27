#!/usr/bin/env python3
"""Run one controlled PE configuration and seed with two I-NTMCP policies.

Both initial directions are NORTH. Only the evaluator's initial state and the
model's cached starts are set explicitly; normal policy beliefs, observations,
transitions, and rewards are unchanged. In particular, the chaser's initial
belief still samples possible runner goals, rather than knowing the true goal.

Python and NumPy are seeded before setup and again immediately before policy
resets. Equal seeds give reproducibility, NOT synchronized common random numbers:
different nesting levels consume different amounts of randomness. There is no
reseeding between actions or timesteps.

T transitions produce T+1 diagnostic samples, including the initial observation
at t=0 and the terminal observation. All distance statistics include t=0.
First-detection times are blank if never detected. Initial terminal states are
rejected; seeing terminates the episode, so seen=True occurs at most once and
seen-count is not a repeated-observation measure. Unreachable distances are inf.
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import intmcp.model as model_lib
from intmcp.envs.pe import PEModel
from intmcp.envs.pe import grid as pe_grid
from intmcp.envs.pe.shortest_path import all_shortest_paths
from intmcp.envs.pe.state import PEState
from intmcp.policy import BasePolicy
from run_pe_pair import (
    PAIR_EPISODE_FIELDS,
    TIMESTEP_FIELDS,
    get_episode_diagnostics,
    get_observation_diagnostics,
    selected_policy_name,
)
from run_pe_table import (
    CHASER_IDX,
    RUNNER_IDX,
    SUMMARY_FIELDS,
    close_logger,
    make_cell_logger,
    make_policy,
)


IDENTIFIER_FIELDS = [
    "grid", "runner_start", "runner_goal", "chaser_start", "episode_seed",
    "pursuer_level", "evader_level", "num_sims",
]
# Copy/deduplicate schemas without mutating either existing runner's lists.
CONTROLLED_SUMMARY_FIELDS = list(dict.fromkeys(SUMMARY_FIELDS + IDENTIFIER_FIELDS))
CONTROLLED_EPISODE_FIELDS = list(dict.fromkeys(PAIR_EPISODE_FIELDS + IDENTIFIER_FIELDS))
CONTROLLED_TIMESTEP_FIELDS = list(dict.fromkeys(TIMESTEP_FIELDS + IDENTIFIER_FIELDS))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid_name", type=str.lower, default="maze8",
                        choices=sorted(pe_grid.SUPPORTED_GRIDS))
    parser.add_argument("--num_sims", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    for location in ("runner_start", "runner_goal", "chaser_start"):
        parser.add_argument("--" + location, type=int, required=True,
                            help="Integer grid location ID (row * width + column).")
    parser.add_argument("--pursuer_level", type=int, default=1)
    parser.add_argument("--evader_level", type=int, default=0)
    parser.add_argument("--output_prefix", default="results/pe_controlled")
    # Preserve the existing runner's remaining policy and episode settings.
    parser.add_argument("--step_limit", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--uct_c", type=float, default=230.0)
    args = parser.parse_args(argv)
    for name in ("num_sims", "step_limit"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive.".format(name))
    for name in ("pursuer_level", "evader_level"):
        if getattr(args, name) < 0:
            parser.error("--{} must be non-negative.".format(name))
    for name in ("gamma", "epsilon"):
        if not 0.0 < getattr(args, name) < 1.0:
            parser.error("--{} must be between 0 and 1 (exclusive).".format(name))
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1 for NumPy.")
    return args


def controlled_initial_state(
    env: PEModel, runner_start: int, runner_goal: int, chaser_start: int,
) -> Tuple[PEState, model_lib.JointObservation]:
    """Validate and install starts without sampling or narrowing goal beliefs."""
    grid = env.grid
    for label, loc in (("runner_start", runner_start),
                       ("runner_goal", runner_goal),
                       ("chaser_start", chaser_start)):
        if loc not in grid.valid_locs:
            raise ValueError("{}={} is not a traversable grid location.".format(label, loc))
    if runner_start == chaser_start:
        raise ValueError("runner_start must differ from chaser_start.")
    if runner_goal == runner_start:
        raise ValueError("runner_goal must differ from runner_start.")
    if runner_goal == chaser_start:
        raise ValueError("runner_goal must differ from chaser_start.")
    if runner_start not in grid.runner_start_locs:
        raise ValueError("runner_start must be one of {}.".format(sorted(grid.runner_start_locs)))
    if chaser_start not in grid.chaser_start_locs:
        raise ValueError("chaser_start must be one of {}.".format(sorted(grid.chaser_start_locs)))
    goals = grid.get_runner_goal_locs(runner_start)
    if runner_goal not in goals:
        raise ValueError("runner_goal must be legal for runner_start={}: {}.".format(
            runner_start, sorted(goals)
        ))

    state = PEState(
        runner_loc=runner_start,
        chaser_loc=chaser_start,
        runner_dir=pe_grid.NORTH,
        chaser_dir=pe_grid.NORTH,
        runner_goal_loc=runner_goal,
        grid=grid,
    )
    if env.is_terminal(state):
        raise ValueError("Initial state is terminal (including chaser visibility).")

    # PE has no public start setter. Its normal beliefs and rollout policies
    # depend on these cached starts. Do not store the evaluator's true goal on
    # the model, change grid goal lists, or replace either initial belief.
    env._runner_start_loc = runner_start
    env._chaser_start_loc = chaser_start
    return state, env.sample_initial_obs(state)


def run_controlled_episode(
    env: PEModel,
    evader: BasePolicy,
    pursuer: BasePolicy,
    state: PEState,
    joint_obs: model_lib.JointObservation,
    seed: int,
    step_limit: int,
    distance_lookup: Dict[int, Dict[int, float]],
) -> Tuple[model_lib.Outcomes, int, PEState, bool, List[Dict[str, object]]]:
    """Use the pair runner's action order, with no environment reset."""
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

    record(0, state, joint_obs)
    # Setup is complete. Both policies use the usual shared global RNG stream.
    random.seed(seed)
    np.random.seed(seed)
    evader.reset()
    pursuer.reset()

    steps = 0
    done = False
    while not done and steps < step_limit:
        actions = model_lib.JointAction((
            evader.step(joint_obs[RUNNER_IDX]),
            pursuer.step(joint_obs[CHASER_IDX]),
        ))
        state, joint_obs, _, done = env.step(state, actions)
        steps += 1
        record(steps, state, joint_obs)
    return env.get_outcome(state)[CHASER_IDX], steps, state, done, samples


def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = PEModel(grid_name=args.grid_name)
    state, joint_obs = controlled_initial_state(
        env, args.runner_start, args.runner_goal, args.chaser_start
    )
    distance_lookup = all_shortest_paths(sorted(env.grid.valid_locs), env.grid)
    pursuer_name = selected_policy_name("intmcp", args.pursuer_level)
    evader_name = selected_policy_name("intmcp", args.evader_level)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path("{}_logs".format(prefix))
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = make_cell_logger(log_dir, pursuer_name, evader_name, args.seed)
    try:
        # The shared factory forwards only planning parameters to policies.
        # The explicit goal/state are never policy inputs.
        pursuer = make_policy(pursuer_name, env, CHASER_IDX, args, logger)
        evader = make_policy(evader_name, env, RUNNER_IDX, args, logger)
        outcome, steps, final_state, done, samples = run_controlled_episode(
            env, evader, pursuer, state, joint_obs, args.seed,
            args.step_limit, distance_lookup,
        )
        win = int(outcome == model_lib.Outcomes.WIN)
        common = {
            "grid": args.grid_name,
            "runner_start": args.runner_start,
            "runner_goal": args.runner_goal,
            "chaser_start": args.chaser_start,
            "episode_seed": args.seed,
            "pursuer_level": args.pursuer_level,
            "evader_level": args.evader_level,
            "num_sims": args.num_sims,
            "pursuer_policy": pursuer_name,
            "evader_policy": evader_name,
            "seed": args.seed,
        }
        episode_row = {
            **common,
            "sampled_evader_policy": evader_name,
            "sampled_evader_level": args.evader_level,
            "num_episodes": 1,
            "episode": 0,
            "pursuer_win": win,
            "pursuer_outcome": str(outcome),
            "episode_steps": steps,
            **get_episode_diagnostics(env, outcome, final_state, done),
            **get_observation_diagnostics(samples),
        }
        summary_row = {**common, "num_episodes": 1, "score": float(win), "wins": win}
        summary_path = Path("{}_summary.csv".format(prefix))
        episode_path = Path("{}_episodes.csv".format(prefix))
        timestep_path = Path("{}_timesteps.csv".format(prefix))
        write_csv(summary_path, CONTROLLED_SUMMARY_FIELDS, [summary_row])
        write_csv(episode_path, CONTROLLED_EPISODE_FIELDS, [episode_row])
        write_csv(timestep_path, CONTROLLED_TIMESTEP_FIELDS, (
            {**common, "episode": 0, **sample} for sample in samples
        ))
        logger.info("episode=0 outcome=%s steps=%s", outcome, steps)
    finally:
        close_logger(logger)
    print("{} vs {} (seed {}): {}, {} transitions, {} samples".format(
        pursuer_name, evader_name, args.seed, outcome, steps, len(samples)
    ))
    print("Summary CSV: {}".format(summary_path))
    print("Episode CSV: {}".format(episode_path))
    print("Timestep CSV: {}".format(timestep_path))


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except ValueError as error:
        raise SystemExit("Error: {}".format(error)) from error


if __name__ == "__main__":
    main()
