#!/usr/bin/env python3
"""Reproduce the Pursuit-Evasion win-rate table and optionally add MIXED.

This is deliberately only an evaluation pipeline.  It uses the PE model and
policy implementations from this repository and does not alter their search
algorithm, transition function, observations, rewards, or terminal rules.
"""

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import intmcp.model as model_lib
from intmcp.envs.pe import PEModel, PESPPolicy
from intmcp.envs.pe import grid as pe_grid
from intmcp.policy import BasePolicy, RandomPolicy
from intmcp.tree import NestedSearchTree


RUNNER_IDX = 0
CHASER_IDX = 1

RANDOM = "random"
SHORTEST_PATH = "shortest path"
INTMCP_PREFIX = "I-NTMCP l="
MIXED = "MIXED"

PURSUER_POLICIES = [
    RANDOM,
    SHORTEST_PATH,
    "I-NTMCP l=0",
    "I-NTMCP l=1",
    "I-NTMCP l=2",
    "I-NTMCP l=3",
]
BASE_EVADER_POLICIES = [
    RANDOM,
    SHORTEST_PATH,
    "I-NTMCP l=0",
    "I-NTMCP l=1",
]

SUMMARY_FIELDS = [
    "pursuer_policy",
    "evader_policy",
    "pursuer_level",
    "evader_level",
    "num_sims",
    "num_episodes",
    "seed",
    "score",
    "wins",
]
EPISODE_FIELDS = [
    "pursuer_policy",
    "evader_policy",
    "sampled_evader_policy",
    "pursuer_level",
    "sampled_evader_level",
    "num_sims",
    "num_episodes",
    "seed",
    "episode",
    "pursuer_win",
    "pursuer_outcome",
    "episode_steps",
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PE Table II policy matrix (pursuer win rate)."
    )
    parser.add_argument("--grid_name", default="8by8")
    parser.add_argument("--num_sims", type=int, default=512)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed list; when supplied, it replaces --seed.",
    )
    parser.add_argument(
        "--include_mixed",
        action="store_true",
        help="Append the episode-level MIXED evader column.",
    )
    parser.add_argument(
        "--mixed_only",
        action="store_true",
        help="Evaluate only the MIXED evader column.",
    )
    parser.add_argument("--output", default="results/pe_table_with_mixed.csv")
    parser.add_argument(
        "--episode_output",
        default="results/pe_table_with_mixed_episodes.csv",
    )
    parser.add_argument("--log_dir", default="results/pe_table_logs")

    # These reproduce the remaining paper settings while making small smoke
    # tests possible without editing the script.
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
    return args


def policy_level(policy_name: str) -> Optional[int]:
    """Return the explicit nesting level, or None for a baseline/MIXED."""
    if policy_name.startswith(INTMCP_PREFIX):
        return int(policy_name[len(INTMCP_PREFIX):])
    return None


def csv_level(policy_name: str) -> object:
    level = policy_level(policy_name)
    return "" if level is None else level


def make_policy(
    policy_name: str,
    model: PEModel,
    agent_id: int,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> BasePolicy:
    """Construct one of the paper policies using the repo implementations."""
    common = {
        "model": model,
        "ego_agent": agent_id,
        "gamma": args.gamma,
        "logger": logger,
    }
    if policy_name == RANDOM:
        return RandomPolicy.initialize(**common)
    if policy_name == SHORTEST_PATH:
        # r_hi/r_lo are the preferred-action values in the paper config. They
        # do not change PESPPolicy's environment-facing shortest-path action.
        return PESPPolicy.initialize(**common, r_hi=90, r_lo=-138)

    level = policy_level(policy_name)
    if level is None:
        raise ValueError("Unknown policy: {}".format(policy_name))

    # This exactly mirrors pe_exp_win_rate.yaml: runner rollouts use the PE
    # shortest-path policy and chaser rollouts use the generic random policy.
    rollout_policies = {
        RUNNER_IDX: (PESPPolicy, {"r_hi": 90, "r_lo": -138}),
        CHASER_IDX: (RandomPolicy, {}),
    }
    return NestedSearchTree.initialize(
        **common,
        nesting_level=level,
        num_sims=args.num_sims,
        rollout_policies=rollout_policies,
        uct_c=args.uct_c,
        epsilon=args.epsilon,
        step_limit=args.step_limit,
    )


def run_episode(
    env: PEModel,
    evader: BasePolicy,
    pursuer: BasePolicy,
    step_limit: int,
) -> Tuple[model_lib.Outcomes, int]:
    """Run one non-rendered episode and return agent 1's outcome and length."""
    state, joint_obs = env.reset()
    evader.reset()
    pursuer.reset()

    steps = 0
    done = False
    while not done and steps < step_limit:
        # PE defines the runner/evader as agent 0 and chaser/pursuer as agent 1.
        actions = model_lib.JointAction(
            (evader.step(joint_obs[RUNNER_IDX]),
             pursuer.step(joint_obs[CHASER_IDX]))
        )
        state, joint_obs, _, done = env.step(state, actions)
        steps += 1

    # At the step limit PE reports a DRAW unless capture/evasion occurred.
    # Scoring from get_outcome avoids guessing from accumulated rewards.
    return env.get_outcome(state)[CHASER_IDX], steps


def make_cell_logger(log_dir: Path, pursuer: str, evader: str, seed: int):
    safe = lambda value: value.lower().replace(" ", "_").replace("=", "")
    name = "pe_table.{}.{}.{}".format(safe(pursuer), safe(evader), seed)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.FileHandler(
        log_dir / "{}_vs_{}_seed_{}.log".format(
            safe(pursuer), safe(evader), seed
        ),
        mode="w",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def selected_evaders(args: argparse.Namespace) -> List[str]:
    if args.mixed_only:
        return [MIXED]
    policies = list(BASE_EVADER_POLICIES)
    if args.include_mixed:
        policies.append(MIXED)
    return policies


def print_table(
    results: Dict[Tuple[str, str], Tuple[int, int]],
    evader_policies: Sequence[str],
) -> None:
    """Print dependency-free table cells in score (wins/episodes) form."""
    headers = ["pursuer \\ evader"] + list(evader_policies)
    rows: List[List[str]] = []
    for pursuer in PURSUER_POLICIES:
        row = [pursuer]
        for evader in evader_policies:
            wins, episodes = results[(pursuer, evader)]
            row.append("{:.3f} ({}/{})".format(wins / episodes, wins, episodes))
        rows.append(row)

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]
    print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def run(args: argparse.Namespace) -> None:
    summary_path = Path(args.output)
    episode_path = Path(args.episode_output)
    log_dir = Path(args.log_dir)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds: Iterable[int] = args.seeds if args.seeds is not None else [args.seed]
    seeds = list(seeds)
    evader_policies = selected_evaders(args)
    totals: Dict[Tuple[str, str], Tuple[int, int]] = {
        (pursuer, evader): (0, 0)
        for pursuer in PURSUER_POLICIES
        for evader in evader_policies
    }

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file, \
            episode_path.open("w", newline="", encoding="utf-8") as episode_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        episode_writer = csv.DictWriter(episode_file, fieldnames=EPISODE_FIELDS)
        summary_writer.writeheader()
        episode_writer.writeheader()

        for seed in seeds:
            for pursuer_name in PURSUER_POLICIES:
                for evader_name in evader_policies:
                    # Re-seeding every table cell makes a run reproducible and
                    # gives matchups the same initial pseudo-random stream.
                    random.seed(seed)
                    np.random.seed(seed)
                    mixed_rng = random.Random(seed)
                    env = PEModel(grid_name=args.grid_name)
                    logger = make_cell_logger(
                        log_dir, pursuer_name, evader_name, seed
                    )
                    logger.info(
                        "start pursuer=%s evader=%s seed=%s episodes=%s sims=%s",
                        pursuer_name, evader_name, seed,
                        args.num_episodes, args.num_sims,
                    )

                    pursuer = make_policy(
                        pursuer_name, env, CHASER_IDX, args, logger
                    )
                    concrete_evaders = (
                        BASE_EVADER_POLICIES
                        if evader_name == MIXED else [evader_name]
                    )
                    evaders = {
                        name: make_policy(name, env, RUNNER_IDX, args, logger)
                        for name in concrete_evaders
                    }

                    wins = 0
                    try:
                        for episode in range(args.num_episodes):
                            # The separate RNG is sampled exactly once here;
                            # the selected policy object is fixed for the full
                            # episode and is never changed inside the step loop.
                            sampled_name = (
                                mixed_rng.choice(BASE_EVADER_POLICIES)
                                if evader_name == MIXED else evader_name
                            )
                            outcome, steps = run_episode(
                                env,
                                evaders[sampled_name],
                                pursuer,
                                args.step_limit,
                            )
                            pursuer_win = outcome == model_lib.Outcomes.WIN
                            wins += int(pursuer_win)
                            episode_writer.writerow({
                                "pursuer_policy": pursuer_name,
                                "evader_policy": evader_name,
                                "sampled_evader_policy": sampled_name,
                                "pursuer_level": csv_level(pursuer_name),
                                "sampled_evader_level": csv_level(sampled_name),
                                "num_sims": args.num_sims,
                                "num_episodes": args.num_episodes,
                                "seed": seed,
                                "episode": episode,
                                "pursuer_win": int(pursuer_win),
                                "pursuer_outcome": str(outcome),
                                "episode_steps": steps,
                            })
                            episode_file.flush()
                            logger.info(
                                "episode=%s sampled_evader=%s outcome=%s steps=%s",
                                episode, sampled_name, outcome, steps,
                            )

                        summary_writer.writerow({
                            "pursuer_policy": pursuer_name,
                            "evader_policy": evader_name,
                            "pursuer_level": csv_level(pursuer_name),
                            "evader_level": csv_level(evader_name),
                            "num_sims": args.num_sims,
                            "num_episodes": args.num_episodes,
                            "seed": seed,
                            "score": wins / args.num_episodes,
                            "wins": wins,
                        })
                        summary_file.flush()
                        old_wins, old_episodes = totals[
                            (pursuer_name, evader_name)
                        ]
                        totals[(pursuer_name, evader_name)] = (
                            old_wins + wins,
                            old_episodes + args.num_episodes,
                        )
                        print(
                            "{} vs {} (seed {}): {:.3f} ({}/{})".format(
                                pursuer_name, evader_name, seed,
                                wins / args.num_episodes,
                                wins, args.num_episodes,
                            ),
                            flush=True,
                        )
                    finally:
                        close_logger(logger)

    print()
    print_table(totals, evader_policies)
    print("\nSummary CSV: {}".format(summary_path))
    print("Episode CSV: {}".format(episode_path))
    print("Logs: {}".format(log_dir))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
