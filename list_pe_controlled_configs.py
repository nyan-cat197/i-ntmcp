#!/usr/bin/env python3
"""List valid NORTH/NORTH PE starts for a supplied --grid_name.

Configuration IDs are zero-based and assigned in sorted
(runner_start, runner_goal, chaser_start) order after filtering. They are stable
for an unchanged grid and validation rules, independently of random seeds.
Reachability requires both runner->goal and chaser->runner paths through
traversable cells. Distances in the output are between chaser and runner starts.
The CSV is saved under results/ beside this script. No episodes are run.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from prettytable import PrettyTable

from intmcp.envs.pe import PEModel
from intmcp.envs.pe import grid as pe_grid
from intmcp.envs.pe.shortest_path import all_shortest_paths
from intmcp.envs.pe.state import PEState


FIELDS = [
    "config_id", "grid", "runner_start", "runner_goal", "chaser_start",
    "initial_manhattan_distance", "initial_shortest_path_distance",
]


def enumerate_configs(env: PEModel, grid_name: str) -> List[Dict[str, object]]:
    """Use the actual Grid mapping and PE terminal predicate, without reset."""
    grid = env.grid
    starts = sorted(set(grid.runner_start_locs) & grid.valid_locs)
    chasers = sorted(set(grid.chaser_start_locs) & grid.valid_locs)
    distances = all_shortest_paths(sorted(set(starts + chasers)), grid)
    rows = []
    for runner_start in starts:
        # Only mapped goals are candidates; do not use all endpoint pairs.
        goals = sorted(set(grid.get_runner_goal_locs(runner_start)) & grid.valid_locs)
        for runner_goal in goals:
            for chaser_start in chasers:
                if len({runner_start, runner_goal, chaser_start}) != 3:
                    continue
                if runner_goal not in distances[runner_start]:
                    continue
                if runner_start not in distances[chaser_start]:
                    continue
                state = PEState(
                    runner_loc=runner_start,
                    chaser_loc=chaser_start,
                    runner_dir=pe_grid.NORTH,
                    chaser_dir=pe_grid.NORTH,
                    runner_goal_loc=runner_goal,
                    grid=grid,
                )
                if env.is_terminal(state):
                    continue
                rows.append({
                    "config_id": len(rows),
                    "grid": grid_name,
                    "runner_start": runner_start,
                    "runner_goal": runner_goal,
                    "chaser_start": chaser_start,
                    "initial_manhattan_distance": grid.manhattan_dist(
                        chaser_start, runner_start
                    ),
                    "initial_shortest_path_distance": int(
                        distances[chaser_start][runner_start]
                    ),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid_name", required=True, type=str.lower,
                        choices=sorted(pe_grid.SUPPORTED_GRIDS))
    args = parser.parse_args()
    # Construction loads the real grid. Its provisional random starts are
    # irrelevant: enumeration uses every configured start, never env.reset().
    env = PEModel(grid_name=args.grid_name)
    rows = enumerate_configs(env, args.grid_name)
    table = PrettyTable(FIELDS)
    for row in rows:
        table.add_row([row[field] for field in FIELDS])
    print(table)

    output = Path(__file__).resolve().parent / "results" / (
        args.grid_name + "_controlled_configs.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("\n{} valid configurations. Saved {}".format(len(rows), output))


if __name__ == "__main__":
    main()
