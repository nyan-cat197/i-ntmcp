#!/usr/bin/env python3
"""Classify maze8 controlled configurations using static grid geometry only.

The only data input is results/maze8_controlled_configs.csv. All distances are
recomputed from the actual Grid with its existing shortest-path utilities;
cached distances in the input are ignored. No episode, policy, win-rate, or
trajectory data is read, and no environment is instantiated or stepped.

interception_index = d(chaser, goal) - d(chaser, runner).
Positive: INTERCEPTION; zero: BALANCED; negative: GOAL_SIDE.
Discovery configurations are exactly those with runner_start == 61.
Paths are relative to this script, independently of the working directory.
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List

from prettytable import PrettyTable

from intmcp.envs.pe import grid as pe_grid
from intmcp.envs.pe.shortest_path import all_shortest_paths


RESULTS = Path(__file__).resolve().parent / "results"
INPUT = RESULTS / "maze8_controlled_configs.csv"
OUTPUT = RESULTS / "maze8_geometry_classes.csv"
CLASSES = ("INTERCEPTION", "BALANCED", "GOAL_SIDE")
TABLE_FIELDS = [
    "config_id", "runner_start", "runner_goal", "chaser_start",
    "d_chaser_runner", "d_chaser_goal", "d_runner_goal",
    "interception_index", "geometry_class",
]
CSV_FIELDS = ["grid"] + TABLE_FIELDS + ["discovery_configuration"]


def load_configs(path: Path) -> List[Dict[str, int]]:
    """Read only configuration identifiers and locations from the input CSV."""
    fields = ("config_id", "runner_start", "runner_goal", "chaser_start")
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = set(fields + ("grid",)) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing configuration columns: {}".format(sorted(missing)))
        configs = []
        for row in reader:
            if row["grid"] != "maze8":
                raise ValueError("Expected only maze8 configurations.")
            configs.append({field: int(row[field]) for field in fields})
    if not configs:
        raise ValueError("The configuration CSV is empty.")
    if len({row["config_id"] for row in configs}) != len(configs):
        raise ValueError("Configuration IDs must be unique.")
    return sorted(configs, key=lambda row: row["config_id"])


def classify_configs(grid: pe_grid.Grid, configs: List[Dict[str, int]]) -> List[Dict]:
    """Calculate geometry without consulting observations or episode outcomes."""
    for config in configs:
        runner, goal, chaser = (config[key] for key in
                                ("runner_start", "runner_goal", "chaser_start"))
        if not {runner, goal, chaser} <= grid.valid_locs:
            raise ValueError("Config {} contains a non-traversable location.".format(config["config_id"]))
        if (len({runner, goal, chaser}) != 3
                or runner not in grid.runner_start_locs
                or chaser not in grid.chaser_start_locs
                or goal not in grid.get_runner_goal_locs(runner)):
            raise ValueError("Config {} has invalid starts or goal mapping.".format(config["config_id"]))
    sources = sorted({config[key] for config in configs
                      for key in ("runner_start", "chaser_start")})
    distances = all_shortest_paths(sources, grid)
    rows = []
    for config in configs:
        runner, goal, chaser = (config[key] for key in
                                ("runner_start", "runner_goal", "chaser_start"))
        if runner not in distances[chaser] or goal not in distances[runner]:
            raise ValueError("Config {} contains unreachable locations.".format(config["config_id"]))
        d_chaser_runner = int(distances[chaser][runner])
        d_chaser_goal = int(distances[chaser][goal])
        d_runner_goal = int(distances[runner][goal])
        index = d_chaser_goal - d_chaser_runner
        geometry_class = "INTERCEPTION" if index > 0 else "GOAL_SIDE" if index < 0 else "BALANCED"
        rows.append({
            "grid": "maze8", **config,
            "d_chaser_runner": d_chaser_runner,
            "d_chaser_goal": d_chaser_goal,
            "d_runner_goal": d_runner_goal,
            "interception_index": index,
            "geometry_class": geometry_class,
            "discovery_configuration": runner == 61,
        })
    return rows


def main() -> None:
    configs = load_configs(INPUT)
    rows = classify_configs(pe_grid.load_grid("maze8"), configs)
    non_discovery = [row for row in rows if not row["discovery_configuration"]]
    counts = Counter(row["geometry_class"] for row in rows)
    remaining_counts = Counter(row["geometry_class"] for row in non_discovery)
    counts_table = PrettyTable(["geometry_class", "all configurations", "excluding runner_start 61"])
    for geometry_class in CLASSES:
        counts_table.add_row([geometry_class, counts[geometry_class], remaining_counts[geometry_class]])
    print(counts_table)
    print("\nNon-discovery configurations (runner_start != 61):")
    table = PrettyTable(TABLE_FIELDS)
    for row in non_discovery:
        table.add_row([row[field] for field in TABLE_FIELDS])
    print(table)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("\nSaved {} configurations to {}".format(len(rows), OUTPUT))


if __name__ == "__main__":
    main()
