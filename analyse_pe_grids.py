#!/usr/bin/env python3
"""Characterise PE maps using their actual Grid objects.

Run ``python analyse_pe_grids.py`` to print all five maps and write
results/grid_characteristics.csv (relative to this script). The CSV has one
row per grid; location lists and per-pair distances are JSON-encoded columns.
Locations include integer IDs and zero-based [row, column] coordinates.

Distances count orthogonal traversable-cell edges, ignoring agent orientation
and interactions. Runner statistics weight each directed, allowed start-goal
pair equally, using Grid.get_runner_goal_locs rather than a Cartesian product.
Chaser statistics weight all chaser-start/runner-start combinations equally.
Unreachable distances are infinity ("inf" inside JSON), retained in aggregates;
statistics over an empty set are blank. No environment behaviour is changed.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import networkx as nx
from prettytable import PrettyTable

from intmcp.envs.pe import grid as pe_grid


GRID_NAMES = ("8by8", "maze8", "open8", "corridor8", "bottleneck8")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/grid_characteristics.csv"


def build_graph(grid: pe_grid.Grid) -> nx.Graph:
    """Include every traversable cell, even isolated ones."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted(grid.valid_locs))
    for loc in sorted(grid.valid_locs):
        graph.add_edges_from(
            (loc, neighbour)
            for neighbour in grid.get_neighbouring_locs(loc, include_blocks=False)
        )
    return graph


def location_records(grid: pe_grid.Grid, locs: Iterable[int]) -> List[Dict]:
    """Make location IDs and coordinates explicit in the exported data."""
    return [
        {"loc": loc, "coord": list(grid.loc_to_coord(loc))}
        for loc in sorted(set(locs))
    ]


def analyse_grid(grid: pe_grid.Grid, name: str) -> Dict:
    """Compute topology and distances without mutating the supplied grid."""
    graph = build_graph(grid)
    starts = sorted(set(grid.runner_start_locs))
    chasers = sorted(set(grid.chaser_start_locs))
    distances = {
        loc: nx.single_source_shortest_path_length(graph, loc)
        for loc in sorted(set(starts + chasers))
    }
    pairs = [
        (start, goal, distances[start].get(goal, math.inf))
        for start in starts
        for goal in sorted(set(grid.get_runner_goal_locs(start)))
        if goal != start
    ]
    runner_distances = [distance for _, _, distance in pairs]
    chaser_distances = [
        distances[chaser].get(start, math.inf)
        for chaser in chasers for start in starts
    ]
    degrees = [degree for _, degree in graph.degree()]
    return {
        "grid": name,
        "width": grid.width,
        "height": grid.height,
        "traversable_cells": len(grid.valid_locs),
        "wall_cells": len(grid.block_locs),
        "percentage_traversable": 100.0 * len(grid.valid_locs) / grid.num_locs,
        "runner_start_locations": location_records(grid, starts),
        "runner_goal_locations": location_records(grid, grid.runner_goal_locs),
        "chaser_start_locations": location_records(grid, chasers),
        "runner_start_goal_distances": [
            {"start": start, "goal": goal,
             "distance": "inf" if math.isinf(distance) else distance}
            for start, goal, distance in pairs
        ],
        "runner_start_goal_pair_count": len(pairs),
        "runner_unreachable_pair_count": sum(map(math.isinf, runner_distances)),
        "runner_distance_mean": mean(runner_distances) if runner_distances else "",
        "runner_distance_min": min(runner_distances) if runner_distances else "",
        "runner_distance_max": max(runner_distances) if runner_distances else "",
        "chaser_runner_distance_mean": mean(chaser_distances) if chaser_distances else "",
        "chaser_runner_pair_count": len(chaser_distances),
        "chaser_runner_unreachable_pair_count": sum(map(math.isinf, chaser_distances)),
        "degree_0_cells": degrees.count(0),
        "degree_1_cells": degrees.count(1),
        "degree_2_cells": degrees.count(2),
        "degree_3_plus_cells": sum(degree >= 3 for degree in degrees),
        "articulation_points": len(list(nx.articulation_points(graph))),
        "connected_components": nx.number_connected_components(graph),
    }


def format_number(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def format_locations(records: List[Dict]) -> str:
    return ", ".join(
        f"{record['loc']} ({record['coord'][0]}, {record['coord'][1]})"
        for record in records
    )


def print_report(rows: List[Dict]) -> None:
    """Print a compact comparison, followed by locations and every valid pair."""
    columns = {
        "Grid": "grid", "W": "width", "H": "height",
        "Open": "traversable_cells", "Walls": "wall_cells",
        "Open %": "percentage_traversable", "R mean": "runner_distance_mean",
        "R min": "runner_distance_min", "R max": "runner_distance_max",
        "C->R mean": "chaser_runner_distance_mean",
        "Deg 0": "degree_0_cells", "Deg 1": "degree_1_cells",
        "Deg 2": "degree_2_cells", "Deg 3+": "degree_3_plus_cells",
        "Artic.": "articulation_points", "Components": "connected_components",
    }
    table = PrettyTable(list(columns))
    for row in rows:
        table.add_row([format_number(row[key]) for key in columns.values()])
    print(table)
    print("R: runner start->goal; C->R: all chaser-start/runner-start pairs.")
    print("Distances count orthogonal steps; means weight pairs equally.")
    print("Artic.: cells whose removal increases the number of connected components.")
    print("Locations: ID (zero-based row, column). Unreachable: inf, included in means.")
    for row in rows:
        print(f"\n{row['grid']}")
        print("  Runner starts: " + format_locations(row["runner_start_locations"]))
        print("  Runner goals:  " + format_locations(row["runner_goal_locations"]))
        print("  Chaser starts: " + format_locations(row["chaser_start_locations"]))
        print(
            f"  Unreachable pairs: runner {row['runner_unreachable_pair_count']}"
            f"/{row['runner_start_goal_pair_count']}, chaser->runner "
            f"{row['chaser_runner_unreachable_pair_count']}"
            f"/{row['chaser_runner_pair_count']}"
        )
        pair_table = PrettyTable(["Runner start ID", "Goal ID", "Distance"])
        for pair in row["runner_start_goal_distances"]:
            pair_table.add_row([pair["start"], pair["goal"], pair["distance"]])
        print(pair_table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="CSV destination (default: results/grid_characteristics.csv).")
    args = parser.parse_args()
    rows = [analyse_grid(pe_grid.load_grid(name), name) for name in GRID_NAMES]
    print_report(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, allow_nan=False) if isinstance(value, list) else value
                for key, value in row.items()
            })
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
