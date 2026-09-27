"""Validate completeness, ordering, and rejection of invalid PE starts."""
import copy
import itertools
import random
import unittest
from unittest.mock import patch

import networkx as nx

from intmcp.envs.pe import PEModel
from intmcp.envs.pe.grid import Grid, NORTH, SUPPORTED_GRIDS
from intmcp.envs.pe.state import PEState
from list_pe_controlled_configs import enumerate_configs


class ListControlledConfigsTest(unittest.TestCase):
    def test_all_grids_match_independent_cartesian_filter(self):
        for name in SUPPORTED_GRIDS:
            with self.subTest(grid=name):
                env = PEModel(grid_name=name)
                grid = env.grid
                graph = nx.Graph()
                graph.add_nodes_from(grid.valid_locs)
                for loc in grid.valid_locs:
                    graph.add_edges_from((loc, n) for n in grid.get_neighbouring_locs(loc, False))
                expected = []
                for runner, goal, chaser in itertools.product(
                    grid.runner_start_locs, grid.runner_goal_locs, grid.chaser_start_locs
                ):
                    if len({runner, goal, chaser}) != 3:
                        continue
                    if goal not in grid.get_runner_goal_locs(runner):
                        continue
                    if not nx.has_path(graph, runner, goal) or not nx.has_path(graph, chaser, runner):
                        continue
                    state = PEState(runner, chaser, NORTH, NORTH, goal, grid)
                    if not env.sample_initial_obs(state)[env.CHASER_IDX].seen:
                        expected.append((runner, goal, chaser))
                before = copy.deepcopy(vars(grid))
                cached_starts = (env.runner_start_loc, env.chaser_start_loc)
                rng_before = random.getstate()
                with patch.object(env, "reset", side_effect=AssertionError("reset called")):
                    rows = enumerate_configs(env, name)
                self.assertEqual(random.getstate(), rng_before)
                self.assertEqual(vars(grid), before)
                self.assertEqual((env.runner_start_loc, env.chaser_start_loc), cached_starts)
                actual = [(r["runner_start"], r["runner_goal"], r["chaser_start"]) for r in rows]
                self.assertEqual(actual, sorted(set(expected)))
                self.assertEqual([r["config_id"] for r in rows], list(range(len(rows))))
                for row in rows:
                    self.assertEqual(row["initial_shortest_path_distance"], nx.shortest_path_length(
                        graph, row["chaser_start"], row["runner_start"]
                    ))
                # Input list order and duplicates must not change IDs.
                grid.runner_start_locs = list(reversed(grid.runner_start_locs)) * 2
                grid.chaser_start_locs = list(reversed(grid.chaser_start_locs)) * 2
                self.assertEqual(enumerate_configs(env, name), rows)

    def test_disconnected_overlapping_blocked_and_visible_configs(self):
        env = PEModel(grid_name="maze8")
        # Two components: columns 0-1 and column 3, separated by walls.
        # Mapped goals include a self-goal, a wall, and an unreachable goal.
        env.grid = Grid(3, 4, {2, 6, 10}, [8, 2], {8: [8, 9, 2, 11], 2: [9]},
                        [0, 2, 3, 8, 9, 12])
        rows = enumerate_configs(env, "synthetic")
        self.assertEqual([(r["runner_start"], r["runner_goal"], r["chaser_start"])
                          for r in rows], [(8, 9, 0)])
        self.assertEqual(rows[0]["initial_manhattan_distance"], 2)
        self.assertEqual(rows[0]["initial_shortest_path_distance"], 2)
        # Reverse the relative positions: the north-facing chaser now sees
        # the runner, so every remaining candidate must be rejected.
        env.grid = Grid(3, 4, {2, 6, 10}, [0], {0: [1]}, [8])
        self.assertEqual(enumerate_configs(env, "synthetic"), [])


if __name__ == "__main__":
    unittest.main()
