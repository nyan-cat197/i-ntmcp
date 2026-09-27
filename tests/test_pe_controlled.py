"""Checks for controlled initialization, passive logging, and reproducibility."""
import contextlib
import copy
import csv
import io
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import run_pe_controlled as controlled
from intmcp.envs.pe.action import PEAction
from intmcp.envs.pe.obs import PEChaserObs, PERunnerObs


class ControlledPETest(unittest.TestCase):
    def setUp(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        self.env = controlled.PEModel(grid_name="maze8")

    def tearDown(self):
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)

    def test_initial_state_preserves_rng_grid_and_normal_beliefs(self):
        grid_before = copy.deepcopy(vars(self.env.grid))
        before = random.getstate()
        numpy_before = np.random.get_state()
        state, obs = controlled.controlled_initial_state(self.env, 56, 3, 26)
        self.assertEqual(random.getstate(), before)
        numpy_after = np.random.get_state()
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(vars(self.env.grid), grid_before)
        self.assertEqual((state.runner_loc, state.runner_goal_loc, state.chaser_loc),
                         (56, 3, 26))
        self.assertEqual((state.runner_dir, state.chaser_dir),
                         (controlled.pe_grid.NORTH, controlled.pe_grid.NORTH))
        self.assertEqual((self.env.runner_start_loc, self.env.chaser_start_loc), (56, 26))
        self.assertIsInstance(obs[0], PERunnerObs)
        self.assertIsInstance(obs[1], PEChaserObs)
        self.assertFalse(hasattr(obs[1], "goal"))
        self.assertFalse(hasattr(obs[1], "runner_loc"))
        self.assertEqual(self.env.get_init_belief(0, obs[0]).sample(), state)
        random.seed(0)
        belief = self.env.get_init_belief(1, obs[1])
        particles = [belief.sample() for _ in range(64)]
        self.assertEqual({p.runner_goal_loc for p in particles}, {0, 3, 5})
        self.assertTrue(all((p.runner_loc, p.chaser_loc) == (56, 26) for p in particles))

    def test_invalid_starts_and_goals_do_not_change_cached_starts(self):
        starts = (self.env.runner_start_loc, self.env.chaser_start_loc)
        cases = [
            (-1, 3, 26), (64, 3, 26), (2, 3, 26), (1, 3, 26),
            (56, 3, 1), (56, 59, 26), (56, 56, 26),
            (56, 26, 26), (56, 3, 56), (56, 2, 26), (56, 3, 2),
        ]
        for locations in cases:
            with self.subTest(locations=locations), self.assertRaises(ValueError):
                controlled.controlled_initial_state(self.env, *locations)
            self.assertEqual((self.env.runner_start_loc, self.env.chaser_start_loc), starts)

    def test_initial_visibility_is_rejected(self):
        # On open8, the north-facing centre chaser sees runner start (0, 3).
        env = controlled.PEModel(grid_name="open8")
        with self.assertRaisesRegex(ValueError, "terminal"):
            controlled.controlled_initial_state(env, 3, 56, 27)

    def test_episode_call_order_observation_inputs_and_timeout(self):
        state, obs = controlled.controlled_initial_state(self.env, 56, 3, 26)
        distances = controlled.all_shortest_paths(sorted(self.env.grid.valid_locs), self.env.grid)
        events = []
        evader, pursuer = Mock(), Mock()
        evader.reset.side_effect = lambda: events.append("evader.reset")
        pursuer.reset.side_effect = lambda: events.append("pursuer.reset")

        def action(label, expected, received):
            self.assertIs(received, expected)
            events.append(label)
            return PEAction(controlled.pe_grid.NORTH)

        evader.step.side_effect = lambda received: action("evader.step", obs[0], received)
        pursuer.step.side_effect = lambda received: action("pursuer.step", obs[1], received)
        real_step = self.env.step

        def step(*args):
            events.append("env.step")
            return real_step(*args)

        with patch.object(self.env, "reset", side_effect=AssertionError("reset called")), \
             patch.object(self.env, "step", side_effect=step) as step_mock, \
             patch.object(random, "seed", wraps=random.seed) as python_seed, \
             patch.object(np.random, "seed", wraps=np.random.seed) as numpy_seed:
            _, steps, final, done, samples = controlled.run_controlled_episode(
                self.env, evader, pursuer, state, obs, 7, 1, distances
            )
        python_seed.assert_called_once_with(7)
        numpy_seed.assert_called_once_with(7)
        step_mock.assert_called_once()
        self.assertEqual(events, ["evader.reset", "pursuer.reset", "evader.step",
                                  "pursuer.step", "env.step"])
        self.assertEqual(steps, 1)
        self.assertFalse(done)
        self.assertEqual([s["timestep"] for s in samples], [0, 1])
        self.assertEqual(samples[0]["runner_loc"], 56)
        self.assertEqual(samples[-1]["runner_loc"], final.runner_loc)

    def test_real_intmcp_run_is_reproducible_and_csvs_are_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            prefixes = [Path(directory) / name for name in ("first", "second")]
            rng_states = []
            for prefix in prefixes:
                args = controlled.parse_args([
                    "--grid_name", "maze8", "--runner_start", "56",
                    "--runner_goal", "3", "--chaser_start", "26",
                    "--pursuer_level", "1", "--evader_level", "0",
                    "--num_sims", "32", "--seed", "0", "--output_prefix", str(prefix),
                ])
                with patch.object(controlled.PEModel, "reset",
                                  side_effect=AssertionError("reset called")), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(random, "seed", wraps=random.seed) as python_seed, \
                     patch.object(np.random, "seed", wraps=np.random.seed) as numpy_seed:
                    controlled.run(args)
                self.assertEqual(python_seed.call_count, 2)
                self.assertEqual(numpy_seed.call_count, 2)
                rng_states.append((random.getstate(), np.random.get_state()))
            self.assertEqual(rng_states[0][0], rng_states[1][0])
            np.testing.assert_array_equal(rng_states[0][1][1], rng_states[1][1][1])
            self.assertEqual(rng_states[0][1][2:], rng_states[1][1][2:])
            tables = {}
            for suffix in ("summary", "episodes", "timesteps"):
                first, second = [Path(str(p) + "_" + suffix + ".csv") for p in prefixes]
                self.assertEqual(first.read_bytes(), second.read_bytes())
                with first.open(newline="", encoding="utf-8") as source:
                    tables[suffix] = list(csv.DictReader(source))
                for row in tables[suffix]:
                    for field, value in dict(grid="maze8", runner_start="56", runner_goal="3",
                                            chaser_start="26", episode_seed="0", pursuer_level="1",
                                            evader_level="0", num_sims="32").items():
                        self.assertEqual(row[field], value)
            self.assertEqual(len(tables["summary"]), 1)
            self.assertEqual(len(tables["episodes"]), 1)
            episode = tables["episodes"][0]
            samples = tables["timesteps"]
            self.assertEqual([int(s["timestep"]) for s in samples],
                             list(range(int(episode["episode_steps"]) + 1)))
            self.assertEqual(samples[0]["runner_loc"], "56")
            self.assertEqual(samples[0]["chaser_loc"], "26")
            self.assertEqual(samples[-1]["runner_loc"], episode["final_runner_loc"])
            self.assertEqual(samples[-1]["chaser_loc"], episode["final_chaser_loc"])
            self.assertEqual(samples[-1]["pursuer_seen"], "1")
            self.assertEqual(episode["runner_seen_timesteps"], "1")
            for distance in ("manhattan_distance", "shortest_path_distance"):
                values = [float(s[distance]) for s in samples]
                self.assertEqual(float(episode["initial_" + distance]), values[0])
                self.assertEqual(float(episode["final_" + distance]), values[-1])
                self.assertEqual(float(episode["min_" + distance]), min(values))
                self.assertEqual(float(episode["mean_" + distance]), sum(values) / len(values))


if __name__ == "__main__":
    unittest.main()
