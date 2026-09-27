"""Regressions for PE policy state, simulation updates, and episode evaluation."""
import contextlib
import csv
import io
import logging
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import intmcp.model as M
from intmcp.envs.pe import PEModel, PESPPolicy
from intmcp.envs.pe import grid
from intmcp.envs.pe.action import PEAction
from intmcp.envs.pe.shortest_path import all_shortest_paths
from intmcp.envs.pe.state import PEState
from intmcp.run import runner, stats
from intmcp.tree.history_state import HistoryState
from intmcp.tree.node import Node
import run_pe_controlled as controlled
import run_pe_pair as pair
import run_pe_table as table


def trees(tree):
    yield tree
    for child in tree.nested_trees.values():
        yield from trees(child)


class PECorrectnessTest(unittest.TestCase):
    def setUp(self):
        self.python_rng = random.getstate()
        self.numpy_rng = np.random.get_state()
        random.seed(0)
        self.logger = logging.getLogger(self.id())
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False
        self.args = table.parse_args(["--num_sims", "8", "--step_limit", "6"])
        self.env = PEModel("open8")
        self.state, self.obs = controlled.controlled_initial_state(
            self.env, 56, 3, 27
        )

    def tearDown(self):
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        self.logger.handlers.clear()

    def make_tree(self, agent, level, env=None):
        return table.make_policy(
            "I-NTMCP l={}".format(level), env or self.env,
            agent, self.args, self.logger,
        )

    def make_sp(self, agent):
        return PESPPolicy(self.env, agent, .95, r_hi=90, r_lo=-138)

    def snapshot(self, policy):
        return (policy.history, policy._loc, policy._prev_loc,
                policy._update_num, policy._last_action)

    def test_reset_clears_episode_state_for_both_agents(self):
        for agent in (0, 1):
            with self.subTest(agent=agent):
                policy = self.make_sp(agent)
                policy.step(self.obs[agent])
                policy.step(self.obs[agent])
                self.env._runner_start_loc = 63
                self.env._chaser_start_loc = 36
                policy.reset()
                self.assertEqual(policy._loc, (63, 36)[agent])
                self.assertEqual(policy._prev_loc, -1)
                self.assertEqual(policy._update_num, 0)
                self.assertEqual(policy.history, M.AgentHistory.get_init_history())
                self.assertIsInstance(policy._last_action, M.NullAction)
                # The initial observation must not execute an old action.
                policy.update(policy._last_action, self.obs[agent])
                self.assertEqual(policy._loc, (63, 36)[agent])
                self.assertEqual(policy._prev_loc, -1)

    def test_reset_history_reconstructs_state_independently_of_prior_use(self):
        for agent in (0, 1):
            with self.subTest(agent=agent):
                policy = self.make_sp(agent)
                initial = M.AgentHistory.get_init_history(self.obs[agent])
                history = initial.extend(PEAction(grid.EAST), self.obs[agent])
                history = history.extend(PEAction(grid.NORTH), self.obs[agent])
                for target in (M.AgentHistory.get_init_history(), initial, history):
                    policy.reset_history(target)
                    expected = self.snapshot(policy)
                    for direction in (grid.WEST, grid.SOUTH, grid.EAST):
                        policy.update(PEAction(direction), self.obs[agent])
                    policy.get_action()
                    policy.reset_history(target)
                    self.assertEqual(self.snapshot(policy), expected)
                    policy.reset_history(target)
                    self.assertEqual(self.snapshot(policy), expected)
                self.assertEqual((policy._loc, policy._prev_loc),
                                 ((49, 57), (20, 28))[agent])
                self.assertEqual(policy._update_num, 3)
                policy.update(PEAction(grid.EAST), self.obs[agent])
                self.assertEqual((policy._loc, policy._prev_loc),
                                 ((50, 49), (21, 20))[agent])
                self.assertEqual(policy._update_num, 4)

    def test_history_reconstruction_preserves_blocked_moves(self):
        policy = self.make_sp(0)
        history = M.AgentHistory.get_init_history(self.obs[0]).extend(
            PEAction(grid.WEST), self.obs[0]
        )
        policy.reset_history(history)
        self.assertEqual((policy._loc, policy._prev_loc), (56, 56))
        self.assertEqual(policy._update_num, 2)

    def test_rollout_actions_use_assigned_history_across_simulations_and_episodes(self):
        # Includes the fresh controlled-state case and every requested level.
        action_calls = []
        original = PESPPolicy.get_action

        def checked_action(policy):
            self.assertEqual((policy._loc, policy._prev_loc),
                             policy._get_loc_from_history(policy.history))
            self.assertEqual(policy._update_num, len(policy.history.history))
            action_calls.append(policy.ego_agent)
            return original(policy)

        env = PEModel("maze8")
        for agent, level in ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)):
            with self.subTest(agent=agent, level=level):
                _, obs = controlled.controlled_initial_state(env, 56, 3, 26)
                tree = self.make_tree(agent, level, env)
                rollout_ids = [id(t._rollout_policy) for t in trees(tree)]
                before_calls = len(action_calls)
                with patch.object(PESPPolicy, "get_action", checked_action):
                    tree.reset()
                    tree.step(obs[agent])
                    # A different real episode, reusing every rollout object.
                    _, obs = controlled.controlled_initial_state(env, 59, 0, 36)
                    tree.reset()
                    for t in trees(tree):
                        p = t._rollout_policy
                        if isinstance(p, PESPPolicy):
                            self.assertEqual(p._loc, env.runner_start_loc)
                            self.assertEqual(p._prev_loc, -1)
                            self.assertEqual(p._update_num, 0)
                            self.assertEqual(p.history, M.AgentHistory.get_init_history())
                    tree.step(obs[agent])
                self.assertEqual(rollout_ids, [id(t._rollout_policy) for t in trees(tree)])
                if (agent, level) != (1, 0):
                    self.assertGreater(len(action_calls), before_calls)

    def test_simulate_advances_each_participating_rollout_once(self):
        joint_action = M.JointAction((PEAction(grid.NORTH), PEAction(grid.NORTH)))
        next_state, next_obs, _, done = self.env.step(self.state, joint_action)
        self.assertFalse(done)
        for agent in (0, 1):
            for level in (0, 1, 2, 3):
                with self.subTest(agent=agent, level=level):
                    tree = self.make_tree(agent, level)
                    history = M.JointHistory.get_init_history(2, self.obs)
                    node = Node(None, None, M.ParticleBelief())
                    tree.expand(node, history.get_agent_history(agent))
                    tree._depth_limit = 0  # Exactly one nonterminal transition.
                    all_trees = list(trees(tree))
                    for t in all_trees:
                        t._rollout_policy.reset_history(history.get_agent_history(t.ego_agent))
                    with contextlib.ExitStack() as stack:
                        updates = [stack.enter_context(patch.object(
                            t._rollout_policy, "update", wraps=t._rollout_policy.update
                        )) for t in all_trees]
                        stack.enter_context(patch.object(tree, "get_joint_sim_action",
                                                       return_value=joint_action))
                        step = stack.enter_context(patch.object(self.env, "step", wraps=self.env.step))
                        tree.simulate(HistoryState(self.state, history), node, 0)
                    step.assert_called_once_with(self.state, joint_action)
                    for index, (t, update) in enumerate(zip(all_trees, updates)):
                        if index < (2 if level else 1):
                            update.assert_called_once_with(joint_action[t.ego_agent], next_obs[t.ego_agent])
                            self.assertEqual(t._rollout_policy.history,
                                             history.get_agent_history(t.ego_agent).extend(
                                                 joint_action[t.ego_agent], next_obs[t.ego_agent]))
                        else:
                            update.assert_not_called()

    def terminal_reset(self):
        env = PEModel("open8")
        state = PEState(3, 27, grid.NORTH, grid.NORTH, 56, env.grid)
        self.assertTrue(env.is_terminal(state))
        return env, state, env.sample_initial_obs(state)

    def test_terminal_reset_uses_outcome_without_actions_or_transitions(self):
        for evaluator in ("pair", "table"):
            # The mock outcome also proves the evaluator delegates to the API.
            for expected in (M.Outcomes.WIN, M.Outcomes.LOSS):
                with self.subTest(evaluator=evaluator, expected=expected):
                    env, state, obs = self.terminal_reset()
                    policies = [Mock(), Mock()]
                    with patch.object(env, "reset", return_value=(state, obs)), \
                         patch.object(env, "step", side_effect=AssertionError("transition")), \
                         patch.object(env, "get_outcome", return_value=[M.Outcomes.DRAW, expected]) as outcome:
                        if evaluator == "pair":
                            result, steps, final, done, samples = pair.run_episode_with_final_state(
                                env, *policies, 40, all_shortest_paths(sorted(env.grid.valid_locs), env.grid))
                            self.assertIs(final, state)
                            self.assertTrue(done)
                            self.assertEqual([s["timestep"] for s in samples], [0])
                            self.assertEqual(samples[0]["pursuer_seen"], 1)
                            self.assertEqual(pair.get_episode_diagnostics(env, result, final, done)["timeout"], 0)
                        else:
                            result, steps = table.run_episode(env, *policies, 40)
                    self.assertEqual((result, steps), (expected, 0))
                    outcome.assert_called_once_with(state)
                    for p in policies:
                        p.step.assert_not_called()

    def test_generic_evaluator_finalizes_zero_step_episode(self):
        env, state, obs = self.terminal_reset()
        policies = [self.make_tree(i, 0, env) for i in (0, 1)]
        trackers = list(stats.get_default_trackers(.95, policies)) + [
            stats.PolicyKLTracker(2), stats.NestedBeliefEntropyTracker(2, policies)
        ]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(env, "reset", return_value=(state, obs)))
            stack.enter_context(patch.object(env, "step", side_effect=AssertionError("transition")))
            outcome = stack.enter_context(patch.object(env, "get_outcome", wraps=env.get_outcome))
            for p in policies:
                stack.enter_context(patch.object(p, "step", side_effect=AssertionError("action")))
                stack.enter_context(patch.object(p, "get_pi_by_history", side_effect=AssertionError("policy query")))
            result = runner.run_sims(env, policies, trackers, [], 2, 40,
                                     logger=self.logger, render_asci=False)
        self.assertEqual(outcome.call_count, 2)
        self.assertEqual(result[1]["num_episodes"], 2)
        self.assertEqual(result[1]["num_outcome_WIN"], 2)
        self.assertEqual(result[1]["episode_steps_mean"], 0)
        self.assertEqual(result[1]["episode_returns_mean"], 0)
        self.assertEqual(result[1]["episode_dones"], 1)
        for tracker in trackers:
            self.assertEqual(tracker._steps, [0, 0])

    def test_nonterminal_generic_evaluator_counts_actual_transition(self):
        policies = [table.make_policy("random", self.env, i, self.args, self.logger) for i in (0, 1)]
        trackers = stats.get_default_trackers(.95, policies)
        with patch.object(self.env, "reset", return_value=(self.state, self.obs)), \
             patch.object(self.env, "step", wraps=self.env.step) as step:
            result = runner.run_sims(self.env, policies, trackers, [], 1, 1,
                                     logger=self.logger, render_asci=False)
        self.assertEqual(step.call_count, 1)
        self.assertEqual(result[1]["episode_steps_mean"], 1)

    def test_fixed_seed_csvs_are_reproducible_and_include_grid_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for grid_name in ("8by8", "maze8", "open8"):
                for evaluator in ("pair", "table"):
                    with self.subTest(grid=grid_name, evaluator=evaluator):
                        runs = []
                        for repeat in range(2):
                            prefix = root / "{}_{}_{}".format(grid_name, evaluator, repeat)
                            common = ["--grid_name", grid_name, "--num_sims", "4",
                                      "--step_limit", "4", "--seed", "0"]
                            if evaluator == "pair":
                                args = pair.parse_args(common + ["--pursuer", "intmcp", "--pursuer_level", "1",
                                    "--evader", "mixed", "--num_episodes", "12", "--output_prefix", str(prefix)])
                                run = pair.run
                                paths = [Path(str(prefix) + "_" + suffix + ".csv")
                                         for suffix in ("summary", "episodes", "timesteps")]
                            else:
                                paths = [Path(str(prefix) + "_" + suffix + ".csv") for suffix in ("summary", "episodes")]
                                args = table.parse_args(common + ["--mixed_only", "--num_episodes", "4",
                                    "--output", str(paths[0]), "--episode_output", str(paths[1]), "--log_dir", str(prefix) + "_logs"])
                                run = table.run
                            with contextlib.redirect_stdout(io.StringIO()):
                                run(args)
                            runs.append([p.read_bytes() for p in paths])
                            for path in paths:
                                with path.open(newline="", encoding="utf-8") as source:
                                    rows = list(csv.DictReader(source))
                                self.assertTrue(rows)
                                self.assertTrue(all(r["grid_name"] == grid_name for r in rows))
                                if "grid" in rows[0]:
                                    self.assertTrue(all(r["grid"] == grid_name for r in rows))
                        self.assertEqual(runs[0], runs[1])


if __name__ == "__main__":
    unittest.main()
