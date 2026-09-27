"""Analysis tests use recorded/synthetic rows only, never PE environments."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from analyse_controlled_trajectories import (
    Episode, classify_ending, load_episode, load_episodes,
    sequence_distribution, summarize,
)


class ControlledTrajectoryAnalysisTest(unittest.TestCase):
    def test_outcome_precedence_and_nonterminal_end(self):
        self.assertEqual(classify_ending(3, 3, 3, True), ("WIN", "co_location"))
        self.assertEqual(classify_ending(3, 4, 3, True), ("LOSS", "runner_goal"))
        self.assertEqual(classify_ending(5, 4, 3, True), ("WIN", "runner_seen"))
        self.assertEqual(classify_ending(5, 4, 3, False), ("DRAW", "nonterminal_recorded_end"))

    def test_short_episodes_are_not_padded_and_denominators_are_explicit(self):
        episodes = [
            Episode(29, 1, 0, "a.csv", 128, 0,
                    (61, 62, 54, 46, 38), (36, 44, 45, 46, 38),
                    (4, 3, 2, 1, 0), (True, True, True, True, True),
                    (False, False, False, False, True)),
            Episode(29, 1, 1, "b.csv", 128, 0,
                    (61, 62, 54, 46), (36, 44, 45, 45),
                    (4, 5, 3, 1), (False, False, True, True),
                    (False, False, False, True)),
            Episode(29, 1, 2, "c.csv", 128, 0,
                    (61, 62), (36, 36), (4, 4), (False, False), (False, False)),
        ]
        summary, positions = summarize(episodes)
        self.assertEqual(summary["complete_first_five_episode_count"], 1)
        self.assertEqual(summary["shorter_episode_count"], 2)
        self.assertEqual(summary["mean_shortest_path_distance_t1"], 4)
        self.assertEqual(summary["distance_sample_count_t3"], 2)
        self.assertEqual(summary["distance_sample_count_t4"], 1)
        self.assertEqual(summary["mean_shortest_path_distance_t4"], 0)
        self.assertEqual(summary["mean_shortest_path_distance_t5"], "")
        self.assertEqual(summary["distance_sample_count_t5"], 0)
        for change in ("closer", "farther", "same_distance"):
            self.assertEqual(summary["first_transition_" + change + "_fraction"], 1 / 3)
        self.assertEqual(summary["mean_first_heard_timestep_when_heard"], 1)
        self.assertEqual(summary["never_heard_episode_count"], 1)
        details = json.loads(summary["episode_details"])
        self.assertEqual([d["first_heard_timestep"] for d in details], [0, 2, None])
        self.assertEqual([d["outcome"] for d in details], ["WIN", "WIN", "DRAW"])
        self.assertEqual(summary["mean_episode_steps"], 8 / 3)
        at_three = [row for row in positions if row["timestep"] == 3]
        self.assertEqual(len(at_three), 2)
        self.assertTrue(all(row["fraction_at_timestep"] == 0.5 for row in at_three))
        self.assertTrue(all(row["fraction_all_episodes"] == 1 / 3 for row in at_three))
        self.assertEqual(len(json.loads(summary["complete_chaser_first_five_sequences"])), 1)
        self.assertEqual(len(json.loads(summary["shorter_chaser_sequences"])), 2)

    def test_sequence_ranking_and_ties_are_stable(self):
        ranked = sequence_distribution([(36, 44), (36, 37), (36, 44), (36, 36)])
        self.assertEqual([r["locations"] for r in ranked], [[36, 44], [36, 36], [36, 37]])
        self.assertEqual([r["fraction"] for r in ranked], [0.5, 0.25, 0.25])
        self.assertEqual(sequence_distribution([]), [])

    def test_missing_files_are_not_silently_pooled(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing 240"):
                load_episodes(Path(directory))

    def test_loader_rejects_gaps_and_wrong_metadata(self):
        common = dict(grid="maze8", runner_start=61, runner_goal=3, chaser_start=36,
                      episode_seed=0, seed=0, episode=0, pursuer_level=1,
                      evader_level=0, pursuer_policy="I-NTMCP l=1", evader_policy="I-NTMCP l=0",
                      num_sims=128, pursuer_heard=0, pursuer_seen=0)
        rows = [dict(common, timestep=0, runner_loc=61, chaser_loc=36, shortest_path_distance=4),
                dict(common, timestep=1, runner_loc=62, chaser_loc=44, shortest_path_distance=4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.csv"

            def save():
                with path.open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

            save()
            self.assertEqual(load_episode(path, 29, 1, 0).steps, 1)
            rows[1]["timestep"] = 2
            save()
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_episode(path, 29, 1, 0)
            rows[1]["timestep"] = 1
            rows[1]["runner_goal"] = 5
            save()
            with self.assertRaisesRegex(ValueError, "metadata"):
                load_episode(path, 29, 1, 0)


if __name__ == "__main__":
    unittest.main()
