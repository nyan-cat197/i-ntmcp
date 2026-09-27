#!/usr/bin/env python3
"""Analyse existing maze8 config 28-31, seed 0-29, pursuer l=0/1 CSVs.

Reads only controlled_maze8_cfg{config}_p{level}_seed{seed}_timesteps.csv
from --results_dir. Never imports an environment or runs episodes. All 240
files are required, with consistent planning settings and contiguous t=0..T.

The first five timesteps mean t=0..4, including initialization. Complete
five-sample prefixes and shorter episode sequences are counted separately;
ended episodes are never padded. Position probabilities and mean distances use
only episodes recorded at that timestep, with explicit denominators. First-step
distance change is d(t=1)-d(t=0) after BOTH agents move, not a causal estimate of
the chaser's movement. Location sequences are not converted into actions.

Outcomes follow PEModel.get_outcome precedence using the final CSV row:
co-location -> WIN; runner at goal -> LOSS; pursuer_seen -> WIN; otherwise DRAW.
A nonterminal final row is flagged as 'nonterminal_recorded_end': timestep CSVs
alone cannot prove it is a timeout rather than a truncated recording.

Writes runner61_trajectory_summary.csv (8 grouped rows, including JSON sequence
distributions and per-episode trajectories/outcomes/hearing times) and
runner61_position_distributions.csv (chaser positions at t=1,2,3). JSON lists
are ordered by descending count, then location sequence for stable ties.
"""

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


# Verified configuration IDs from the existing maze8 controlled configuration
# mapping; metadata in every input row must agree with these triples.
CONFIGS = {28: (61, 3, 26), 29: (61, 3, 36),
           30: (61, 5, 26), 31: (61, 5, 36)}
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results"
POSITION_FIELDS = [
    "config_id", "grid", "runner_start", "runner_goal", "chaser_start",
    "pursuer_level", "evader_level", "num_sims", "timestep", "chaser_loc",
    "count", "episodes_at_timestep", "episodes_total", "fraction_at_timestep",
    "fraction_all_episodes",
]


@dataclass(frozen=True)
class Episode:
    config_id: int
    level: int
    seed: int
    source: str
    num_sims: int
    evader_level: int
    runner: Tuple[int, ...]
    chaser: Tuple[int, ...]
    distances: Tuple[float, ...]
    heard: Tuple[bool, ...]
    seen: Tuple[bool, ...]

    @property
    def steps(self) -> int:
        return len(self.runner) - 1

    @property
    def first_heard(self) -> Optional[int]:
        return next((t for t, heard in enumerate(self.heard) if heard), None)

    @property
    def ending(self) -> Tuple[str, str]:
        return classify_ending(self.runner[-1], self.chaser[-1],
                               CONFIGS[self.config_id][1], self.seen[-1])


def classify_ending(runner: int, chaser: int, goal: int, seen: bool) -> Tuple[str, str]:
    """Match PE outcome precedence, without inferring hidden directions/actions."""
    if runner == chaser:
        return "WIN", "co_location"
    if runner == goal:
        return "LOSS", "runner_goal"
    if seen:
        return "WIN", "runner_seen"
    return "DRAW", "nonterminal_recorded_end"


def parse_flag(value: str) -> bool:
    if value.lower() in ("1", "true"):
        return True
    if value.lower() in ("0", "false"):
        return False
    raise ValueError("Invalid observation flag: {!r}".format(value))


def load_episode(path: Path, config: int, level: int, seed: int) -> Episode:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {
            "grid", "runner_start", "runner_goal", "chaser_start",
            "episode_seed", "seed", "episode", "pursuer_level", "evader_level",
            "pursuer_policy", "evader_policy", "num_sims", "timestep",
            "runner_loc", "chaser_loc", "shortest_path_distance",
            "pursuer_heard", "pursuer_seen",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("{}: missing columns {}".format(path, sorted(missing)))
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("{}: expected an initial row and at least one transition".format(path))
    rows.sort(key=lambda row: int(row["timestep"]))
    if [int(row["timestep"]) for row in rows] != list(range(len(rows))):
        raise ValueError("{}: timesteps must be unique and contiguous from zero".format(path))
    runner_start, goal, chaser_start = CONFIGS[config]
    sims, evader_level = int(rows[0]["num_sims"]), int(rows[0]["evader_level"])
    if sims <= 0 or evader_level < 0:
        raise ValueError("{}: invalid planning settings".format(path))
    expected = {
        "runner_start": runner_start, "runner_goal": goal, "chaser_start": chaser_start,
        "pursuer_level": level, "seed": seed, "episode_seed": seed,
        "num_sims": sims, "evader_level": evader_level, "episode": 0,
    }
    for row in rows:
        if (row["grid"] != "maze8"
                or row["pursuer_policy"] != "I-NTMCP l={}".format(level)
                or row["evader_policy"] != "I-NTMCP l={}".format(evader_level)
                or any(int(row[key]) != value for key, value in expected.items())):
            raise ValueError("{}: inconsistent episode/configuration metadata".format(path))
    episode = Episode(
        config, level, seed, path.name, sims, evader_level,
        tuple(int(row["runner_loc"]) for row in rows),
        tuple(int(row["chaser_loc"]) for row in rows),
        tuple(float(row["shortest_path_distance"]) for row in rows),
        tuple(parse_flag(row["pursuer_heard"]) for row in rows),
        tuple(parse_flag(row["pursuer_seen"]) for row in rows),
    )
    if episode.runner[0] != runner_start or episode.chaser[0] != chaser_start:
        raise ValueError("{}: initial locations disagree with configuration".format(path))
    if any(not math.isfinite(d) or d < 0 for d in episode.distances):
        raise ValueError("{}: expected finite nonnegative shortest-path distances".format(path))
    for t in range(episode.steps):
        if classify_ending(episode.runner[t], episode.chaser[t], goal, episode.seen[t])[0] != "DRAW":
            raise ValueError("{}: samples continue after a terminal state".format(path))
    return episode


def load_episodes(directory: Path) -> Dict[Tuple[int, int], List[Episode]]:
    paths = [(config, level, seed, directory / (
        "controlled_maze8_cfg{}_p{}_seed{}_timesteps.csv".format(config, level, seed)
    )) for config in CONFIGS for level in (0, 1) for seed in range(30)]
    missing = [path.name for _, _, _, path in paths if not path.is_file()]
    if missing:
        raise ValueError("Missing {} required timestep CSVs: {}".format(
            len(missing), ", ".join(missing)
        ))
    groups = {(config, level): [] for config in CONFIGS for level in (0, 1)}
    for config, level, seed, path in paths:
        groups[config, level].append(load_episode(path, config, level, seed))
    settings = {(e.num_sims, e.evader_level) for group in groups.values() for e in group}
    if len(settings) != 1:
        raise ValueError("Cannot pool differing (num_sims, evader_level) settings: {}".format(settings))
    return groups


def sequence_distribution(sequences: List[Tuple[int, ...]]) -> List[Dict]:
    return [{"locations": list(sequence), "count": count,
             "fraction": count / len(sequences)}
            for sequence, count in sorted(Counter(sequences).items(),
                                          key=lambda item: (-item[1], item[0]))]


def json_value(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def summarize(episodes: List[Episode]) -> Tuple[Dict, List[Dict]]:
    first = episodes[0]
    runner_start, goal, chaser_start = CONFIGS[first.config_id]
    n = len(episodes)
    complete = [e for e in episodes if e.steps >= 4]
    short = [e for e in episodes if e.steps < 4]
    heard_times = [e.first_heard for e in episodes if e.first_heard is not None]
    outcomes = Counter(e.ending[0] for e in episodes)
    closer = sum(e.distances[1] < e.distances[0] for e in episodes)
    farther = sum(e.distances[1] > e.distances[0] for e in episodes)
    same = n - closer - farther
    metadata = {
        "config_id": first.config_id, "grid": "maze8", "runner_start": runner_start,
        "runner_goal": goal, "chaser_start": chaser_start, "pursuer_level": first.level,
        "evader_level": first.evader_level, "num_sims": first.num_sims,
    }
    row = {
        **metadata, "episodes": n, "seeds": json_value([e.seed for e in episodes]),
        "wins": outcomes["WIN"], "losses": outcomes["LOSS"], "draws": outcomes["DRAW"],
        "win_fraction": outcomes["WIN"] / n,
        "mean_episode_steps": mean(e.steps for e in episodes),
        "first_transition_closer_count": closer,
        "first_transition_farther_count": farther,
        "first_transition_same_distance_count": same,
        "first_transition_closer_fraction": closer / n,
        "first_transition_farther_fraction": farther / n,
        "first_transition_same_distance_fraction": same / n,
        "first_chaser_location_transitions": json_value(sequence_distribution([e.chaser[:2] for e in episodes])),
        "heard_episode_count": len(heard_times), "never_heard_episode_count": n - len(heard_times),
        "heard_episode_fraction": len(heard_times) / n,
        "mean_first_heard_timestep_when_heard": mean(heard_times) if heard_times else "",
        "first_heard_timestep_distribution": json_value([
            {"timestep": t, "count": count, "fraction_all_episodes": count / n}
            for t, count in sorted(Counter(heard_times).items())
        ]),
        "first_five_timesteps": "0,1,2,3,4",
        "complete_first_five_episode_count": len(complete),
        "shorter_episode_count": len(short),
        "complete_chaser_first_five_sequences": json_value(sequence_distribution([e.chaser[:5] for e in complete])),
        "complete_runner_first_five_sequences": json_value(sequence_distribution([e.runner[:5] for e in complete])),
        "shorter_chaser_sequences": json_value(sequence_distribution([e.chaser for e in short])),
        "shorter_runner_sequences": json_value(sequence_distribution([e.runner for e in short])),
    }
    for t in range(11):
        available = [e.distances[t] for e in episodes if e.steps >= t]
        row["distance_sample_count_t{}".format(t)] = len(available)
        row["mean_shortest_path_distance_t{}".format(t)] = mean(available) if available else ""
    # Keep every reconstructed trajectory and episode result auditable in the
    # requested grouped summary without creating an additional output file.
    row["episode_details"] = json_value([
        {"seed": e.seed, "source": e.source, "episode_steps": e.steps,
         "outcome": e.ending[0], "ending_evidence": e.ending[1],
         "first_heard_timestep": e.first_heard,
         "chaser_trajectory": list(e.chaser), "runner_trajectory": list(e.runner)}
        for e in episodes
    ])
    positions = []
    for t in (1, 2, 3):
        counts = Counter(e.chaser[t] for e in episodes if e.steps >= t)
        available = sum(counts.values())
        for loc, count in sorted(counts.items()) if counts else [("", 0)]:
            positions.append({
                **metadata, "timestep": t, "chaser_loc": loc, "count": count,
                "episodes_at_timestep": available, "episodes_total": n,
                "fraction_at_timestep": count / available if available else "",
                "fraction_all_episodes": count / n,
            })
    return row, positions


def print_comparison(summaries: List[Dict], positions: List[Dict]) -> None:
    print("First five samples = t=0..4; shorter episodes are separate, never padded.")
    print("Distance change is net separation after both agents move; no actions inferred.")
    print("Later means use only episodes observed at that timestep (see CSV sample counts).")
    for config in (29, 31):
        group = [row for row in summaries if row["config_id"] == config]
        print("\nConfig {}: runner {} -> goal {}, chaser {} ({} episodes/level)".format(
            config, *CONFIGS[config], group[0]["episodes"]
        ))
        print("Level  Wins  Mean steps  Closer/Same/Farther t1   First heard mean (n)  Full/short t0..4")
        for row in group:
            first_heard = row["mean_first_heard_timestep_when_heard"]
            heard_text = "{:.2f}".format(first_heard) if first_heard != "" else "n/a"
            print("l={}    {:2}/{}     {:5.2f}     {:5.1%}/{:5.1%}/{:5.1%}        {} ({})          {}/{}".format(
                row["pursuer_level"], row["wins"], row["episodes"], row["mean_episode_steps"],
                row["first_transition_closer_fraction"], row["first_transition_same_distance_fraction"],
                row["first_transition_farther_fraction"], heard_text, row["heard_episode_count"],
                row["complete_first_five_episode_count"], row["shorter_episode_count"],
            ))
        for t in (1, 2, 3):
            descriptions = []
            for level in (0, 1):
                counts = [r for r in positions if r["config_id"] == config
                          and r["pursuer_level"] == level and r["timestep"] == t]
                descriptions.append("l={}: {}".format(level, ", ".join(
                    "{}:{}".format(r["chaser_loc"], r["count"]) for r in counts
                )))
            print("  t={} chaser location:count | {}".format(t, " | ".join(descriptions)))
        for row in group:
            for label, field in (("complete chaser t0..4", "complete_chaser_first_five_sequences"),
                                 ("complete runner t0..4", "complete_runner_first_five_sequences"),
                                 ("shorter chaser", "shorter_chaser_sequences"),
                                 ("shorter runner", "shorter_runner_sequences")):
                sequences = json.loads(row[field])[:2]
                if sequences:
                    text = "; ".join("{} (n={})".format(
                        "->".join(map(str, s["locations"])), s["count"]
                    ) for s in sequences)
                    print("  l={} top {}: {}".format(row["pursuer_level"], label, text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS,
                        help="Directory containing timestep CSVs and receiving the two reports.")
    args = parser.parse_args()
    try:
        groups = load_episodes(args.results_dir)
        summaries, positions = [], []
        for episodes in groups.values():
            summary, distributions = summarize(episodes)
            summaries.append(summary)
            positions.extend(distributions)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    for filename, fields, rows in (
        ("runner61_trajectory_summary.csv", list(summaries[0]), summaries),
        ("runner61_position_distributions.csv", POSITION_FIELDS, positions),
    ):
        path = args.results_dir / filename
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print("Saved {}".format(path))
    print_comparison(summaries, positions)


if __name__ == "__main__":
    main()
