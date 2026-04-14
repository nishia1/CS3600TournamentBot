from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from collections import Counter, defaultdict
from itertools import combinations, permutations
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_ROOT / "3600-agents"
MATCHES_DIR = AGENTS_DIR / "matches"
RUNNER = PROJECT_ROOT / "engine" / "run_local_agents.py"
RL_STATE_PATH = AGENTS_DIR / "Cassie" / "cassie_rl_state.json"


def load_match(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def match_margin(match: dict[str, Any]) -> int:
    return int(match.get("a_points", [0])[-1]) - int(match.get("b_points", [0])[-1])


def winner_label(match: dict[str, Any]) -> str:
    a_points = match.get("a_points", [0])
    b_points = match.get("b_points", [0])
    final_a = int(a_points[-1]) if a_points else 0
    final_b = int(b_points[-1]) if b_points else 0
    if final_a > final_b:
        return "A"
    if final_b > final_a:
        return "B"
    return "DRAW"


def first_lead_turn(match: dict[str, Any], winner: str) -> tuple[int | None, str | None]:
    a_points = match.get("a_points", [])
    b_points = match.get("b_points", [])
    length = max(len(a_points), len(b_points))

    for idx in range(length):
        a = a_points[idx] if idx < len(a_points) else a_points[-1] if a_points else 0
        b = b_points[idx] if idx < len(b_points) else b_points[-1] if b_points else 0

        if winner == "A" and a > b:
            return idx + 1, "A"
        if winner == "B" and b > a:
            return idx + 1, "B"

    return None, None


def event_turns(match: dict[str, Any]) -> dict[str, list[int]]:
    events: dict[str, list[int]] = defaultdict(list)

    a_points = match.get("a_points", [])
    b_points = match.get("b_points", [])
    rat_caught = match.get("rat_caught", [])
    new_carpets = match.get("new_carpets", [])
    left_behind = match.get("left_behind", [])

    length = max(len(a_points), len(b_points), len(rat_caught), len(new_carpets), len(left_behind))

    prev_a = a_points[0] if a_points else 0
    prev_b = b_points[0] if b_points else 0

    for idx in range(length):
        turn = idx + 1

        if idx < len(a_points):
            current_a = a_points[idx]
            if idx > 0 and current_a != prev_a:
                events[f"a_point_gain_{current_a - prev_a:+d}"].append(turn)
            prev_a = current_a

        if idx < len(b_points):
            current_b = b_points[idx]
            if idx > 0 and current_b != prev_b:
                events[f"b_point_gain_{current_b - prev_b:+d}"].append(turn)
            prev_b = current_b

        if idx < len(rat_caught) and rat_caught[idx]:
            events["rat_caught"].append(turn)

        if idx < len(new_carpets) and new_carpets[idx]:
            events["new_carpet"].append(turn)

        if idx < len(left_behind) and left_behind[idx] != "plain":
            events[f"left_behind_{left_behind[idx]}"] .append(turn)

    return events


def summarize_match(path: pathlib.Path) -> dict[str, Any]:
    match = load_match(path)
    a_name = path.stem.split("_")[0]
    b_name = path.stem.split("_")[1] if "_" in path.stem else "?"
    a_points = match.get("a_points", [0])
    b_points = match.get("b_points", [0])
    turn_count = int(match.get("turn_count", max(len(a_points), len(b_points))))
    margin = match_margin(match)
    events = event_turns(match)
    lead_turn, leader = first_lead_turn(match, winner_label(match))

    first_scoring_turn = None
    for idx, (a, b) in enumerate(zip(a_points, b_points), start=1):
        if idx == 1:
            continue
        if a != a_points[idx - 2] or b != b_points[idx - 2]:
            first_scoring_turn = idx
            break

    return {
        "file": path.name,
        "a": a_name,
        "b": b_name,
        "winner": winner_label(match),
        "result": int(match.get("result", 0)),
        "reason": match.get("reason", ""),
        "turn_count": turn_count,
        "final_a": int(a_points[-1]) if a_points else 0,
        "final_b": int(b_points[-1]) if b_points else 0,
        "margin": margin,
        "first_scoring_turn": first_scoring_turn,
        "lead_start_turn": lead_turn,
        "lead_starter": leader,
        "events": {k: v[:10] for k, v in events.items()},
        "errlog_a": match.get("errlog_a", ""),
        "errlog_b": match.get("errlog_b", ""),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"{summary['file']}: {summary['a']} vs {summary['b']} -> winner={summary['winner']} margin={summary['margin']} turns={summary['turn_count']} reason={summary['reason']}")
    if summary["first_scoring_turn"] is not None:
        print(f"  first scoring turn: {summary['first_scoring_turn']}")
    if summary["lead_start_turn"] is not None:
        print(f"  winner first led on turn: {summary['lead_start_turn']} ({summary['lead_starter']})")
    interesting = [
        key for key in summary["events"].keys()
        if key.startswith("a_point_gain") or key.startswith("b_point_gain") or key in {"rat_caught", "new_carpet"}
    ]
    for key in sorted(interesting):
        turns = summary["events"][key]
        if turns:
            print(f"  {key}: {turns}")


def print_cassie_rl_snapshot() -> None:
    if not RL_STATE_PATH.exists():
        return

    try:
        with RL_STATE_PATH.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except Exception:
        return

    profiles = state.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        return

    print()
    print("=== Cassie RL Snapshot ===")
    print(f"epsilon: {float(state.get('epsilon', 0.0)):.4f}")
    print(f"matches_seen: {int(state.get('matches_seen', 0))}")
    print(f"mutation_period: {int(state.get('mutation_period', 0))}")

    for name in sorted(profiles.keys()):
        p = profiles[name]
        q = float(p.get("q", 0.0))
        n = int(p.get("n", 0))
        print(
            f"{name}: q={q:.3f} n={n} "
            f"cst={int(p.get('convert_soft_threshold', 2))} "
            f"cmp={int(p.get('contest_min_points', 3))} "
            f"rts={float(p.get('root_threat_scale', 1.0)):.2f} "
            f"ect={int(p.get('endgame_convert_turns', 16))}"
        )

        q_seat = p.get("q_seat", {})
        n_seat = p.get("n_seat", {})
        if isinstance(q_seat, dict) and isinstance(n_seat, dict):
            for seat in sorted(set(q_seat.keys()) | set(n_seat.keys())):
                print(f"  seat {seat}: q={float(q_seat.get(seat, 0.0)):.3f} n={int(n_seat.get(seat, 0))}")


def _print_analysis_for_files(files: list[pathlib.Path]) -> None:
    if not files:
        print("No matches to summarize.")
        return

    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    totals = Counter()
    bot_wins = Counter()

    for path in files:
        summary = summarize_match(path)
        print_summary(summary)
        pair = (summary["a"], summary["b"])
        by_pair[pair].append(summary)
        totals[f"{summary['winner']}_wins"] += 1
        totals["matches"] += 1
        totals["margin_sum"] += summary["margin"]
        totals["turn_sum"] += summary["turn_count"]
        if summary["winner"] == "A":
            bot_wins[summary["a"]] += 1
        elif summary["winner"] == "B":
            bot_wins[summary["b"]] += 1
        if summary["errlog_a"]:
            totals["errlog_a"] += 1
        if summary["errlog_b"]:
            totals["errlog_b"] += 1

    print()
    print("=== Aggregate ===")
    print(f"matches: {totals['matches']}")
    print(f"A wins: {totals['A_wins']}  B wins: {totals['B_wins']}  draws: {totals['DRAW_wins']}")
    print(f"avg margin (A-B): {totals['margin_sum'] / max(1, totals['matches']):.2f}")
    print(f"avg turns: {totals['turn_sum'] / max(1, totals['matches']):.2f}")
    print(f"matches with errlog_a: {totals['errlog_a']}  errlog_b: {totals['errlog_b']}")

    print()
    print("=== Bot Wins ===")
    for bot, wins in sorted(bot_wins.items(), key=lambda item: (-item[1], item[0])):
        print(f"{bot}: {wins}")

    print()
    print("=== By Pair ===")
    for (a, b), items in sorted(by_pair.items()):
        a_wins = sum(1 for item in items if item["winner"] == "A")
        b_wins = sum(1 for item in items if item["winner"] == "B")
        avg_margin = sum(item["margin"] for item in items) / len(items)
        avg_turns = sum(item["turn_count"] for item in items) / len(items)
        print(f"{a} vs {b}: n={len(items)} A={a_wins} B={b_wins} avg_margin={avg_margin:.2f} avg_turns={avg_turns:.2f}")

    print_cassie_rl_snapshot()


def analyze_matches(matches_dir: pathlib.Path, pattern: str) -> None:
    files = sorted(matches_dir.glob(pattern))
    if not files:
        print(f"No matches found in {matches_dir} matching {pattern}")
        return
    _print_analysis_for_files(files)


def agent_names_from_matches(matches_dir: pathlib.Path) -> list[str]:
    names = set()
    for path in matches_dir.glob("*.json"):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) >= 2:
            names.add(parts[0])
            names.add(parts[1])
    return sorted(names)


def run_matchup(a: str, b: str, repeats: int, matches_dir: pathlib.Path) -> list[pathlib.Path]:
    created: list[pathlib.Path] = []
    for i in range(repeats):
        before = set(matches_dir.glob("*.json"))
        print(f"[{i + 1}/{repeats}] {a} vs {b}")
        subprocess.run([sys.executable, str(RUNNER), a, b], cwd=str(PROJECT_ROOT), check=False)
        after = set(matches_dir.glob("*.json"))
        new_files = sorted(after - before)
        if new_files:
            created.extend(new_files)
    return created


def run_round_robin(names: list[str], repeats: int, include_reverse: bool, matches_dir: pathlib.Path) -> list[pathlib.Path]:
    if len(names) < 2:
        raise SystemExit("Need at least two agent names.")

    created: list[pathlib.Path] = []
    pair_iter = permutations(names, 2) if include_reverse else combinations(names, 2)
    for a, b in pair_iter:
        created.extend(run_matchup(a, b, repeats, matches_dir))
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-run agents and analyze saved match JSON files.")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Summarize existing match JSON files.")
    analyze.add_argument("--matches-dir", type=pathlib.Path, default=MATCHES_DIR)
    analyze.add_argument("--pattern", type=str, default="*.json")

    run = sub.add_parser("run", help="Run agents against each other and save matches.")
    run.add_argument("agents", nargs="*", help="Agent names. If omitted, they are inferred from existing match files.")
    run.add_argument("--matches-dir", type=pathlib.Path, default=MATCHES_DIR)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--reverse", action="store_true", help="Run both A vs B and B vs A.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        analyze_matches(args.matches_dir, args.pattern)
        return

    if args.command == "run":
        names = list(args.agents)
        if not names:
            names = agent_names_from_matches(args.matches_dir)
        created_files = run_round_robin(names, args.repeats, args.reverse, args.matches_dir)

        print()
        print("=== Summary for Newly Generated Matches ===")
        if created_files:
            _print_analysis_for_files(sorted(set(created_files)))
        else:
            print("No new match JSON files were detected.")
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
