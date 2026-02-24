from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pulp


# ----------------------------
# DraftKings NBA Classic config
# ----------------------------
DK_SLOTS = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
SALARY_CAP = 50_000

DEFAULT_MAX_PER_TEAM = 4
DEFAULT_N_SIMS = 5000
DEFAULT_RANDOM_SEED = 7


# ----------------------------
# Data model
# ----------------------------
@dataclass(frozen=True)
class Player:
    player_id: str
    name: str
    team: str
    positions: Tuple[str, ...]
    salary: int
    min_mu: float
    min_sd: float
    fppm_mu: float
    fppm_sd: float
    proj_own: Optional[float] = None


def parse_positions(pos_str: str) -> Tuple[str, ...]:
    # Accept "PG/SG" or "PG,SG"
    s = pos_str.replace(",", "/")
    parts = [p.strip().upper() for p in s.split("/") if p.strip()]
    return tuple(parts)


def eligible(slot: str, positions: Tuple[str, ...]) -> bool:
    pos = set(positions)
    if slot in pos:
        return True
    if slot == "G" and (("PG" in pos) or ("SG" in pos)):
        return True
    if slot == "F" and (("SF" in pos) or ("PF" in pos)):
        return True
    if slot == "UTIL":
        return True
    return False


# ----------------------------
# Simulation
# ----------------------------
def simulate_player_points(
    players: List[Player],
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, np.ndarray]:
    """
    Baseline NBA model:
      fantasy_points = minutes * fppm
    Both minutes and fppm are modeled as normals and clipped to reasonable ranges.

    Returns:
      dict[player_id] -> array shape (n_sims,)
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}

    for p in players:
        # Minutes: clip to [0, mu + 2.5 sd] as a crude cap (improve later)
        mins = rng.normal(p.min_mu, p.min_sd, n_sims)
        mins_high = max(6.0, p.min_mu + 2.5 * p.min_sd)
        mins = np.clip(mins, 0.0, mins_high)

        # FPPM: clip to [0, mu + 3 sd] to prevent absurd tails
        fppm = rng.normal(p.fppm_mu, p.fppm_sd, n_sims)
        fppm_high = max(0.1, p.fppm_mu + 3.0 * p.fppm_sd)
        fppm = np.clip(fppm, 0.0, fppm_high)

        out[p.player_id] = mins * fppm

    return out


def player_scores_from_sims(
    sim_by_player: Dict[str, np.ndarray],
    mode: str,
    own_by_player: Optional[Dict[str, float]] = None,
    own_lambda: float = 0.0,
) -> Dict[str, float]:
    """
    Cash: mean points
    GPP: 95th percentile points (proxy for ceiling)
    Optionally subtract an ownership penalty: score -= own_lambda * ownership
    """
    scores: Dict[str, float] = {}
    for pid, arr in sim_by_player.items():
        if mode == "cash":
            base = float(np.mean(arr))
        elif mode == "gpp":
            base = float(np.quantile(arr, 0.95))
        else:
            raise ValueError("mode must be 'cash' or 'gpp'")

        if own_by_player is not None and own_lambda > 0.0:
            base -= own_lambda * float(own_by_player.get(pid, 0.0))

        scores[pid] = base
    return scores


# ----------------------------
# Optimization (MILP)
# ----------------------------
def optimize_lineup(
    players: List[Player],
    score_by_player: Dict[str, float],
    salary_cap: int = SALARY_CAP,
    max_per_team: int = DEFAULT_MAX_PER_TEAM,
    min_salary_used: Optional[int] = None,  # e.g. 48_500 for GPP "leave salary"
    force_unique_vs: Optional[List[set]] = None,  # overlap constraints against prior lineups
    max_overlap: int = 6,
) -> List[Tuple[str, Player]]:
    """
    DraftKings NBA Classic lineup optimizer using MILP.

    force_unique_vs:
      list of prior lineup player_id sets.
      For each prior set S, we add: overlap_with_S <= max_overlap
    """
    prob = pulp.LpProblem("DK_NBA_Lineup", pulp.LpMaximize)

    # Decision variable: assign player to slot
    x = {
        (p.player_id, s): pulp.LpVariable(f"x_{p.player_id}_{s}", cat="Binary")
        for p in players
        for s in DK_SLOTS
    }

    # Exactly one player per slot
    for s in DK_SLOTS:
        prob += pulp.lpSum(x[(p.player_id, s)] for p in players) == 1, f"fill_{s}"

    # Each player at most once across all slots
    for p in players:
        prob += pulp.lpSum(x[(p.player_id, s)] for s in DK_SLOTS) <= 1, f"once_{p.player_id}"

    # Eligibility constraints
    for p in players:
        for s in DK_SLOTS:
            if not eligible(s, p.positions):
                prob += x[(p.player_id, s)] == 0, f"elig_{p.player_id}_{s}"

    # Salary cap + optional minimum salary used
    total_salary = pulp.lpSum(
        p.salary * pulp.lpSum(x[(p.player_id, s)] for s in DK_SLOTS) for p in players
    )
    prob += total_salary <= salary_cap, "salary_cap"
    if min_salary_used is not None:
        prob += total_salary >= min_salary_used, "min_salary_used"

    # Max players per team
    teams = sorted({p.team for p in players})
    for t in teams:
        team_count = pulp.lpSum(
            pulp.lpSum(x[(p.player_id, s)] for s in DK_SLOTS) for p in players if p.team == t
        )
        prob += team_count <= max_per_team, f"max_team_{t}"

    # Overlap constraints vs previous lineups
    if force_unique_vs:
        for i, prev_set in enumerate(force_unique_vs):
            overlap = pulp.lpSum(
                pulp.lpSum(x[(p.player_id, s)] for s in DK_SLOTS)
                for p in players
                if p.player_id in prev_set
            )
            prob += overlap <= max_overlap, f"overlap_{i}"

    # Objective: sum of player scores
    prob += pulp.lpSum(
        score_by_player[p.player_id] * pulp.lpSum(x[(p.player_id, s)] for s in DK_SLOTS)
        for p in players
    ), "objective"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return []

    chosen: List[Tuple[str, Player]] = []
    for s in DK_SLOTS:
        for p in players:
            if pulp.value(x[(p.player_id, s)]) == 1:
                chosen.append((s, p))
                break

    return chosen


def lineup_summary(lineup: List[Tuple[str, Player]]) -> Dict[str, float]:
    if not lineup:
        return {"salary": 0.0}
    salary = sum(p.salary for _, p in lineup)
    return {"salary": float(salary)}


# ----------------------------
# Main runner
# ----------------------------
def load_players(csv_path: str) -> List[Player]:
    df = pd.read_csv(csv_path)

    required = {"player_id", "name", "team", "positions", "salary", "min_mu", "min_sd", "fppm_mu", "fppm_sd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    players: List[Player] = []
    for _, r in df.iterrows():
        proj_own = None
        if "proj_own" in df.columns and not (pd.isna(r["proj_own"])):
            proj_own = float(r["proj_own"])
        players.append(
            Player(
                player_id=str(r["player_id"]),
                name=str(r["name"]),
                team=str(r["team"]).upper(),
                positions=parse_positions(str(r["positions"])),
                salary=int(r["salary"]),
                min_mu=float(r["min_mu"]),
                min_sd=float(r["min_sd"]),
                fppm_mu=float(r["fppm_mu"]),
                fppm_sd=float(r["fppm_sd"]),
                proj_own=proj_own,
            )
        )
    return players


def print_lineup(title: str, lineup: List[Tuple[str, Player]], sim_by_player: Dict[str, np.ndarray]) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if not lineup:
        print("No lineup found.")
        return

    rows = []
    total_salary = 0
    total_mean = 0.0
    total_p95 = 0.0
    for slot, p in lineup:
        arr = sim_by_player[p.player_id]
        mean = float(np.mean(arr))
        p95 = float(np.quantile(arr, 0.95))
        rows.append((slot, p.name, "/".join(p.positions), p.team, p.salary, mean, p95))
        total_salary += p.salary
        total_mean += mean
        total_p95 += p95

    df = pd.DataFrame(rows, columns=["Slot", "Player", "Pos", "Team", "Salary", "MeanFP", "P95FP"])
    print(df.to_string(index=False))
    print(f"\nTotal Salary: {total_salary}")
    print(f"Sum MeanFP : {total_mean:.2f}")
    print(f"Sum P95FP  : {total_p95:.2f}")


def build_portfolio(
    players: List[Player],
    sim_by_player: Dict[str, np.ndarray],
    n_lineups: int = 20,
    max_overlap: int = 6,
    own_lambda: float = 0.0,
    min_salary_used: Optional[int] = 48_500,
) -> List[List[Tuple[str, Player]]]:
    own_by_player = {p.player_id: p.proj_own for p in players if p.proj_own is not None} if any(p.proj_own is not None for p in players) else None
    score = player_scores_from_sims(sim_by_player, mode="gpp", own_by_player=own_by_player, own_lambda=own_lambda)

    portfolio: List[List[Tuple[str, Player]]] = []
    prior_sets: List[set] = []

    for _ in range(n_lineups):
        lineup = optimize_lineup(
            players,
            score_by_player=score,
            min_salary_used=min_salary_used,
            force_unique_vs=prior_sets,
            max_overlap=max_overlap,
        )
        if not lineup:
            break

        pset = {p.player_id for _, p in lineup}
        portfolio.append(lineup)
        prior_sets.append(pset)

    return portfolio


def main():
    csv_path = "players.csv"

    players = load_players(csv_path)
    print(f"Loaded {len(players)} players from {csv_path}")

    sim_by_player = simulate_player_points(players, n_sims=DEFAULT_N_SIMS, seed=DEFAULT_RANDOM_SEED)

    # CASH lineup: maximize mean
    cash_scores = player_scores_from_sims(sim_by_player, mode="cash")
    cash = optimize_lineup(players, cash_scores)
    print_lineup("CASH (maximize mean projection)", cash, sim_by_player)

    # GPP lineup: maximize P95, optional ownership penalty, optional min salary used
    own_by_player = {p.player_id: p.proj_own for p in players if p.proj_own is not None} if any(p.proj_own is not None for p in players) else None
    gpp_scores = player_scores_from_sims(sim_by_player, mode="gpp", own_by_player=own_by_player, own_lambda=0.0)
    gpp = optimize_lineup(players, gpp_scores, min_salary_used=48_500)  # tweak or set None
    print_lineup("GPP (maximize P95; min salary used = 48,500)", gpp, sim_by_player)

    # Portfolio of GPP lineups
    portfolio = build_portfolio(
        players,
        sim_by_player,
        n_lineups=20,
        max_overlap=6,       # <=6 shared players (out of 8)
        own_lambda=0.0,      # set e.g. 0.5 if proj_own is 0-1; tune later
        min_salary_used=48_500,
    )

    print("\n" + "=" * 60)
    print(f"GPP PORTFOLIO ({len(portfolio)} lineups, max_overlap=6)")
    print("=" * 60)
    for i, lu in enumerate(portfolio, 1):
        salary = sum(p.salary for _, p in lu)
        p95sum = sum(float(np.quantile(sim_by_player[p.player_id], 0.95)) for _, p in lu)
        names = ", ".join([p.name for _, p in lu])
        print(f"{i:02d}) Salary={salary}  SumP95={p95sum:.1f}  | {names}")

if __name__ == "__main__":
    main()