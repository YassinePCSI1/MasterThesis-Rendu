"""Multi-Agent Q-learning for the global crude oil market (Section 8b).

Two simultaneously-learning Q-learning agents (default: OPEC and US) operate
in the same triopoly market while RUS plays a myopic Cournot best-response.
Each learning agent observes ONLY the market price (plus its own previous
quantity) — exactly the information structure of Green-Porter (1984), already
implemented in the stochastic / repeated-game sections of this project.

Why this matters
----------------
The single-agent Q-learning baseline in ``rl_agent.py`` has only one learner
(OPEC) while US and RUS play deterministic best responses.  The opponents
therefore behave like passive followers — far from the real oil market in
which several major producers learn and adapt simultaneously.

This module adds a much stronger empirical test of the Folk Theorem:

    *Without knowing the model, without communication, and observing only
     the market price, do two independent Q-learning agents spontaneously
     converge towards the cartel benchmark?*

If yes, this is a strong empirical validation of:
  - **Folk Theorem (Section 5)**: cooperation can emerge through long-run
    repeated interaction with sufficient patience (here gamma = 0.95).
  - **Green-Porter (Section 7)**: collusion can be sustained even when only
    a noisy aggregate signal (the price) is observable.
  - **Evolutionary 'Punish' strategy (Section 9)**: any learned reciprocity
    pattern (output cuts triggered by past low prices) is the algorithmic
    counterpart of the Punish strategy in the replicator dynamics.

Information structure
---------------------
Each learning agent's state is a 2-tuple ``(price_bin, own_q_bin)``:
  - ``price_bin``  : market price last period, discretised.
  - ``own_q_bin``  : the agent's own previous output, discretised.

Crucially, NEITHER agent observes the rivals' quantities directly — they
must infer the rivals' aggregate behaviour through the price signal alone.
This is what we mean by ``observation_mode = "price_only"``.

Reward
------
``reward_i_t = (P_t - c_i) * q_i_t``  (per-step profit, no penalty).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    AdjustmentCostParams,
    CapacityParams,
    CostParams,
    DemandParams,
    MultiAgentRLParams,
)
from .cournot_repeated import best_response
from .demand import price_from_quantity


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiAgentRLOutcome:
    """Full output of the multi-agent Q-learning simulation.

    Fields
    ------
    q_history : per-step quantities for ALL players (including RUS).
    price_history : per-step market price.
    reward_history : per-step per-agent reward (only for learning agents).
    q_tables : final Q-table per learning agent (shape: (price_bins+1, own_q_bins+1, action_grid_size)).
    action_grids : per-agent discrete action grid.
    converged_price : average price over the last ``eval_last_fraction`` of training.
    converged_outputs : per-player average output over the same evaluation window.
    collusion_index : (P_conv - P_nash) / (P_cartel - P_nash).
                      0 = pure Nash, 1 = full cartel; values in (0, 1) indicate
                      partial / tacit collusion — the signature of Folk-Theorem
                      sustainability under price-only information.
    """

    q_history: List[Dict[str, float]]
    price_history: List[float]
    reward_history: Dict[str, List[float]]
    q_tables: Dict[str, np.ndarray]
    action_grids: Dict[str, np.ndarray]
    converged_price: float
    converged_outputs: Dict[str, float]
    collusion_index: float
    learning_players: List[str] = field(default_factory=list)
    # Optional: list of per-step event flags filled in by ``train_multiagent_ql``
    # when shock_schedule / forced_deviation are used.  Each entry is a dict
    # with keys ``step``, ``shock_active``, ``shock_delta_a``, ``forced``,
    # ``forced_player``.  Empty when no hooks were applied.
    event_history: List[Dict[str, Any]] = field(default_factory=list)
    # ── Post-training diagnostics (audit Option B) ─────────────────────────
    # Greedy-rollout evaluation: K rollouts of T steps with ε=0 on the
    # frozen Q-tables (RUS plays myopic BR).  Honest "converged" metric.
    greedy_price: float = float("nan")
    greedy_price_std: float = float("nan")
    greedy_collusion_index: float = float("nan")
    greedy_outputs: Dict[str, float] = field(default_factory=dict)
    # Q-table coverage diagnostics (averaged over learning agents).
    q_undervisited_pct: float = float("nan")
    q_visit_mean: float = float("nan")
    q_visit_min: float = float("nan")


# ---------------------------------------------------------------------------
# Helpers (kept local to avoid circular import with rl_agent)
# ---------------------------------------------------------------------------

def _cost_map(costs: CostParams) -> Dict[str, float]:
    return {"US": costs.c_us, "OPEC": costs.c_opec, "RUS": costs.c_rus}


def _cap_map(caps: CapacityParams) -> Dict[str, float]:
    return {"US": caps.cap_us, "OPEC": caps.cap_opec, "RUS": caps.cap_rus}


def _agent_action_grid(
    player: str,
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    grid_size: int,
    q_max_override: Optional[float] = None,
) -> np.ndarray:
    """Build a discrete action grid for an individual learning agent.

    The grid spans ``[0, q_max]`` where ``q_max`` defaults to the cooperative
    monopoly upper bound for that player when alone in the market — wide
    enough to contain both Nash and cartel quantities for every realistic
    calibration.  When a Stackelberg-style stress-test needs to reach larger
    quantities (a leader output can exceed the cooperative bound), the caller
    can pass an explicit ``q_max_override``.
    """
    if q_max_override is not None:
        return np.linspace(0.0, float(q_max_override), grid_size)
    cost_map = _cost_map(costs)
    cap_map = _cap_map(capacities)
    if capacities.enabled:
        q_max = cap_map[player]
    else:
        q_max = (demand.a - cost_map[player]) / (2 * demand.b)
    return np.linspace(0.0, q_max, grid_size)


def _bin_edges(low: float, high: float, n_bins: int) -> np.ndarray:
    """Internal bin edges for ``np.digitize`` — returns n_bins-1 edges."""
    return np.linspace(low, high, n_bins + 1)[1:-1]


def _digitize(value: float, edges: np.ndarray) -> int:
    return int(np.digitize(value, edges))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_multiagent_ql(
    learning_players: List[str],
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price: float,
    cartel_price: float,
    seed: int = 123,
    *,
    shock_schedule: Optional[List[Tuple[int, int, float]]] = None,
    forced_deviation: Optional[Dict[str, Any]] = None,
    action_q_max_override: Optional[Dict[str, float]] = None,
) -> MultiAgentRLOutcome:
    """Train independent Q-learning agents under Green-Porter information.

    Mechanics
    ---------
    * Every step, all learning agents pick an action **simultaneously** from
      their own epsilon-greedy policy on a 2-D state ``(price_bin, own_q_bin)``.
    * Non-learning players (here ``RUS``) play a myopic Cournot best-response
      against the *current* outputs chosen this period.
    * The market clears: ``Q = sum(q_i)``, ``P = a - b Q``.
    * Each learning agent receives ``reward = (P - c_i) * q_i`` and updates
      its Q-table independently with standard Q-learning.
    * Epsilon decays geometrically each episode, bounded below by
      ``epsilon_end`` to preserve a small permanent exploration noise.

    Optional hooks (audit Part III stress-tests)
    --------------------------------------------
    ``shock_schedule`` : list of ``(start_step, end_step, delta_a)`` triples.
        At every *global* step ``t`` (concatenated across episodes) such that
        ``start_step <= t < end_step``, the demand intercept ``a`` is shifted
        by ``delta_a`` (use a negative value for a Green-Porter style downturn).
        Outside any interval the calibrated ``demand`` is used unchanged.
        This implements the Green-Porter (1984) stress-test of the audit:
        *what happens to learned cooperation when negative demand shocks hit
        during training?*

    ``forced_deviation`` : mapping with keys
        ``player``        — name of the agent to force.
        ``q``             — exact output to inject (nearest action grid point is used).
        ``start_step``    — first global step at which the deviation starts.
        ``duration``      — number of consecutive steps to enforce the deviation.
        The forced player still receives the realised reward and updates its
        Q-table, so we measure both the *cost* of the deviation for the deviator
        and the *reaction* of the other learners (algorithmic punishment).

    Parameters
    ----------
    learning_players : agents that learn (typically ``["OPEC", "US"]``).
    all_players      : the full triopoly (``["US", "OPEC", "RUS"]``).
    demand, costs, capacities : calibrated model parameters.
    params : ``MultiAgentRLParams`` controlling training schedule.
    nash_price, cartel_price : pre-computed benchmark prices used to derive
                               the collusion index.

    Returns
    -------
    MultiAgentRLOutcome with full trajectories, Q-tables, converged averages,
    and the collusion index.
    """
    rng = np.random.default_rng(seed)
    cost_map = _cost_map(costs)

    non_learning = [p for p in all_players if p not in learning_players]

    # --- Build agent-specific action grids and Q-tables ---
    action_grids: Dict[str, np.ndarray] = {
        p: _agent_action_grid(
            p, demand, costs, capacities, params.action_grid_size,
            q_max_override=(action_q_max_override or {}).get(p),
        )
        for p in learning_players
    }
    q_tables: Dict[str, np.ndarray] = {
        p: np.zeros(
            (params.price_bins + 1, params.own_q_bins + 1, params.action_grid_size)
        )
        for p in learning_players
    }
    # Visit counts (one per (state, action) cell) for coverage diagnostics.
    q_visits: Dict[str, np.ndarray] = {
        p: np.zeros_like(q_tables[p], dtype=np.int64)
        for p in learning_players
    }

    # --- State binning ---
    price_min = demand.price_floor if demand.price_floor is not None else 0.0
    price_edges = _bin_edges(price_min, demand.a, params.price_bins)
    own_q_edges_per_agent: Dict[str, np.ndarray] = {
        p: _bin_edges(0.0, float(action_grids[p][-1]), params.own_q_bins)
        for p in learning_players
    }

    adj_no = AdjustmentCostParams(enabled=False, k=0.0)

    q_history: List[Dict[str, float]] = []
    price_history: List[float] = []
    reward_history: Dict[str, List[float]] = {p: [] for p in learning_players}
    # Optional: per-step trace of whether each step is currently shocked / forced.
    # Useful for downstream plotting / event detection.
    event_history: List[Dict[str, Any]] = []

    epsilon = params.epsilon_start
    global_step = 0

    # Pre-compute forced-deviation action index (nearest grid point) if requested.
    forced_action_idx: Optional[int] = None
    forced_player: Optional[str] = None
    forced_start: int = -1
    forced_end: int = -1
    if forced_deviation is not None:
        forced_player = str(forced_deviation["player"])
        if forced_player not in learning_players:
            raise ValueError(
                f"forced_deviation.player='{forced_player}' must be in learning_players={learning_players}"
            )
        target_q = float(forced_deviation["q"])
        grid = action_grids[forced_player]
        forced_action_idx = int(np.argmin(np.abs(grid - target_q)))
        forced_start = int(forced_deviation["start_step"])
        forced_end = forced_start + int(forced_deviation.get("duration", 1))

    shocks = list(shock_schedule) if shock_schedule else []

    for ep in range(params.episodes):
        # Reset per-episode state — start from zero outputs, "blind" market
        outputs = {p: 0.0 for p in all_players}
        last_price = price_from_quantity(0.0, demand)

        for _t in range(params.steps_per_episode):
            # --- Apply demand shock for this step (if any) ---
            current_a = float(demand.a)
            shock_active = False
            for s_start, s_end, delta_a in shocks:
                if s_start <= global_step < s_end:
                    current_a += float(delta_a)
                    shock_active = True
            demand_t = replace(demand, a=current_a) if shock_active else demand

            # --- Apply forced deviation for this step (if any) ---
            forcing_now = (
                forced_player is not None
                and forced_start <= global_step < forced_end
            )

            # --- Build each learning agent's current state ---
            states: Dict[str, Tuple[int, int]] = {}
            actions: Dict[str, int] = {}
            for p in learning_players:
                price_bin = _digitize(last_price, price_edges)
                own_q_bin = _digitize(outputs[p], own_q_edges_per_agent[p])
                states[p] = (price_bin, own_q_bin)

                if forcing_now and p == forced_player:
                    # Force this player's action; no exploration override.
                    a_idx = int(forced_action_idx)  # type: ignore[arg-type]
                elif rng.random() < epsilon:
                    a_idx = int(rng.integers(0, params.action_grid_size))
                else:
                    a_idx = int(np.argmax(q_tables[p][states[p]]))
                actions[p] = a_idx
                outputs[p] = float(action_grids[p][a_idx])

            # --- Non-learning players: myopic best-response under current demand ---
            for p in non_learning:
                q_others = sum(outputs[k] for k in all_players if k != p)
                outputs[p] = best_response(
                    q_others, outputs[p], demand_t, cost_map[p],
                    capacities, p, adj_no,
                )

            # --- Market clears under current demand ---
            Q = sum(outputs.values())
            price = price_from_quantity(Q, demand_t)

            # --- Q-learning updates ---
            for p in learning_players:
                reward = (price - cost_map[p]) * outputs[p]
                next_price_bin = _digitize(price, price_edges)
                next_own_q_bin = _digitize(outputs[p], own_q_edges_per_agent[p])
                next_state = (next_price_bin, next_own_q_bin)

                td_target = reward + params.gamma * float(np.max(q_tables[p][next_state]))
                a_idx = actions[p]
                td_error = td_target - q_tables[p][states[p] + (a_idx,)]
                q_tables[p][states[p] + (a_idx,)] += params.alpha * td_error
                q_visits[p][states[p] + (a_idx,)] += 1

                reward_history[p].append(reward)

            q_history.append(outputs.copy())
            price_history.append(price)
            event_history.append({
                "step": global_step,
                "shock_active": shock_active,
                "shock_delta_a": current_a - demand.a if shock_active else 0.0,
                "forced": forcing_now,
                "forced_player": forced_player if forcing_now else "",
            })
            last_price = price
            global_step += 1

        # --- Anneal exploration ---
        epsilon = max(params.epsilon_end, params.epsilon_start * (params.epsilon_decay ** (ep + 1)))

    # --- Converged averages over the last `eval_last_fraction` of training ---
    n_total = len(price_history)
    eval_start = int((1.0 - params.eval_last_fraction) * n_total)
    eval_start = max(0, min(eval_start, n_total - 1))

    converged_price = float(np.mean(price_history[eval_start:]))
    converged_outputs: Dict[str, float] = {
        p: float(np.mean([q[p] for q in q_history[eval_start:]]))
        for p in all_players
    }

    denom = (cartel_price - nash_price)
    if abs(denom) < 1e-9:
        collusion_index = 0.0
    else:
        collusion_index = (converged_price - nash_price) / denom

    # ── Post-training diagnostics (audit Option B) ─────────────────────────
    # 1. Q-cell coverage (averaged across learning agents).
    visit_counts_all = np.concatenate(
        [q_visits[p].ravel() for p in learning_players]
    ) if learning_players else np.array([0], dtype=np.int64)
    under_visited_mask = visit_counts_all < 5
    q_undervisited_pct = (
        float(under_visited_mask.mean()) if visit_counts_all.size else float("nan")
    )
    q_visit_mean = float(visit_counts_all.mean()) if visit_counts_all.size else float("nan")
    q_visit_min = float(visit_counts_all.min()) if visit_counts_all.size else float("nan")

    # 2. Greedy rollout (ε=0) on the frozen Q-tables for an honest estimate
    #    of the post-training equilibrium price.
    greedy = _greedy_rollout(
        learning_players=learning_players,
        all_players=all_players,
        demand=demand,
        costs=costs,
        capacities=capacities,
        params=params,
        q_tables=q_tables,
        action_grids=action_grids,
        price_edges=price_edges,
        own_q_edges_per_agent=own_q_edges_per_agent,
        seed=seed + 10_000,
        n_rollouts=20,
        T=params.steps_per_episode,
    )
    greedy_denom = (cartel_price - nash_price)
    greedy_ci = (
        (greedy["mean_price"] - nash_price) / greedy_denom
        if abs(greedy_denom) > 1e-9 else 0.0
    )

    return MultiAgentRLOutcome(
        q_history=q_history,
        price_history=price_history,
        reward_history=reward_history,
        q_tables=q_tables,
        action_grids=action_grids,
        converged_price=converged_price,
        converged_outputs=converged_outputs,
        collusion_index=float(collusion_index),
        learning_players=list(learning_players),
        event_history=event_history,
        greedy_price=float(greedy["mean_price"]),
        greedy_price_std=float(greedy["std_price"]),
        greedy_collusion_index=float(greedy_ci),
        greedy_outputs={p: float(greedy["outputs"][p]) for p in all_players},
        q_undervisited_pct=q_undervisited_pct,
        q_visit_mean=q_visit_mean,
        q_visit_min=q_visit_min,
    )


# ---------------------------------------------------------------------------
# Post-training diagnostics (audit Option B)
# ---------------------------------------------------------------------------

def _greedy_rollout(
    *,
    learning_players: List[str],
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    q_tables: Dict[str, np.ndarray],
    action_grids: Dict[str, np.ndarray],
    price_edges: np.ndarray,
    own_q_edges_per_agent: Dict[str, np.ndarray],
    seed: int,
    n_rollouts: int = 20,
    T: int = 50,
) -> Dict[str, Any]:
    """Greedy (ε=0) evaluation of the frozen Q-tables.

    Runs ``n_rollouts`` independent rollouts of length ``T`` where every
    learning agent picks ``argmax Q[s, ·]`` at each step and non-learning
    agents (e.g. RUS) play a myopic Cournot best-response.  No learning,
    no exploration.  The mean price across all rollout steps is the
    honest "converged" market outcome (replacing the tail-of-training
    estimate that still includes residual ε-greedy noise).
    """
    rng = np.random.default_rng(seed)
    cost_map = _cost_map(costs)
    non_learning = [p for p in all_players if p not in learning_players]
    adj_no = AdjustmentCostParams(enabled=False, k=0.0)

    rollout_prices: List[float] = []
    rollout_outputs: Dict[str, List[float]] = {p: [] for p in all_players}

    for _r in range(n_rollouts):
        outputs = {p: 0.0 for p in all_players}
        last_price = price_from_quantity(0.0, demand)
        for _t in range(T):
            for p in learning_players:
                price_bin = _digitize(last_price, price_edges)
                own_q_bin = _digitize(outputs[p], own_q_edges_per_agent[p])
                a_idx = int(np.argmax(q_tables[p][price_bin, own_q_bin]))
                outputs[p] = float(action_grids[p][a_idx])
            for p in non_learning:
                q_others = sum(outputs[k] for k in all_players if k != p)
                outputs[p] = best_response(
                    q_others, outputs[p], demand, cost_map[p],
                    capacities, p, adj_no,
                )
            Q = sum(outputs.values())
            price = price_from_quantity(Q, demand)
            rollout_prices.append(price)
            for p in all_players:
                rollout_outputs[p].append(outputs[p])
            last_price = price

    # rng is currently unused beyond seeding the wrapper; reserved for future
    # stochastic-policy extensions (kept for API stability).
    _ = rng
    prices = np.asarray(rollout_prices, dtype=float)
    return {
        "mean_price": float(prices.mean()) if prices.size else float("nan"),
        "std_price": float(prices.std(ddof=1)) if prices.size > 1 else 0.0,
        "outputs": {
            p: float(np.mean(rollout_outputs[p])) if rollout_outputs[p] else float("nan")
            for p in all_players
        },
        "n_rollouts": int(n_rollouts),
        "T": int(T),
    }


def run_random_policy_baseline(
    *,
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price: float,
    cartel_price: float,
    n_seeds: int = 20,
    base_seed: int = 999,
) -> Dict[str, Any]:
    """Random-policy baseline — every player samples uniformly from its
    action grid at every step.  Used as a sanity-check anchor for the
    collusion index: if the random baseline already sits at CI ≈ 0.4 the
    discretisation is biased and "learned collusion" claims need a caveat.
    """
    grids = {
        p: _agent_action_grid(p, demand, costs, capacities, params.action_grid_size)
        for p in all_players
    }

    per_seed_prices: List[float] = []
    per_seed_ci: List[float] = []
    denom = cartel_price - nash_price

    for i in range(n_seeds):
        rng = np.random.default_rng(base_seed + i)
        prices: List[float] = []
        for _t in range(int(params.episodes) * int(params.steps_per_episode)):
            outputs = {
                p: float(grids[p][int(rng.integers(0, params.action_grid_size))])
                for p in all_players
            }
            Q = sum(outputs.values())
            prices.append(price_from_quantity(Q, demand))
        mean_p = float(np.mean(prices))
        per_seed_prices.append(mean_p)
        per_seed_ci.append(
            float((mean_p - nash_price) / denom) if abs(denom) > 1e-9 else 0.0
        )

    pr = np.asarray(per_seed_prices, dtype=float)
    ci = np.asarray(per_seed_ci, dtype=float)
    halfwidth = 1.96 * float(ci.std(ddof=1)) / sqrt(n_seeds) if n_seeds > 1 else 0.0
    return {
        "n_seeds": int(n_seeds),
        "mean_price": float(pr.mean()),
        "std_price": float(pr.std(ddof=1)) if n_seeds > 1 else 0.0,
        "mean_collusion_index": float(ci.mean()),
        "std_collusion_index": float(ci.std(ddof=1)) if n_seeds > 1 else 0.0,
        "ci_95_low": float(ci.mean() - halfwidth),
        "ci_95_high": float(ci.mean() + halfwidth),
        "per_seed_prices": per_seed_prices,
        "per_seed_collusion_indices": per_seed_ci,
    }


# ---------------------------------------------------------------------------
# Robustness analysis (multi-seed)
# ---------------------------------------------------------------------------

def run_multiagent_robustness(
    learning_players: List[str],
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price: float,
    cartel_price: float,
    n_seeds: int = 20,
    base_seed: int = 100,
) -> Dict[str, Any]:
    """Run ``train_multiagent_ql`` across ``n_seeds`` independent random seeds
    to assess the statistical robustness of the collusion result.

    A single seed of any RL experiment proves nothing: training trajectories
    are stochastic and can converge to qualitatively different basins of
    attraction depending on initial exploration noise.  This robustness
    analysis quantifies the uncertainty around the headline collusion
    index by reporting its full distribution and a 95% confidence interval
    of the mean (using a normal-approximation CI:
    ``mean ± 1.96 · std / sqrt(n_seeds)``).

    Parameters
    ----------
    learning_players, all_players, demand, costs, capacities, params,
    nash_price, cartel_price : forwarded to :func:`train_multiagent_ql`.
    n_seeds : number of independent training runs.
    base_seed : seeds will be ``[base_seed, base_seed+1, …, base_seed+n_seeds-1]``.

    Returns
    -------
    dict with keys:
        ``collusion_indices``  : list of length ``n_seeds``.
        ``converged_prices``   : list of converged prices.
        ``converged_outputs``  : list of dicts (per-player average output).
        ``mean_collusion_index``, ``std_collusion_index`` : summary stats.
        ``ci_95_low``, ``ci_95_high`` : 95% normal-approximation CI of the mean.
        ``mean_converged_price``, ``std_converged_price`` : price summary stats.
        ``seeds`` : list of seeds actually used.
        ``all_outcomes``       : list of full :class:`MultiAgentRLOutcome` objects
                                 (kept for downstream debugging / re-plotting).
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")

    collusion_indices: List[float] = []
    converged_prices: List[float] = []
    converged_outputs: List[Dict[str, float]] = []
    greedy_prices: List[float] = []
    greedy_collusion: List[float] = []
    q_undervisited_pcts: List[float] = []
    q_visit_means: List[float] = []
    q_visit_mins: List[float] = []
    all_outcomes: List[MultiAgentRLOutcome] = []
    seeds: List[int] = []

    for i in range(n_seeds):
        seed = base_seed + i
        seeds.append(seed)
        print(f"[multi-agent robustness] Seed {i + 1}/{n_seeds} (seed={seed})")

        outcome = train_multiagent_ql(
            learning_players=learning_players,
            all_players=all_players,
            demand=demand,
            costs=costs,
            capacities=capacities,
            params=params,
            nash_price=nash_price,
            cartel_price=cartel_price,
            seed=seed,
        )

        collusion_indices.append(float(outcome.collusion_index))
        converged_prices.append(float(outcome.converged_price))
        converged_outputs.append(dict(outcome.converged_outputs))
        greedy_prices.append(float(outcome.greedy_price))
        greedy_collusion.append(float(outcome.greedy_collusion_index))
        q_undervisited_pcts.append(float(outcome.q_undervisited_pct))
        q_visit_means.append(float(outcome.q_visit_mean))
        q_visit_mins.append(float(outcome.q_visit_min))
        all_outcomes.append(outcome)

    ci_arr = np.asarray(collusion_indices, dtype=float)
    pr_arr = np.asarray(converged_prices, dtype=float)
    gp_arr = np.asarray(greedy_prices, dtype=float)
    gci_arr = np.asarray(greedy_collusion, dtype=float)

    mean_ci = float(ci_arr.mean())
    std_ci = float(ci_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    halfwidth = 1.96 * std_ci / sqrt(n_seeds) if n_seeds > 1 else 0.0
    mean_gci = float(gci_arr.mean())
    std_gci = float(gci_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    g_half = 1.96 * std_gci / sqrt(n_seeds) if n_seeds > 1 else 0.0

    return {
        "seeds": seeds,
        "collusion_indices": collusion_indices,
        "converged_prices": converged_prices,
        "converged_outputs": converged_outputs,
        "mean_collusion_index": mean_ci,
        "std_collusion_index": std_ci,
        "ci_95_low": float(mean_ci - halfwidth),
        "ci_95_high": float(mean_ci + halfwidth),
        "mean_converged_price": float(pr_arr.mean()),
        "std_converged_price": float(pr_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
        # Greedy-rollout aggregates (audit Option B).
        "greedy_prices": greedy_prices,
        "greedy_collusion_indices": greedy_collusion,
        "mean_greedy_price": float(gp_arr.mean()),
        "std_greedy_price": float(gp_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
        "mean_greedy_collusion_index": mean_gci,
        "std_greedy_collusion_index": std_gci,
        "greedy_ci_95_low": float(mean_gci - g_half),
        "greedy_ci_95_high": float(mean_gci + g_half),
        # Q-coverage diagnostics.
        "q_undervisited_pcts": q_undervisited_pcts,
        "q_visit_means": q_visit_means,
        "q_visit_mins": q_visit_mins,
        "mean_q_undervisited_pct": float(np.mean(q_undervisited_pcts)) if q_undervisited_pcts else float("nan"),
        "mean_q_visit_mean": float(np.mean(q_visit_means)) if q_visit_means else float("nan"),
        "min_q_visit_min": float(np.min(q_visit_mins)) if q_visit_mins else float("nan"),
        "n_seeds": n_seeds,
        "all_outcomes": all_outcomes,
    }


# ---------------------------------------------------------------------------
# Emergent punishment detection (Green-Porter / Folk-Theorem signature)
# ---------------------------------------------------------------------------

@dataclass
class PunishmentEpisode:
    """A single punishment episode detected in the multi-agent price trajectory.

    A punishment episode is the algorithmic counterpart of the cooperation →
    defection → punishment → recovery cycle predicted by the Folk Theorem
    (Section 5) and implemented analytically in Green-Porter (1984, Section 7).

    Fields
    ------
    start_step      : index where the (smoothed) price starts dropping.
    trough_step     : index of the price minimum during the episode.
    recovery_step   : index where the price first returns to
                      ``recovery_threshold * pre_price``.
    price_drop      : amplitude of the drop ($/bbl), ``pre_price - trough_price``.
    duration        : ``recovery_step - trough_step`` (recovery length only).
    pre_price       : average smoothed price just before ``start_step``.
    trough_price    : smoothed price at the trough.
    post_price      : average smoothed price just after recovery.
    """

    start_step: int
    trough_step: int
    recovery_step: int
    price_drop: float
    duration: int
    pre_price: float
    trough_price: float
    post_price: float


def _rolling_mean_local(x: np.ndarray, window: int) -> np.ndarray:
    """Centred-trailing rolling mean, same length as ``x`` (left-padded).

    Duplicated from ``plotting._rolling_mean`` to keep this module standalone
    and avoid a circular import.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) <= window:
        return x
    rolled = np.convolve(x, np.ones(window) / window, mode="valid")
    pad = np.full(window - 1, rolled[0])
    return np.concatenate([pad, rolled])


def _local_max_indices(r: np.ndarray, half_window: int) -> List[int]:
    """Indices that are the (strict-or-equal) maximum of their ±half_window neighbourhood."""
    n = len(r)
    out: List[int] = []
    for i in range(half_window, n - half_window):
        seg = r[i - half_window: i + half_window + 1]
        if r[i] >= seg.max() - 1e-12:
            out.append(i)
    return out


def _local_min_indices(r: np.ndarray, half_window: int) -> List[int]:
    """Indices that are the (strict-or-equal) minimum of their ±half_window neighbourhood."""
    n = len(r)
    out: List[int] = []
    for i in range(half_window, n - half_window):
        seg = r[i - half_window: i + half_window + 1]
        if r[i] <= seg.min() + 1e-12:
            out.append(i)
    return out


def detect_punishment_episodes(
    outcome: MultiAgentRLOutcome,
    window: int = 50,
    drop_threshold: float = 3.0,
    recovery_threshold: float = 0.8,
    min_drop_steps: int = 5,
    lookahead_factor: int = 4,
    *,
    drop_sigma_k: Optional[float] = 2.0,
    recovery_sigma_k: Optional[float] = 1.0,
    sigma_window_frac: float = 0.20,
) -> List[PunishmentEpisode]:
    """Detect emergent punishment episodes in the multi-agent price trajectory.

    Pre-specified detection criteria (σ-calibrated)
    -----------------------------------------------
    The naive absolute thresholds (``drop_threshold = 3$/bbl``,
    ``recovery_threshold = 0.8 · pre_price``) are kept as *defaults* for
    backward compatibility but are deliberately re-calibrated against the
    *late-training empirical volatility* of the price series:

    * The post-convergence standard deviation ``σ_P`` is computed on the
      last ``sigma_window_frac`` (default 20%) of the rolling-mean
      trajectory.
    * The effective drop threshold becomes
      ``max(drop_threshold, drop_sigma_k · σ_P)``.  Punishment candidates
      whose drop is below ~ ``drop_sigma_k`` standard deviations of the
      converged noise are treated as exploration jitter, not Folk-Theorem
      cycles.  Default ``drop_sigma_k = 2.0``.
    * The recovery is considered complete when the price returns to within
      ``recovery_sigma_k · σ_P`` of the pre-drop level (instead of falling
      80% of pre_price, a far weaker criterion in practice).  Default
      ``recovery_sigma_k = 1.0``.

    Set ``drop_sigma_k=None`` and ``recovery_sigma_k=None`` to restore the
    legacy absolute-threshold behaviour.

    Returns
    -------
    list of :class:`PunishmentEpisode` sorted by ``start_step``.
    """
    prices = np.asarray(outcome.price_history, dtype=float)
    if len(prices) < 4 * window:
        return []

    r = _rolling_mean_local(prices, window=window)
    half_window = max(window // 2, 5)
    lookahead = lookahead_factor * window

    # Calibrate thresholds on late-training volatility (post-convergence).
    n = len(r)
    tail_start = max(0, int((1.0 - sigma_window_frac) * n))
    tail = r[tail_start:] if tail_start < n else r
    sigma_p = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
    effective_drop = (
        max(drop_threshold, drop_sigma_k * sigma_p)
        if drop_sigma_k is not None else drop_threshold
    )

    max_idx = _local_max_indices(r, half_window)
    min_idx_set = set(_local_min_indices(r, half_window))

    episodes: List[PunishmentEpisode] = []
    last_recovery = -1

    for peak in max_idx:
        if peak <= last_recovery:
            continue

        end_search = min(peak + lookahead, len(r) - 1)
        if end_search - peak < min_drop_steps:
            continue

        # Find the local minimum within the look-ahead window
        candidate_mins = [j for j in range(peak + 1, end_search + 1)
                          if j in min_idx_set]
        if not candidate_mins:
            local_trough = int(np.argmin(r[peak + 1: end_search + 1])) + peak + 1
        else:
            local_trough = int(min(
                candidate_mins,
                key=lambda j: r[j],
            ))

        pre_price = float(r[peak])
        trough_price = float(r[local_trough])
        drop = pre_price - trough_price
        if drop < effective_drop:
            continue
        if local_trough - peak < min_drop_steps:
            continue

        # Recovery search.  Prefer the σ-tight criterion when σ is known;
        # otherwise fall back to the legacy fraction-of-peak target.
        if recovery_sigma_k is not None and sigma_p > 0.0:
            target = pre_price - recovery_sigma_k * sigma_p
        else:
            target = recovery_threshold * pre_price
        recovery_step = -1
        for s in range(local_trough + 1, len(r)):
            if r[s] >= target:
                recovery_step = s
                break
        if recovery_step < 0:
            continue

        post_window_end = min(recovery_step + window, len(r))
        post_price = float(np.mean(r[recovery_step:post_window_end]))

        episodes.append(PunishmentEpisode(
            start_step=int(peak),
            trough_step=int(local_trough),
            recovery_step=int(recovery_step),
            price_drop=float(drop),
            duration=int(recovery_step - local_trough),
            pre_price=float(pre_price),
            trough_price=float(trough_price),
            post_price=float(post_price),
        ))
        last_recovery = recovery_step

    return episodes


def compute_punishment_statistics(
    episodes: List[PunishmentEpisode],
    total_steps: int = 0,
) -> Dict[str, float]:
    """Aggregate statistics over a list of detected punishment episodes.

    Parameters
    ----------
    episodes    : output of :func:`detect_punishment_episodes`.
    total_steps : total length (in steps) of the underlying trajectory.
                  Used to express ``punishment_frequency`` per 1000 steps.
                  When omitted (or 0), the frequency is reported as 0.

    Returns
    -------
    dict with keys (all numeric, suitable for CSV serialisation):
        ``n_episodes``           : number of detected episodes.
        ``mean_duration``        : mean *full-cycle* length (start → recovery), steps.
        ``mean_recovery_time``   : mean recovery length (trough → recovery), steps.
        ``mean_drop``            : mean drop amplitude ($/bbl).
        ``mean_trough_price``    : mean price at trough ($/bbl).
        ``mean_pre_price``       : mean cooperative price preceding the drop ($/bbl).
        ``punishment_frequency`` : episodes per 1000 steps (0 if total_steps == 0).
    """
    n = len(episodes)
    if n == 0:
        return {
            "n_episodes": 0,
            "mean_duration": 0.0,
            "mean_recovery_time": 0.0,
            "mean_drop": 0.0,
            "mean_trough_price": 0.0,
            "mean_pre_price": 0.0,
            "punishment_frequency": 0.0,
        }

    cycle_lengths = [ep.recovery_step - ep.start_step for ep in episodes]
    return {
        "n_episodes": int(n),
        "mean_duration": float(np.mean(cycle_lengths)),
        "mean_recovery_time": float(np.mean([ep.duration for ep in episodes])),
        "mean_drop": float(np.mean([ep.price_drop for ep in episodes])),
        "mean_trough_price": float(np.mean([ep.trough_price for ep in episodes])),
        "mean_pre_price": float(np.mean([ep.pre_price for ep in episodes])),
        "punishment_frequency": (
            1000.0 * n / total_steps if total_steps and total_steps > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Learner-count comparison (1 vs 2 vs 3 learning agents)
# ---------------------------------------------------------------------------

def run_multiagent_comparison(
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price: float,
    cartel_price: float,
    nash_quantities: Dict[str, float],
    cartel_quantities: Dict[str, float],
    n_seeds: int = 10,
    base_seed: int = 200,
) -> Dict[str, Any]:
    """Compare the converged collusion outcome for three learner-count regimes.

    The configurations tested are:

    * ``"single"``   — only OPEC learns; US and RUS play myopic Cournot
                       best-response (matches Section 8 single-agent baseline).
    * ``"duopoly"``  — OPEC and US learn; RUS plays myopic best-response
                       (the headline Section 8b configuration).
    * ``"triopoly"`` — every player in ``all_players`` learns simultaneously;
                       no agent is myopic.

    For each regime we run ``n_seeds`` independent Q-learning trainings with
    consecutive seeds starting at ``base_seed`` (the SAME seed list is used
    across regimes to make the comparison paired).  The function returns
    summary statistics on the collusion index, the converged price and the
    converged per-player outputs.

    Theoretical question
    --------------------
    The standard oligopoly intuition is that collusion is harder to sustain
    as the number of strategic players grows.  Going from 2 to 3 learning
    agents should therefore *lower* the collusion index — but the picture
    is ambiguous in our setup, because the duopoly regime contains a
    deterministic free-rider (myopic RUS) that mechanically pulls the price
    back towards the Nash benchmark.  Letting RUS learn removes that
    free-rider and could plausibly *increase* the collusion index instead.
    The empirical result is reported as-is.

    Note on ``train_multiagent_ql`` reuse
    -------------------------------------
    The training function is already generic enough: when
    ``learning_players == all_players`` the internal ``non_learning`` list
    is empty and no myopic best-response is called.  No modification of
    the inner training loop is required.

    Returns
    -------
    dict with keys ``"single"``, ``"duopoly"``, ``"triopoly"``, each mapping
    to a dict with fields ``learning_players``, ``n_learners``, ``n_seeds``,
    ``collusion_indices``, ``converged_prices``, ``converged_outputs``,
    ``mean_collusion_index``, ``std_collusion_index``,
    ``mean_converged_price``, ``std_converged_price``,
    ``mean_outputs``, ``std_outputs``.  Two additional top-level keys
    ``nash_quantities`` and ``cartel_quantities`` echo the benchmark
    quantities for convenience downstream.
    """
    if n_seeds <= 0:
        raise ValueError("n_seeds must be positive")

    configurations: Dict[str, List[str]] = {
        "single":   ["OPEC"],
        "duopoly":  ["OPEC", "US"],
        "triopoly": list(all_players),
    }

    total_runs = len(configurations) * n_seeds
    counter = 0
    results: Dict[str, Any] = {}

    for label, learners in configurations.items():
        collusion_indices: List[float] = []
        converged_prices: List[float] = []
        converged_outputs: List[Dict[str, float]] = []

        for i in range(n_seeds):
            counter += 1
            seed = base_seed + i
            print(
                f"[learner-comparison] {label} ({len(learners)} learners) — "
                f"seed {i + 1}/{n_seeds}  [{counter}/{total_runs}]"
            )

            outcome = train_multiagent_ql(
                learning_players=list(learners),
                all_players=list(all_players),
                demand=demand,
                costs=costs,
                capacities=capacities,
                params=params,
                nash_price=nash_price,
                cartel_price=cartel_price,
                seed=seed,
            )

            collusion_indices.append(float(outcome.collusion_index))
            converged_prices.append(float(outcome.converged_price))
            converged_outputs.append(dict(outcome.converged_outputs))

        ci_arr = np.asarray(collusion_indices, dtype=float)
        pr_arr = np.asarray(converged_prices, dtype=float)
        per_player = {
            p: np.asarray([d[p] for d in converged_outputs], dtype=float)
            for p in all_players
        }

        results[label] = {
            "learning_players": list(learners),
            "n_learners": len(learners),
            "n_seeds": n_seeds,
            "collusion_indices": collusion_indices,
            "converged_prices": converged_prices,
            "converged_outputs": converged_outputs,
            "mean_collusion_index": float(ci_arr.mean()),
            "std_collusion_index": float(ci_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
            "mean_converged_price": float(pr_arr.mean()),
            "std_converged_price": float(pr_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
            "mean_outputs": {p: float(v.mean()) for p, v in per_player.items()},
            "std_outputs": {
                p: float(v.std(ddof=1)) if n_seeds > 1 else 0.0
                for p, v in per_player.items()
            },
        }

    results["nash_quantities"] = dict(nash_quantities)
    results["cartel_quantities"] = dict(cartel_quantities)
    results["nash_price"] = float(nash_price)
    results["cartel_price"] = float(cartel_price)
    results["players"] = list(all_players)
    return results


# ---------------------------------------------------------------------------
# Audit Part III stress-tests: γ sweep, shocks, forced deviation, capacities
# ---------------------------------------------------------------------------

def _summarise_seed_runs(
    collusion_indices: List[float],
    converged_prices: List[float],
    per_player_outputs: Dict[str, List[float]],
) -> Dict[str, float]:
    """Mean / std / 95% CI over a list of independent seed runs."""
    n = len(collusion_indices)
    ci = np.asarray(collusion_indices, dtype=float)
    pr = np.asarray(converged_prices, dtype=float)
    mean_ci = float(ci.mean())
    std_ci = float(ci.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std_ci / sqrt(n) if n > 1 else 0.0
    out: Dict[str, float] = {
        "n_seeds": n,
        "mean_collusion_index": mean_ci,
        "std_collusion_index": std_ci,
        "ci_95_low": mean_ci - half,
        "ci_95_high": mean_ci + half,
        "mean_converged_price": float(pr.mean()),
        "std_converged_price": float(pr.std(ddof=1)) if n > 1 else 0.0,
    }
    for p, vals in per_player_outputs.items():
        arr = np.asarray(vals, dtype=float)
        out[f"mean_q_{p}"] = float(arr.mean())
        out[f"std_q_{p}"] = float(arr.std(ddof=1)) if n > 1 else 0.0
    return out


def run_marl_gamma_sweep(
    learning_players: List[str],
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price: float,
    cartel_price: float,
    gamma_values: List[float],
    n_seeds: int = 3,
    base_seed: int = 500,
    *,
    return_per_seed: bool = False,
) -> Any:
    """Stress-test patience: re-train MARL at several discount factors γ.

    Each γ value is tested with ``n_seeds`` independent seeds; the returned
    object is either:

    * an aggregated list of dicts (default, backwards compatible) — one row
      per γ with the same summary fields as :func:`run_multiagent_robustness`;
    * or, when ``return_per_seed=True``, a dict ``{"aggregated": rows,
      "per_seed": per_seed_rows}`` where ``per_seed_rows`` is a list of dicts
      with one entry per (γ, seed) pair, exposing the full distribution of
      converged prices and collusion indices used downstream by basin /
      density plots.

    Theoretical link
    ----------------
    γ plays the role of the discount factor δ in the Folk Theorem: with
    impatient agents (γ small) any tacitly cooperative price should
    unravel toward Nash, while patient agents (γ → 1) make algorithmic
    cooperation easier to sustain.  This sweep visualises that boundary.
    """
    rows: List[Dict[str, float]] = []
    per_seed_rows: List[Dict[str, float]] = []
    total = len(gamma_values) * n_seeds
    counter = 0
    for gamma in gamma_values:
        ci_list: List[float] = []
        pr_list: List[float] = []
        per_player: Dict[str, List[float]] = {p: [] for p in all_players}
        for i in range(n_seeds):
            counter += 1
            seed = base_seed + i + int(round(gamma * 1000))
            print(
                f"[marl γ-sweep] γ={gamma:.2f}, seed {i + 1}/{n_seeds}"
                f"  [{counter}/{total}]"
            )
            params_g = replace(params, gamma=float(gamma))
            outcome = train_multiagent_ql(
                learning_players=list(learning_players),
                all_players=list(all_players),
                demand=demand,
                costs=costs,
                capacities=capacities,
                params=params_g,
                nash_price=nash_price,
                cartel_price=cartel_price,
                seed=seed,
            )
            ci_list.append(float(outcome.collusion_index))
            pr_list.append(float(outcome.converged_price))
            for p in all_players:
                per_player[p].append(float(outcome.converged_outputs[p]))

            per_seed_rows.append({
                "gamma": float(gamma),
                "seed": int(seed),
                "converged_price": float(outcome.converged_price),
                "collusion_index": float(outcome.collusion_index),
                "greedy_price": float(outcome.greedy_price),
                "greedy_collusion_index": float(outcome.greedy_collusion_index),
                "q_undervisited_pct": float(outcome.q_undervisited_pct),
                **{f"q_{p}": float(outcome.converged_outputs[p]) for p in all_players},
            })

        summary = _summarise_seed_runs(ci_list, pr_list, per_player)
        rows.append({"gamma": float(gamma), **summary})

    if return_per_seed:
        return {"aggregated": rows, "per_seed": per_seed_rows}
    return rows


def run_marl_shock_experiment(
    learning_players: List[str],
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price: float,
    cartel_price: float,
    shock_schedule: List[Tuple[int, int, float]],
    n_seeds: int = 3,
    base_seed: int = 600,
) -> Dict[str, Any]:
    """Stress-test Green-Porter: train MARL while injecting demand shocks.

    Returns a dict with the multi-seed summary plus the *headline* outcome
    (first seed) — kept so that downstream plotting can show one full
    trajectory of the price with shock markers.
    """
    ci_list: List[float] = []
    pr_list: List[float] = []
    per_player: Dict[str, List[float]] = {p: [] for p in all_players}
    headline: Optional[MultiAgentRLOutcome] = None
    for i in range(n_seeds):
        seed = base_seed + i
        print(f"[marl shock-experiment] seed {i + 1}/{n_seeds} (seed={seed})")
        outcome = train_multiagent_ql(
            learning_players=list(learning_players),
            all_players=list(all_players),
            demand=demand,
            costs=costs,
            capacities=capacities,
            params=params,
            nash_price=nash_price,
            cartel_price=cartel_price,
            seed=seed,
            shock_schedule=list(shock_schedule),
        )
        if headline is None:
            headline = outcome
        ci_list.append(float(outcome.collusion_index))
        pr_list.append(float(outcome.converged_price))
        for p in all_players:
            per_player[p].append(float(outcome.converged_outputs[p]))

    summary = _summarise_seed_runs(ci_list, pr_list, per_player)
    return {
        "schedule": list(shock_schedule),
        "headline_outcome": headline,
        **summary,
    }


def run_marl_forced_deviation(
    learning_players: List[str],
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price: float,
    cartel_price: float,
    deviator: str,
    deviation_q: float,
    deviation_start: int,
    deviation_duration: int,
    n_seeds: int = 3,
    base_seed: int = 700,
) -> Dict[str, Any]:
    """Force a learner to deviate for ``deviation_duration`` steps and
    measure (i) the deviator's drop in payoff and (ii) the other learners'
    reaction (algorithmic punishment).

    The returned dict carries:
      * summary statistics across ``n_seeds`` runs,
      * the first-seed trajectory (used for the response plot),
      * pre-/post-deviation mean prices and per-player outputs in the window
        ``[start-W, start)`` (pre) and ``[start+duration, start+duration+W)`` (post)
        with ``W = max(50, deviation_duration)``.
    """
    if deviator not in learning_players:
        raise ValueError(
            f"deviator='{deviator}' must be in learning_players={learning_players}"
        )

    ci_list: List[float] = []
    pr_list: List[float] = []
    per_player: Dict[str, List[float]] = {p: [] for p in all_players}
    all_outcomes: List[MultiAgentRLOutcome] = []
    pre_prices: List[float] = []
    during_prices: List[float] = []
    post_prices: List[float] = []
    # Pre/during/post per-player MEAN quantities -- needed for the
    # mechanical-vs-strategic share decomposition of the price drop.
    pre_q:    Dict[str, List[float]] = {p: [] for p in all_players}
    during_q: Dict[str, List[float]] = {p: [] for p in all_players}
    post_q:   Dict[str, List[float]] = {p: [] for p in all_players}
    window = max(50, deviation_duration)

    for i in range(n_seeds):
        seed = base_seed + i
        print(f"[marl forced-deviation] seed {i + 1}/{n_seeds} (seed={seed})")
        outcome = train_multiagent_ql(
            learning_players=list(learning_players),
            all_players=list(all_players),
            demand=demand,
            costs=costs,
            capacities=capacities,
            params=params,
            nash_price=nash_price,
            cartel_price=cartel_price,
            seed=seed,
            forced_deviation={
                "player": deviator,
                "q": float(deviation_q),
                "start_step": int(deviation_start),
                "duration": int(deviation_duration),
            },
        )
        all_outcomes.append(outcome)
        ci_list.append(float(outcome.collusion_index))
        pr_list.append(float(outcome.converged_price))
        for p in all_players:
            per_player[p].append(float(outcome.converged_outputs[p]))

        prices = np.asarray(outcome.price_history, dtype=float)
        pre_lo = max(0, deviation_start - window)
        during_hi = min(len(prices), deviation_start + deviation_duration)
        post_lo = during_hi
        post_hi = min(len(prices), post_lo + window)
        if deviation_start > pre_lo:
            pre_prices.append(float(prices[pre_lo:deviation_start].mean()))
        if during_hi > deviation_start:
            during_prices.append(float(prices[deviation_start:during_hi].mean()))
        if post_hi > post_lo:
            post_prices.append(float(prices[post_lo:post_hi].mean()))

        # Per-player average quantity in each window.
        q_arrays: Dict[str, np.ndarray] = {
            p: np.asarray([q[p] for q in outcome.q_history], dtype=float)
            for p in all_players
        }
        for p in all_players:
            if deviation_start > pre_lo:
                pre_q[p].append(float(q_arrays[p][pre_lo:deviation_start].mean()))
            if during_hi > deviation_start:
                during_q[p].append(float(q_arrays[p][deviation_start:during_hi].mean()))
            if post_hi > post_lo:
                post_q[p].append(float(q_arrays[p][post_lo:post_hi].mean()))

    summary = _summarise_seed_runs(ci_list, pr_list, per_player)

    # --- Mechanical / strategic decomposition of the price drop ---------
    # Mechanical: extra supply from the deviator alone shifts the price by
    #   ΔP_mech = -b · Δq_dev    where  Δq_dev = q_during_dev - q_pre_dev
    # Strategic: any *additional* price drop coming from non-deviators
    # changing their quantities ; if Σ_{i≠dev} Δq_i > 0 (i.e. they expand)
    # the deviator suffers algorithmic punishment, otherwise they
    # absorb the deviation.
    # Decomposition uses MEAN per-window per-player quantities across
    # seeds (paired by seed → averaged once aggregated, then differenced).
    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    b = float(demand.b)
    mean_pre_p = _mean(pre_prices)
    mean_during_p = _mean(during_prices)
    obs_drop = mean_pre_p - mean_during_p  # positive when the price falls
    delta_q_dev = _mean(during_q[deviator]) - _mean(pre_q[deviator])
    delta_q_non_dev_total = sum(
        _mean(during_q[p]) - _mean(pre_q[p])
        for p in all_players if p != deviator
    )
    mech_drop = b * max(0.0, delta_q_dev)
    nondev_drop = b * delta_q_non_dev_total
    strategic_drop = max(0.0, obs_drop - mech_drop)
    if abs(obs_drop) > 1e-9:
        mech_share = mech_drop / abs(obs_drop)
        strategic_share = strategic_drop / abs(obs_drop)
    else:
        mech_share = float("nan")
        strategic_share = float("nan")

    return {
        "deviator": deviator,
        "deviation_q": float(deviation_q),
        "deviation_start": int(deviation_start),
        "deviation_duration": int(deviation_duration),
        "window": int(window),
        "mean_pre_price": mean_pre_p,
        "mean_during_price": mean_during_p,
        "mean_post_price": _mean(post_prices),
        # New per-player quantity windows (paired-mean across seeds).
        **{f"pre_q_{p}":    _mean(pre_q[p])    for p in all_players},
        **{f"during_q_{p}": _mean(during_q[p]) for p in all_players},
        **{f"post_q_{p}":   _mean(post_q[p])   for p in all_players},
        "delta_q_dev_during_minus_pre": float(delta_q_dev),
        "delta_q_non_dev_total":        float(delta_q_non_dev_total),
        "observed_price_drop":          float(obs_drop),
        "mechanical_price_drop":        float(mech_drop),
        "strategic_price_drop":         float(strategic_drop),
        "mechanical_share":             float(mech_share),
        "strategic_share":              float(strategic_share),
        "headline_outcome": all_outcomes[0] if all_outcomes else None,
        "all_outcomes": all_outcomes,
        **summary,
    }


def run_marl_capacity_experiment(
    learning_players: List[str],
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price_unconstrained: float,
    cartel_price_unconstrained: float,
    nash_price_constrained: float,
    cartel_price_constrained: float,
    n_seeds: int = 3,
    base_seed: int = 800,
) -> Dict[str, Any]:
    """Train MARL with and without capacity constraints active and compare.

    Each regime is benchmarked against its *own* Nash / cartel prices: the
    collusion index of the constrained regime uses the constrained
    benchmarks, otherwise the metric would be biased by the change in
    feasible set rather than by behavioural cooperation.
    """
    def _do(cap_enabled: bool, nash_p: float, cartel_p: float) -> Dict[str, float]:
        caps = replace(capacities, enabled=cap_enabled)
        ci_list: List[float] = []
        pr_list: List[float] = []
        per_player: Dict[str, List[float]] = {p: [] for p in all_players}
        for i in range(n_seeds):
            seed = base_seed + i + (0 if cap_enabled else 5000)
            print(
                f"[marl capacity] {'enabled' if cap_enabled else 'disabled'}"
                f" — seed {i + 1}/{n_seeds}"
            )
            outcome = train_multiagent_ql(
                learning_players=list(learning_players),
                all_players=list(all_players),
                demand=demand,
                costs=costs,
                capacities=caps,
                params=params,
                nash_price=nash_p,
                cartel_price=cartel_p,
                seed=seed,
            )
            ci_list.append(float(outcome.collusion_index))
            pr_list.append(float(outcome.converged_price))
            for p in all_players:
                per_player[p].append(float(outcome.converged_outputs[p]))
        return _summarise_seed_runs(ci_list, pr_list, per_player)

    unconstrained = _do(False, nash_price_unconstrained, cartel_price_unconstrained)
    constrained = _do(True, nash_price_constrained, cartel_price_constrained)
    return {"unconstrained": unconstrained, "constrained": constrained}


def run_marl_stackelberg_comparison(
    all_players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: MultiAgentRLParams,
    nash_price: float,
    cartel_price: float,
    static_leader_outputs: Dict[str, float],
    n_seeds: int = 3,
    base_seed: int = 900,
    *,
    static_leader_prices: Optional[Dict[str, float]] = None,
) -> List[Dict[str, float]]:
    """For each candidate leader, train MARL with that player as the sole
    learner (and the others as myopic best-responders).

    The intra-step decision order in ``train_multiagent_ql`` is
    learners → non-learners, so a sole learner is *de facto* a Stackelberg
    leader: it commits to a quantity that the followers see (within the same
    step) before they best-respond.  Comparing the learner's converged
    output to the static Stackelberg leader quantity tells us whether
    decentralised learning recovers the closed-form leadership equilibrium.

    Parameters
    ----------
    static_leader_outputs : mapping ``player → q_leader^static`` from §3 of
        the report (Stackelberg comparison CSV).
    """
    rows: List[Dict[str, float]] = []
    for leader in all_players:
        # Widen the leader's action grid so it can mechanically reach the
        # static Stackelberg quantity (which exceeds the cooperative monopoly
        # bound used by default).  We allow up to 1.5× the static leader q
        # but never below the cooperative bound.
        static_q = float(static_leader_outputs.get(leader, 0.0))
        q_max_override = {leader: max(1.5 * static_q, (demand.a - {
            "US": costs.c_us, "OPEC": costs.c_opec, "RUS": costs.c_rus,
        }[leader]) / (2 * demand.b))}

        ci_list: List[float] = []
        pr_list: List[float] = []
        learner_q_list: List[float] = []
        for i in range(n_seeds):
            seed = base_seed + i
            print(f"[marl stackelberg] leader={leader} — seed {i + 1}/{n_seeds}")
            outcome = train_multiagent_ql(
                learning_players=[leader],
                all_players=list(all_players),
                demand=demand,
                costs=costs,
                capacities=capacities,
                params=params,
                nash_price=nash_price,
                cartel_price=cartel_price,
                seed=seed,
                action_q_max_override=q_max_override,
            )
            ci_list.append(float(outcome.collusion_index))
            pr_list.append(float(outcome.converged_price))
            learner_q_list.append(float(outcome.converged_outputs[leader]))

        ci_arr = np.asarray(ci_list, dtype=float)
        pr_arr = np.asarray(pr_list, dtype=float)
        lq_arr = np.asarray(learner_q_list, dtype=float)
        static_p = (
            float(static_leader_prices.get(leader, float("nan")))
            if static_leader_prices is not None else float("nan")
        )
        rows.append({
            "leader": leader,
            "static_leader_q": float(static_leader_outputs.get(leader, float("nan"))),
            "static_leader_price": static_p,
            "marl_leader_q_mean": float(lq_arr.mean()),
            "marl_leader_q_std": float(lq_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
            "marl_converged_price_mean": float(pr_arr.mean()),
            "marl_converged_price_std": float(pr_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
            "marl_collusion_index_mean": float(ci_arr.mean()),
            "marl_collusion_index_std": float(ci_arr.std(ddof=1)) if n_seeds > 1 else 0.0,
            "n_seeds": int(n_seeds),
        })
    return rows


def policy_heatmap(
    outcome: MultiAgentRLOutcome,
    player: str,
    metric: str = "argmax",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce an agent's Q-table to a 2-D matrix indexed by (price, own_q).

    Two reductions are supported:

      * ``metric='argmax'`` — the *chosen* action quantity in each state.
        Rows correspond to discretised market-price bins (high → low) and
        columns to the agent's previous-output bins.  Cells are absolute
        outputs (mbd) read from the agent's own action grid.
      * ``metric='value'`` — the state value ``max_a Q(s, a)``.

    Returns ``(matrix, price_bin_centres, own_q_bin_centres)``.
    """
    q = outcome.q_tables[player]  # shape (price_bins+1, own_q_bins+1, n_actions)
    grid = outcome.action_grids[player]
    if metric == "argmax":
        idx = np.argmax(q, axis=2)
        mat = grid[idx]
    elif metric == "value":
        mat = np.max(q, axis=2)
    else:
        raise ValueError(f"unknown metric: {metric}")

    n_pbins, n_qbins, _ = q.shape
    # Approximate bin centres along [0, max_price] and [0, max_q].
    price_centres = np.linspace(0.0, 100.0, n_pbins)
    q_centres = np.linspace(0.0, float(grid[-1]), n_qbins)
    return mat, price_centres, q_centres
