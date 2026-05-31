"""Evolutionary Game Dynamics for the global crude oil market.

Models a population of oil producers (e.g. OPEC member states) choosing among
production strategies. Uses replicator dynamics — the biological metaphor for
cultural imitation in economics — to track how strategy shares evolve over time
as producers copy more successful peers.

Three strategies
----------------
C – Cooperate : produce at the joint-profit-maximising (cartel) quota
D – Defect    : produce at the Cournot-Nash best response (over-quota)
P – Punish    : produce aggressively (> Nash) to deter or penalise defectors
                (analogous to Saudi Arabia's 1986 'netback pricing' retaliation)

Payoff matrix entries are derived analytically from the calibrated Cournot model,
ensuring full consistency with the non-cooperative game results.

Replicator dynamics
-------------------
    ẋ_s = x_s · (f_s(x) − f̄(x))

where  x_s   = share of strategy s in the population,
       f_s   = average payoff of s given current population mix x,
       f̄    = Σ_s x_s f_s  (mean population fitness).

Phase diagrams
--------------
- 2-strategy case:  1-D portrait of ẋ as a function of x ∈ [0,1].
- 3-strategy case:  trajectories projected onto the 2-D barycentric simplex
                    (equilateral triangle with vertices C, D, P).

References
----------
- Weibull, J.W. (1995). Evolutionary Game Theory. MIT Press.
- Friedman, D. (1991). Evolutionary games in economics. Econometrica 59(3).
- Hofbauer, J. & Sigmund, K. (1998). Evolutionary Games and Population Dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .config import CapacityParams, CostParams, DemandParams, EvoGameParams
from .cooperation_punishment import cartel_quotas
from .cournot_repeated import best_response, AdjustmentCostParams
from .cournot_static import _cost_map, cournot_equilibrium
from .demand import price_from_quantity

STRATEGIES_2 = ["C", "D"]
STRATEGIES_3 = ["C", "D", "P"]

PUNISHMENT_SWEEP_MULTIPLIERS = [1.05, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]


# ---------------------------------------------------------------------------
# Payoff matrix dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PayoffMatrix:
    """Payoff matrix for an evolutionary game derived from the Cournot model.

    A[i, j] = payoff (per-period profit) to a player using strategy i
               when matched against a player (or population share) using strategy j.
    """
    strategies: List[str]
    A: np.ndarray
    descriptions: Dict[str, str]
    profit_labels: Dict[str, float]    # named scalar profits for reporting


@dataclass
class EvoOutcome:
    """Full output of the evolutionary dynamics simulation."""

    # 2-strategy (C vs D)
    payoff_2: PayoffMatrix
    trajectories_2: List[np.ndarray]        # each (T+1, 2)
    initial_conditions_2: np.ndarray        # (n_traj,) initial x_C values
    interior_eq_2: Optional[float]          # x* in (0,1), or None
    ess_labels_2: List[str]                 # human-readable ESS description
    phase_x: np.ndarray                     # x_C grid for 1-D portrait
    phase_dx: np.ndarray                    # ẋ_C evaluated on grid

    # 3-strategy (C, D, P)
    payoff_3: PayoffMatrix
    trajectories_3: List[np.ndarray]        # each (T+1, 3)
    initial_conditions_3: np.ndarray        # (n_traj, 3) on simplex

    t_grid: np.ndarray
    params: EvoGameParams
    focal_player: str


# ---------------------------------------------------------------------------
# Payoff matrix construction
# ---------------------------------------------------------------------------

def _build_payoff_2x2(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    focal_player: str = "OPEC",
) -> PayoffMatrix:
    """Build the 2×2 payoff matrix for the Cooperate / Defect game.

    Cooperation uses **proportional cartel quotas** (each player cuts
    output by the same percentage from Nash), not the joint-profit-max
    that concentrates production on the cheapest player.  This ensures
    a meaningful Prisoner's Dilemma structure.

    Economic intuition
    ------------------
    (C, C): both produce at cartel quota → high price → π_coop
    (D, D): both produce at Nash level   → Nash price → π_nash  (< π_coop)
    (C, D): focal cooperates; rival defects to Nash BR → focal earns π_sucker  (< π_nash)
    (D, C): focal defects; rival cooperates            → focal earns π_dev     (> π_coop)

    This is a Prisoner's Dilemma in the typical oil-market calibration:
        π_dev > π_coop > π_nash > π_sucker
    Defection is individually rational (dominant strategy) but collectively
    sub-optimal — the core tension in cartel economics.
    """
    cost_map = _cost_map(costs)
    adj = AdjustmentCostParams(enabled=False, k=0.0)

    nash_res = cournot_equilibrium(players, demand, costs, capacities)
    pi_nash = nash_res.profits[focal_player]

    cartel = cartel_quotas(players, demand, costs, capacities)
    q_coop_focal = cartel.quotas[focal_player]
    pi_coop = cartel.quota_profits[focal_player]

    # Deviation profit: focal best-responds while all others stay at quota
    q_others_coop = sum(cartel.quotas[p] for p in players if p != focal_player)
    q_dev = best_response(
        q_others_coop, q_coop_focal, demand, cost_map[focal_player],
        capacities, focal_player, adj,
    )
    p_dev = price_from_quantity(q_dev + q_others_coop, demand)
    pi_dev = (p_dev - cost_map[focal_player]) * q_dev

    # Sucker payoff: focal stays at quota while rivals revert to Nash
    q_others_nash = sum(nash_res.quantities[p] for p in players if p != focal_player)
    p_sucker = price_from_quantity(q_coop_focal + q_others_nash, demand)
    pi_sucker = max(0.0, (p_sucker - cost_map[focal_player]) * q_coop_focal)

    A = np.array([
        [pi_coop,  pi_sucker],   # row C: payoffs vs (C, D)
        [pi_dev,   pi_nash  ],   # row D: payoffs vs (C, D)
    ])

    return PayoffMatrix(
        strategies=["C", "D"],
        A=A,
        descriptions={
            "C": "Cooperate — produce at proportional cartel quota",
            "D": "Defect   — produce at Cournot-Nash best response (over-quota)",
        },
        profit_labels={
            "pi_coop":   round(pi_coop,   2),
            "pi_dev":    round(pi_dev,    2),
            "pi_nash":   round(pi_nash,   2),
            "pi_sucker": round(pi_sucker, 2),
        },
    )


def _build_payoff_3x3(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    focal_player: str = "OPEC",
    punishment_multiplier: float = 1.6,
    conditional: bool = True,
) -> PayoffMatrix:
    """Build the 3×3 payoff matrix (C, D, P).

    Cooperation uses **proportional cartel quotas** (same as 2×2).

    P = Punish: produce at punishment_multiplier × Nash quantity against D,
    but cooperate normally with C and P (if conditional=True).

    Two punishment models:
    - **Conditional** (default, Tit-for-Tat-like): P cooperates with C & P,
      punishes only D.  Models OPEC's actual behaviour: maintain quotas with
      compliant members, retaliate against cheaters.
    - **Unconditional**: P always floods the market regardless of opponent.
      This is the 'nuclear option' (Saudi 1986 / 2020 price war).

    With conditional punishment the key dynamics on the simplex are:
    - D invades C  (sucker payoff temptation, as in PD)
    - P invades D  (punishers deter defectors — P vs D, while D vs P crashes)
    - C cannot invade P  (they earn the same against all opponents → neutral)
    → Cooperation can be sustained through credible conditional punishment.
    """
    cost_map = _cost_map(costs)
    adj = AdjustmentCostParams(enabled=False, k=0.0)

    nash_res = cournot_equilibrium(players, demand, costs, capacities)
    cartel = cartel_quotas(players, demand, costs, capacities)

    q_nash_focal = nash_res.quantities[focal_player]
    q_coop_focal = cartel.quotas[focal_player]
    q_others_coop = sum(cartel.quotas[p] for p in players if p != focal_player)
    q_others_nash = sum(nash_res.quantities[p] for p in players if p != focal_player)

    pi_coop = cartel.quota_profits[focal_player]
    pi_nash = nash_res.profits[focal_player]

    q_dev = best_response(q_others_coop, q_coop_focal, demand, cost_map[focal_player], capacities, focal_player, adj)
    p_dev = price_from_quantity(q_dev + q_others_coop, demand)
    pi_dev = (p_dev - cost_map[focal_player]) * q_dev

    p_sucker = price_from_quantity(q_coop_focal + q_others_nash, demand)
    pi_sucker = max(0.0, (p_sucker - cost_map[focal_player]) * q_coop_focal)

    # Punishment quantity (capped at physical maximum)
    q_punish = min(q_nash_focal * punishment_multiplier, demand.a / demand.b * 0.6)

    # Row P: payoffs when focal player punishes (overproduces)
    # vs C: punisher + cooperating rivals
    p_P_vs_C = price_from_quantity(q_punish + q_others_coop, demand)
    pi_P_vs_C = max(0.0, (p_P_vs_C - cost_map[focal_player]) * q_punish)

    # vs D: punisher + defecting rivals (both high output → price crash)
    p_P_vs_D = price_from_quantity(q_punish + q_others_nash, demand)
    pi_P_vs_D = max(0.0, (p_P_vs_D - cost_map[focal_player]) * q_punish)

    # vs P: both punishing (assume symmetric: all rivals also punish)
    p_P_vs_P = price_from_quantity(q_punish * len(players), demand)
    pi_P_vs_P = max(0.0, (p_P_vs_P - cost_map[focal_player]) * q_punish)

    # Row C vs P: focal cooperates while rivals punish (bad: price is low)
    p_C_vs_P = price_from_quantity(q_coop_focal + q_punish * (len(players) - 1), demand)
    pi_C_vs_P = max(0.0, (p_C_vs_P - cost_map[focal_player]) * q_coop_focal)

    # Row D vs P: focal defects while rivals punish
    p_D_vs_P = price_from_quantity(q_nash_focal + q_punish * (len(players) - 1), demand)
    pi_D_vs_P = max(0.0, (p_D_vs_P - cost_map[focal_player]) * q_nash_focal)

    A_unconditional = np.array([
        [pi_coop,   pi_sucker,  pi_C_vs_P],   # row C: vs (C, D, P)
        [pi_dev,    pi_nash,    pi_D_vs_P ],   # row D: vs (C, D, P)
        [pi_P_vs_C, pi_P_vs_D,  pi_P_vs_P],   # row P: vs (C, D, P)
    ])

    # Conditional punishment (Tit-for-Tat-like): P cooperates with C and P,
    # but floods market against D.  This models OPEC's actual behaviour:
    # maintain quotas with compliant members, retaliate against cheaters.
    A_conditional = np.array([
        [pi_coop,   pi_sucker,  pi_coop  ],   # C: cooperate with everyone
        [pi_dev,    pi_nash,    pi_D_vs_P],   # D: defect against everyone
        [pi_coop,   pi_P_vs_D,  pi_coop  ],   # P: cooperate with C & P, punish D
    ])

    A = A_conditional if conditional else A_unconditional

    return PayoffMatrix(
        strategies=["C", "D", "P"],
        A=A,
        descriptions={
            "C": "Cooperate — produce at cartel quota",
            "D": "Defect   — produce at Nash best response (over-quota)",
            "P": ("Punish (conditional) — cooperate with C & P, flood market vs D"
                  if conditional else
                  "Punish (unconditional) — flood market regardless of opponent"),
        },
        profit_labels={
            "pi_coop":   round(pi_coop,   2),
            "pi_dev":    round(pi_dev,    2),
            "pi_nash":   round(pi_nash,   2),
            "pi_sucker": round(pi_sucker, 2),
            "pi_P_vs_C": round(pi_P_vs_C, 2),
            "pi_P_vs_D": round(pi_P_vs_D, 2),
        },
    )


# ---------------------------------------------------------------------------
# Replicator dynamics
# ---------------------------------------------------------------------------

def _replicator_rhs(t: float, x: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Replicator ODE right-hand side:  ẋ_i = x_i · (f_i(x) − f̄(x))."""
    f = A @ x                       # fitness of each strategy
    f_bar = float(x @ f)            # mean population fitness
    return x * (f - f_bar)


def simulate_replicator(
    x0: np.ndarray,
    payoff: PayoffMatrix,
    T: int = 400,
    dt: float = 0.05,
) -> np.ndarray:
    """Integrate replicator dynamics from initial condition x0.

    Returns array of shape (n_steps+1, n_strategies) with strategy shares
    clamped to the simplex at each output step.
    """
    t_end = T * dt
    t_eval = np.linspace(0.0, t_end, T + 1)
    sol = solve_ivp(
        _replicator_rhs,
        (0.0, t_end),
        x0,
        t_eval=t_eval,
        args=(payoff.A,),
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    traj = sol.y.T                          # (T+1, n_strat)
    traj = np.clip(traj, 0.0, None)
    row_sums = traj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    return traj / row_sums


# ---------------------------------------------------------------------------
# Equilibrium analysis
# ---------------------------------------------------------------------------

def _interior_equilibrium_2strategy(A: np.ndarray) -> Optional[float]:
    """Analytically find interior fixed point x* ∈ (0,1) for 2-strategy game.

    Condition: f_C(x*) = f_D(x*)
    → x*(A[0,0] − A[1,0]) + (1−x*)(A[0,1] − A[1,1]) = 0
    """
    denom = (A[0, 0] - A[1, 0]) - (A[0, 1] - A[1, 1])
    if abs(denom) < 1e-10:
        return None
    x_star = (A[1, 1] - A[0, 1]) / denom
    return float(x_star) if 0.0 < x_star < 1.0 else None


def _ess_labels_2strategy(A: np.ndarray, x_star: Optional[float]) -> List[str]:
    """Return human-readable ESS labels for the 2-strategy game."""
    labels = []
    # x=1 (all C) is ESS if f_C > f_D at x=1 → A[0,0] > A[1,0]
    if A[0, 0] > A[1, 0]:
        labels.append("x=1 (All Cooperate) — ESS ✓")
    else:
        labels.append("x=1 (All Cooperate) — unstable ✗")
    # x=0 (all D) is ESS if f_D > f_C at x=0 → A[1,1] > A[0,1]
    if A[1, 1] > A[0, 1]:
        labels.append("x=0 (All Defect) — ESS ✓")
    else:
        labels.append("x=0 (All Defect) — unstable ✗")
    if x_star is not None:
        # Interior point: stable if ẋ > 0 below and < 0 above
        eps = 1e-4
        x_hi = np.array([x_star + eps, 1 - (x_star + eps)])
        f_hi = A @ x_hi
        f_hi_bar = float(x_hi @ f_hi)
        dx_hi = x_hi[0] * (f_hi[0] - f_hi_bar)
        stable = (dx_hi < 0)
        tag = "stable" if stable else "unstable"
        labels.append(f"Interior x*={x_star:.3f} — {tag}")
    return labels


# ---------------------------------------------------------------------------
# Phase diagram helpers
# ---------------------------------------------------------------------------

def _phase_diagram_2strategy(
    payoff: PayoffMatrix,
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ẋ_C over a grid x_C ∈ [0,1] for the 1-D phase portrait."""
    x_grid = np.linspace(0.0, 1.0, n_grid)
    dx_grid = np.zeros(n_grid)
    for i, xc in enumerate(x_grid):
        xv = np.array([xc, 1.0 - xc])
        f = payoff.A @ xv
        f_bar = float(xv @ f)
        dx_grid[i] = xc * (f[0] - f_bar)
    return x_grid, dx_grid


def _random_simplex_points(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n random points on the k-dimensional simplex."""
    return rng.dirichlet(np.ones(k), size=n)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_evolutionary_game(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: EvoGameParams,
    focal_player: str = "OPEC",
    seed: int = 42,
) -> EvoOutcome:
    """Run evolutionary game dynamics analysis (2-strategy and 3-strategy).

    Parameters
    ----------
    players : full list of market players.
    demand, costs, capacities : calibrated model parameters.
    params : EvoGameParams controlling simulation resolution and length.
    focal_player : the player whose payoffs define the evolutionary game
                   (default: OPEC, since we model OPEC-member decisions).
    seed : random seed for reproducible initial conditions.

    Returns
    -------
    EvoOutcome with payoff matrices, trajectories, equilibria, and phase data.
    """
    rng = np.random.default_rng(seed)

    # --- Build payoff matrices ---
    payoff_2 = _build_payoff_2x2(players, demand, costs, capacities, focal_player)
    payoff_3 = _build_payoff_3x3(players, demand, costs, capacities, focal_player)

    t_grid = np.linspace(0.0, params.T * params.dt, params.T + 1)

    # --- 2-strategy trajectories ---
    ic_2 = np.linspace(0.05, 0.95, params.n_trajectories)
    trajs_2 = []
    for xc in ic_2:
        x0 = np.array([xc, 1.0 - xc])
        trajs_2.append(simulate_replicator(x0, payoff_2, params.T, params.dt))

    interior_eq = _interior_equilibrium_2strategy(payoff_2.A)
    ess_labels = _ess_labels_2strategy(payoff_2.A, interior_eq)
    phase_x, phase_dx = _phase_diagram_2strategy(payoff_2, n_grid=params.grid_resolution * 4)

    # --- 3-strategy trajectories (random ICs on simplex) ---
    ic_3 = _random_simplex_points(params.n_trajectories, 3, rng)
    # Add corners and edge midpoints for completeness
    corners = np.eye(3)
    edge_mids = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    center = np.array([[1/3, 1/3, 1/3]])
    ic_3 = np.vstack([ic_3, corners, edge_mids, center])
    ic_3 = np.clip(ic_3, 1e-4, 1 - 1e-4)
    ic_3 /= ic_3.sum(axis=1, keepdims=True)

    trajs_3 = []
    for x0 in ic_3:
        trajs_3.append(simulate_replicator(x0, payoff_3, params.T, params.dt))

    return EvoOutcome(
        payoff_2=payoff_2,
        trajectories_2=trajs_2,
        initial_conditions_2=ic_2,
        interior_eq_2=interior_eq,
        ess_labels_2=ess_labels,
        phase_x=phase_x,
        phase_dx=phase_dx,
        payoff_3=payoff_3,
        trajectories_3=trajs_3,
        initial_conditions_3=ic_3,
        t_grid=t_grid,
        params=params,
        focal_player=focal_player,
    )


# ---------------------------------------------------------------------------
# Punishment multiplier sweep
# ---------------------------------------------------------------------------

@dataclass
class PunishmentSweepResult:
    """Outcome of sweeping the punishment severity for the 3-strategy game."""
    multipliers: List[float]
    payoff_matrices: List[np.ndarray]
    convergence: List[np.ndarray]          # final strategy shares per multiplier
    dominant_strategy: List[str]
    has_interior_eq: List[bool]
    mean_payoff_trajectory: List[np.ndarray]  # mean pop payoff over time
    conditional: bool = True


def punishment_multiplier_sweep(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    params: EvoGameParams,
    focal_player: str = "OPEC",
    multipliers: Optional[List[float]] = None,
    conditional: bool = True,
    seed: int = 42,
) -> PunishmentSweepResult:
    """Sweep punishment severity and track how 3-strategy dynamics change.

    For each multiplier, builds the 3x3 payoff matrix, runs replicator
    dynamics from a balanced initial condition, and records where the
    population converges.  This reveals the critical punishment level
    at which the Punish strategy becomes viable enough to deter Defect
    and potentially sustain Cooperate.

    Parameters
    ----------
    conditional : if True, use conditional (Tit-for-Tat-like) punishment;
                  if False, unconditional (always flood).
    """
    if multipliers is None:
        multipliers = PUNISHMENT_SWEEP_MULTIPLIERS

    x0_balanced = np.array([1/3, 1/3, 1/3])

    all_matrices: List[np.ndarray] = []
    all_convergence: List[np.ndarray] = []
    all_dominant: List[str] = []
    all_interior: List[bool] = []
    all_mean_payoff: List[np.ndarray] = []

    for mult in multipliers:
        payoff = _build_payoff_3x3(
            players, demand, costs, capacities,
            focal_player=focal_player,
            punishment_multiplier=mult,
            conditional=conditional,
        )
        all_matrices.append(payoff.A.copy())

        traj = simulate_replicator(x0_balanced, payoff, params.T, params.dt)
        final = traj[-1]
        all_convergence.append(final)

        dom_idx = int(np.argmax(final))
        all_dominant.append(STRATEGIES_3[dom_idx])

        has_int = bool(np.min(final) > 0.05)
        all_interior.append(has_int)

        mean_payoff = np.array([
            float(traj[t] @ payoff.A @ traj[t]) for t in range(traj.shape[0])
        ])
        all_mean_payoff.append(mean_payoff)

    return PunishmentSweepResult(
        multipliers=multipliers,
        payoff_matrices=all_matrices,
        convergence=all_convergence,
        dominant_strategy=all_dominant,
        has_interior_eq=all_interior,
        mean_payoff_trajectory=all_mean_payoff,
        conditional=conditional,
    )
