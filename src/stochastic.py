"""Stochastic demand simulation for the global crude oil market.

Three stochastic specifications are implemented:

1. Pure AR(1) demand shock (baseline):
       ε_t = ρ · ε_{t-1} + η_t,   η_t ~ N(0, σ²)
       P(Q, t) = (a + ε_t) − b · Q
   This captures persistent demand cycles (business cycles, energy transition).

2. Jump-diffusion (extended, more realistic):
       ε_t = ρ · ε_{t-1} + η_t + J_t
       J_t = N(μ_J, σ_J²) with probability λ_J per period, else 0
   The Poisson jump component captures sudden supply-side disruptions.

3. Green-Porter (1984) repeated game under imperfect monitoring:
       Players attempt cartel cooperation; when the observed price drops
       below a trigger level p_bar (due to a demand shock OR a deviation),
       all players revert to Nash for L periods (punishment).  This bridges
       the deterministic Folk Theorem (Section 5) and the evolutionary
       result that Punish is the ESS (Section 9).

Monte Carlo paths are used to compute:
- Price and quantity confidence bands across T periods.
- Cooperation survival fraction and false-trigger rates (Green-Porter).
- Stochastic critical discount factor δ* and convergence after shocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .config import (
    CapacityParams, CostParams, DemandParams, GreenPorterParams,
    SimulationParams, StochasticParams,
)
from .cooperation_punishment import cartel_quotas
from .cournot_static import cournot_equilibrium, CournotResult, _cost_map, _cap_map
from .demand import price_from_quantity


@dataclass
class StochasticOutcome:
    """Price and quantity paths across Monte Carlo replications."""
    price_paths: np.ndarray      # shape (n_paths, T)
    quantity_paths: np.ndarray   # shape (n_paths, T)
    player_paths: Dict[str, np.ndarray]   # each (n_paths, T)
    price_mean: np.ndarray       # shape (T,)
    price_p10: np.ndarray
    price_p25: np.ndarray
    price_p75: np.ndarray
    price_p90: np.ndarray
    players: List[str]
    sigma: float
    rho: float


def simulate_stochastic_cournot(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    T: int = 50,
    n_paths: int = 300,
    sigma: float = 8.0,
    rho: float = 0.6,
    seed: int = 42,
) -> StochasticOutcome:
    """Run Monte Carlo Cournot simulations with AR(1) demand shocks.

    At each period t, the demand intercept becomes a_t = a + ε_t.
    Players solve the static Cournot problem under the realised a_t.
    This corresponds to a 'flexible' equilibrium where producers observe
    the shock before choosing quantities each period.
    """
    rng = np.random.default_rng(seed)
    cost_map = _cost_map(costs)
    cap_map = _cap_map(capacities)

    price_paths = np.zeros((n_paths, T))
    quantity_paths = np.zeros((n_paths, T))
    player_paths: Dict[str, np.ndarray] = {p: np.zeros((n_paths, T)) for p in players}

    for path in range(n_paths):
        eps = 0.0
        for t in range(T):
            eta = rng.normal(0.0, sigma)
            eps = rho * eps + eta
            a_t = demand.a + eps

            shocked_demand = DemandParams(
                a=a_t, b=demand.b, price_floor=demand.price_floor
            )
            try:
                result = cournot_equilibrium(players, shocked_demand, costs, capacities)
                price_paths[path, t] = result.price
                quantity_paths[path, t] = result.total_quantity
                for p in players:
                    player_paths[p][path, t] = result.quantities[p]
            except Exception:
                # propagate last valid observation on numerical failure
                price_paths[path, t] = price_paths[path, t - 1] if t > 0 else demand.a
                quantity_paths[path, t] = quantity_paths[path, t - 1] if t > 0 else 0.0

    return StochasticOutcome(
        price_paths=price_paths,
        quantity_paths=quantity_paths,
        player_paths=player_paths,
        price_mean=price_paths.mean(axis=0),
        price_p10=np.percentile(price_paths, 10, axis=0),
        price_p25=np.percentile(price_paths, 25, axis=0),
        price_p75=np.percentile(price_paths, 75, axis=0),
        price_p90=np.percentile(price_paths, 90, axis=0),
        players=players,
        sigma=sigma,
        rho=rho,
    )


def simulate_stochastic_jump_diffusion(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    T: int = 50,
    n_paths: int = 300,
    sigma: float = 8.0,
    rho: float = 0.6,
    jump_intensity: float = 0.08,
    jump_mu: float = 0.0,
    jump_sigma: float = 18.0,
    seed: int = 42,
) -> StochasticOutcome:
    """Monte Carlo Cournot with AR(1) demand shocks **and** Poisson supply jumps.

    At each period t:
        η_t    ~ N(0, σ²)            — continuous demand drift
        N_t    ~ Bernoulli(λ_J)      — jump arrival indicator
        J_t    ~ N(μ_J, σ_J²) · N_t — jump size (conditional on arrival)
        ε_t    = ρ · ε_{t-1} + η_t + J_t
        a_t    = a + ε_t

    This produces price paths with the spikes and crashes characteristic
    of real oil markets (cf. Brent 1986 crash, 2008 spike, 2020 COVID collapse).

    Parameters
    ----------
    jump_intensity : λ_J, expected number of jumps per period.
                     0.08 ≈ 1 jump per year (quarterly time step).
    jump_mu        : mean jump size in $/bbl. 0 = symmetric; negative = net bearish.
    jump_sigma     : std dev of jump size in $/bbl.
    """
    rng = np.random.default_rng(seed)
    cost_map = _cost_map(costs)
    cap_map = _cap_map(capacities)

    price_paths = np.zeros((n_paths, T))
    quantity_paths = np.zeros((n_paths, T))
    player_paths: Dict[str, np.ndarray] = {p: np.zeros((n_paths, T)) for p in players}

    for path in range(n_paths):
        eps = 0.0
        for t in range(T):
            # Continuous AR(1) component
            eta = rng.normal(0.0, sigma)
            # Poisson jump component
            n_jumps = rng.poisson(jump_intensity)
            jump = 0.0
            if n_jumps > 0:
                jump = sum(rng.normal(jump_mu, jump_sigma) for _ in range(n_jumps))
            eps = rho * eps + eta + jump
            a_t = demand.a + eps

            shocked_demand = DemandParams(a=a_t, b=demand.b, price_floor=demand.price_floor)
            try:
                result = cournot_equilibrium(players, shocked_demand, costs, capacities)
                price_paths[path, t] = result.price
                quantity_paths[path, t] = result.total_quantity
                for p in players:
                    player_paths[p][path, t] = result.quantities[p]
            except Exception:
                price_paths[path, t] = price_paths[path, t - 1] if t > 0 else demand.a
                quantity_paths[path, t] = quantity_paths[path, t - 1] if t > 0 else 0.0

    return StochasticOutcome(
        price_paths=price_paths,
        quantity_paths=quantity_paths,
        player_paths=player_paths,
        price_mean=price_paths.mean(axis=0),
        price_p10=np.percentile(price_paths, 10, axis=0),
        price_p25=np.percentile(price_paths, 25, axis=0),
        price_p75=np.percentile(price_paths, 75, axis=0),
        price_p90=np.percentile(price_paths, 90, axis=0),
        players=players,
        sigma=sigma,
        rho=rho,
    )


@dataclass
class ScenarioShockResult:
    """Outcome of a one-off demand shock (e.g. COVID demand collapse)."""
    label: str
    shock_size: float
    price_before: float
    price_after: float
    quantities_before: Dict[str, float]
    quantities_after: Dict[str, float]
    profit_before: Dict[str, float]
    profit_after: Dict[str, float]
    price_change_pct: float


def demand_shock_scenario(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    shock_sizes: List[float],
) -> List[ScenarioShockResult]:
    """Compare equilibrium before and after permanent demand intercept shocks.

    Parameters
    ----------
    shock_sizes:
        List of shock magnitudes Δa (negative = demand contraction).
        E.g. [-20, -10, 0, +10, +20].
    """
    baseline = cournot_equilibrium(players, demand, costs, capacities)
    cost_map = _cost_map(costs)
    results: List[ScenarioShockResult] = []

    for delta_a in shock_sizes:
        shocked = DemandParams(
            a=demand.a + delta_a,
            b=demand.b,
            price_floor=demand.price_floor,
        )
        try:
            shocked_eq = cournot_equilibrium(players, shocked, costs, capacities)
        except Exception:
            continue

        p_chg = (shocked_eq.price - baseline.price) / baseline.price * 100 if baseline.price > 0 else 0.0
        label = f"Δa={delta_a:+.0f}"
        results.append(ScenarioShockResult(
            label=label,
            shock_size=delta_a,
            price_before=baseline.price,
            price_after=shocked_eq.price,
            quantities_before=dict(baseline.quantities),
            quantities_after=dict(shocked_eq.quantities),
            profit_before=dict(baseline.profits),
            profit_after=dict(shocked_eq.profits),
            price_change_pct=round(p_chg, 2),
        ))

    return results


# ---------------------------------------------------------------------------
# Green-Porter (1984) — repeated game under imperfect public monitoring
# ---------------------------------------------------------------------------

REGIME_COOP = 0
REGIME_PUNISH = 1


@dataclass
class GreenPorterOutcome:
    """Full output of the Green-Porter Monte Carlo simulation."""

    price_paths: np.ndarray             # (n_paths, T)
    regime_paths: np.ndarray            # (n_paths, T)  — 0=coop, 1=punishment
    player_quantity_paths: Dict[str, np.ndarray]  # each (n_paths, T)
    coop_fraction: np.ndarray           # (T,) fraction of paths in coop at each t
    false_trigger_count: int            # punishment episodes not preceded by deviation
    total_trigger_count: int
    mean_coop_spell: float
    mean_punish_spell: float
    expected_discounted_profit: Dict[str, float]
    trigger_price: float
    nash_price: float
    coop_price: float
    players: List[str]
    T: int
    n_paths: int


@dataclass
class GreenPorterDeltaResult:
    """Stochastic δ* from the Green-Porter IC constraint."""

    delta_star_deterministic: Dict[str, float]
    delta_star_stochastic: float
    alpha: float                        # false-trigger probability per period
    binding_player: str
    delta_actual: float
    cooperation_sustainable: bool
    V_coop: float
    V_deviate: float


@dataclass
class ConvergenceAfterShock:
    """Conditional price/quantity paths following a large shock."""

    mean_price_path: np.ndarray         # (horizon,) mean price after shock
    std_price_path: np.ndarray
    mean_quantity_path: np.ndarray
    n_events: int
    half_life_periods: float
    coop_price: float
    nash_price: float
    horizon: int


def simulate_green_porter(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    stoch: StochasticParams,
    gp: GreenPorterParams,
    delta: float = 0.95,
    use_jumps: bool = False,
    seed: int = 42,
) -> GreenPorterOutcome:
    """Monte Carlo simulation of the Green-Porter trigger-price strategy.

    Each path: players cooperate (cartel quotas) until the observed market
    price dips below p_bar, then switch to Nash for L periods.  The trigger
    fires on the *observable price*, which mixes output and demand shocks —
    this is the imperfect-monitoring channel.
    """
    rng = np.random.default_rng(seed)
    n, T = gp.n_paths, gp.T
    cost_map = _cost_map(costs)

    nash = cournot_equilibrium(players, demand, costs, capacities)
    cartel = cartel_quotas(players, demand, costs, capacities)
    p_nash = nash.price
    p_coop = cartel.quota_price
    p_bar = p_coop - gp.trigger_price_offset

    price_paths = np.zeros((n, T))
    regime_paths = np.zeros((n, T), dtype=np.int8)
    player_q = {p: np.zeros((n, T)) for p in players}
    disc_profit = {p: np.zeros(n) for p in players}

    total_triggers = 0
    false_triggers = 0
    coop_spells: List[int] = []
    punish_spells: List[int] = []

    for path_i in range(n):
        eps = 0.0
        regime = REGIME_COOP
        punish_left = 0
        spell_len = 0

        for t in range(T):
            eta = rng.normal(0.0, stoch.sigma)
            jump = 0.0
            if use_jumps:
                nj = rng.poisson(stoch.jump_intensity)
                if nj > 0:
                    jump = rng.normal(stoch.jump_mu, stoch.jump_sigma) * nj
            eps = stoch.rho * eps + eta + jump
            a_t = demand.a + eps

            shocked_demand = DemandParams(a=a_t, b=demand.b,
                                          price_floor=demand.price_floor)

            if regime == REGIME_COOP:
                try:
                    cartel_t = cartel_quotas(players, shocked_demand, costs,
                                            capacities)
                    quantities = dict(cartel_t.quotas)
                except Exception:
                    quantities = dict(cartel.quotas)
            else:
                try:
                    nash_t = cournot_equilibrium(players, shocked_demand, costs,
                                                capacities)
                    quantities = dict(nash_t.quantities)
                except Exception:
                    quantities = dict(nash.quantities)

            Q = sum(quantities.values())
            price = price_from_quantity(Q, shocked_demand)
            price_paths[path_i, t] = price
            regime_paths[path_i, t] = regime
            for p in players:
                player_q[p][path_i, t] = quantities[p]
                disc_profit[p][path_i] += (delta ** t) * (
                    (price - cost_map[p]) * quantities[p]
                )

            spell_len += 1

            if regime == REGIME_COOP and price < p_bar:
                coop_spells.append(spell_len)
                spell_len = 0
                regime = REGIME_PUNISH
                punish_left = gp.punishment_length
                total_triggers += 1
                false_triggers += 1
            elif regime == REGIME_PUNISH:
                punish_left -= 1
                if punish_left <= 0:
                    punish_spells.append(spell_len)
                    spell_len = 0
                    regime = REGIME_COOP

        if spell_len > 0:
            (coop_spells if regime == REGIME_COOP else punish_spells).append(
                spell_len)

    coop_fraction = (regime_paths == REGIME_COOP).mean(axis=0)

    return GreenPorterOutcome(
        price_paths=price_paths,
        regime_paths=regime_paths,
        player_quantity_paths=player_q,
        coop_fraction=coop_fraction,
        false_trigger_count=false_triggers,
        total_trigger_count=total_triggers,
        mean_coop_spell=float(np.mean(coop_spells)) if coop_spells else float(T),
        mean_punish_spell=float(np.mean(punish_spells)) if punish_spells else 0.0,
        expected_discounted_profit={p: float(disc_profit[p].mean()) for p in players},
        trigger_price=p_bar,
        nash_price=p_nash,
        coop_price=p_coop,
        players=players,
        T=T,
        n_paths=n,
    )


def green_porter_delta_star(
    players: List[str],
    demand: DemandParams,
    costs: CostParams,
    capacities: CapacityParams,
    stoch: StochasticParams,
    gp: GreenPorterParams,
    delta_actual: float = 0.95,
) -> GreenPorterDeltaResult:
    """Compute the stochastic δ* under Green-Porter imperfect monitoring.

    Uses the stationary distribution of the AR(1) shock to compute α
    (probability of a false punishment trigger per cooperative period),
    then solves the IC constraint numerically.
    """
    from .coalition import folk_theorem_delta_star
    from .cournot_repeated import best_response, AdjustmentCostParams

    nash = cournot_equilibrium(players, demand, costs, capacities)
    cartel = cartel_quotas(players, demand, costs, capacities)
    cost_map = _cost_map(costs)
    adj = AdjustmentCostParams(enabled=False, k=0.0)

    p_nash = nash.price
    p_coop = cartel.quota_price
    p_bar = p_coop - gp.trigger_price_offset
    L = gp.punishment_length

    sigma_eps = stoch.sigma / np.sqrt(1.0 - stoch.rho ** 2)
    alpha = float(norm.cdf(p_bar - p_coop, loc=0.0, scale=sigma_eps))

    det = folk_theorem_delta_star(players, demand, costs, capacities, delta_actual)

    binding = det.binding_player
    pi_coop = det.pi_cooperative[binding]
    pi_nash = det.pi_nash[binding]
    pi_dev = det.pi_deviation[binding]

    def _ic_gap(d: float) -> float:
        if d <= 0 or d >= 1:
            return -1.0
        dL = d ** L
        v_punish_flow = pi_nash * (1.0 - dL) / (1.0 - d)
        denom = 1.0 - d * (1.0 - alpha) - alpha * d * dL
        if abs(denom) < 1e-15:
            return -1.0
        v_coop = (pi_coop + d * alpha * v_punish_flow) / denom
        v_dev = pi_dev + d * (v_punish_flow + dL * v_coop)
        return v_coop - v_dev

    try:
        delta_star_s = brentq(_ic_gap, 0.01, 0.999, xtol=1e-8)
    except ValueError:
        delta_star_s = 1.0

    dL = delta_actual ** L
    v_punish_flow = pi_nash * (1.0 - dL) / (1.0 - delta_actual)
    denom = 1.0 - delta_actual * (1.0 - alpha) - alpha * delta_actual * dL
    v_coop = (pi_coop + delta_actual * alpha * v_punish_flow) / denom if abs(denom) > 1e-15 else 0.0
    v_dev = pi_dev + delta_actual * (v_punish_flow + dL * v_coop)

    return GreenPorterDeltaResult(
        delta_star_deterministic=det.delta_star,
        delta_star_stochastic=round(delta_star_s, 4),
        alpha=round(alpha, 4),
        binding_player=binding,
        delta_actual=delta_actual,
        cooperation_sustainable=(delta_actual >= delta_star_s),
        V_coop=round(v_coop, 2),
        V_deviate=round(v_dev, 2),
    )


def convergence_after_shock(
    gp_outcome: GreenPorterOutcome,
    shock_threshold: float = 15.0,
    horizon: int = 20,
) -> Optional[ConvergenceAfterShock]:
    """Compute mean conditional price path after a large negative shock.

    Identifies periods where price dropped more than *shock_threshold*
    below the cooperative price, then extracts the next *horizon* periods
    to measure recovery speed and half-life.
    """
    prices = gp_outcome.price_paths
    n_paths, T = prices.shape
    p_coop = gp_outcome.coop_price
    p_nash = gp_outcome.nash_price
    threshold = p_coop - shock_threshold

    snippets_p: List[np.ndarray] = []
    snippets_q: List[np.ndarray] = []
    total_q = np.zeros_like(prices)
    for p in gp_outcome.players:
        total_q += gp_outcome.player_quantity_paths[p]

    for i in range(n_paths):
        for t in range(T - horizon):
            if prices[i, t] < threshold:
                snippets_p.append(prices[i, t:t + horizon])
                snippets_q.append(total_q[i, t:t + horizon])

    if len(snippets_p) < 5:
        return None

    mean_p = np.mean(snippets_p, axis=0)
    std_p = np.std(snippets_p, axis=0)
    mean_q = np.mean(snippets_q, axis=0)

    gap_0 = abs(mean_p[0] - p_coop)
    half_gap = gap_0 / 2.0
    hl = float(horizon)
    for k in range(1, horizon):
        if abs(mean_p[k] - p_coop) <= half_gap:
            frac = 0.0
            prev_gap = abs(mean_p[k - 1] - p_coop)
            curr_gap = abs(mean_p[k] - p_coop)
            if abs(prev_gap - curr_gap) > 1e-10:
                frac = (prev_gap - half_gap) / (prev_gap - curr_gap)
            hl = (k - 1) + frac
            break

    return ConvergenceAfterShock(
        mean_price_path=mean_p,
        std_price_path=std_p,
        mean_quantity_path=mean_q,
        n_events=len(snippets_p),
        half_life_periods=round(hl, 2),
        coop_price=p_coop,
        nash_price=p_nash,
        horizon=horizon,
    )
