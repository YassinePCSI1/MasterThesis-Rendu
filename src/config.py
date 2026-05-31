"""Configuration dataclasses and default calibrated parameters.

This project is calibrated for interpretability, not for forecasting oil prices.

Calibration rationale (see calibration.py for full documentation):
  - Demand intercept a=140: Brent crude choke price consistent with IEA (2019)
    high-end demand projections and structural energy security considerations.
  - Demand slope b=1.0: Normalised so quantities are in mbd (million barrels/day).
    Implied price elasticity of demand ≈ -0.75 at baseline (P≈60, Q≈80),
    consistent with medium-run empirical estimates (IMF: -0.15 to -0.25 long-run;
    EIA short-run: -0.06 to -0.10). The model elasticity reflects the quarterly
    time step where producers can observe and respond to demand shifts.
  - c_US=45: US shale break-even well-documented between $40-50/bbl (Dallas Fed,
    EIA 2020); represents WTI-equivalent lifting + midstream cost.
  - c_OPEC=20: Saudi Aramco published lifting cost ~$2-4/bbl + ~$16 fiscal/overhead;
    Gulf OPEC weighted average ~$18-22/bbl (BP Statistical Review 2022).
  - c_RUS=35: Russian conventional oil (Siberia) $15-20 lifting + $15 transport
    to Urals export terminal (IMF Russia Article IV, 2021).
  - delta=0.95: Quarterly discount factor = 1/1.05^0.25 ≈ 0.988; we use 0.95
    to reflect geopolitical uncertainty and imperfect enforcement (higher impatience).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

PLAYERS = ["US", "OPEC", "RUS"]


@dataclass
class DemandParams:
    """Linear inverse demand P(Q) = a - b Q.

    Units: P in $/bbl, Q in mbd (million barrels/day).
    """

    a: float = 140.0
    b: float = 1.0
    price_floor: float | None = 0.0


@dataclass
class CostParams:
    """Constant marginal costs by player (calibrated, not estimated)."""

    c_us: float = 45.0
    c_opec: float = 20.0
    c_rus: float = 35.0


@dataclass
class CapacityParams:
    """Capacity constraints q_i in [0, cap_i]."""

    enabled: bool = False
    cap_us: float = 30.0
    cap_opec: float = 40.0
    cap_rus: float = 35.0


@dataclass
class AdjustmentCostParams:
    """Ramping/adjustment cost for repeated-game dynamics."""

    enabled: bool = False
    k: float = 2.0


@dataclass
class RepeatedGameParams:
    """Repeated-game simulation settings."""

    T: int = 50
    delta: float = 0.95
    punishment_length: int = 10
    inertia: float = 0.2
    deviation_tolerance: float = 1e-3
    grim_trigger: bool = False


@dataclass
class RLParams:
    """Q-learning configuration for a learning agent."""

    episodes: int = 300
    steps_per_episode: int = 50
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon: float = 0.1
    action_grid_size: int = 15
    state_bins: Dict[str, int] = field(
        default_factory=lambda: {
            "price": 10,
            "Q": 10,
            "own_q": 8,
            "others_q": 8,
        }
    )


@dataclass
class MultiAgentRLParams:
    """Multi-agent Q-learning — deux agents observant uniquement le prix de marché.

    Correspond au cadre Green-Porter (1984) : les agents ne voient pas les
    quantités individuelles des adversaires, seulement le signal de prix agrégé.

    Notes
    -----
    - ``learning_players`` : list of agents that learn simultaneously
      (default: OPEC and US; RUS plays myopic best-response).
    - ``epsilon_*`` : exploration is annealed geometrically each episode,
      ``epsilon_t = max(epsilon_end, epsilon_start * epsilon_decay^t)``.
    - ``price_bins`` / ``own_q_bins`` : 2-D state — price observed and
      agent's own previous output. No rival quantities in state space.
    - ``eval_last_fraction`` : tail fraction of training used to compute
      converged averages and the collusion index.
    """

    learning_players: List[str] = field(default_factory=lambda: ["OPEC", "US"])
    episodes: int = 500
    steps_per_episode: int = 50
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon_start: float = 0.30
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.99
    action_grid_size: int = 15
    price_bins: int = 12
    own_q_bins: int = 10
    eval_last_fraction: float = 0.20
    robustness_n_seeds: int = 20
    # Budget for the 1-vs-2-vs-3 learner-count comparison (per regime).
    learner_comparison_n_seeds: int = 10
    # Stress-test battery (γ-sweep, shocks, deviation, Stackelberg, capacity).
    stress_n_seeds: int = 3
    # Stress-test training budget as a fraction of `episodes` (0 < x <= 1).
    stress_episodes_fraction: float = 0.25


@dataclass
class StackelbergParams:
    """Stackelberg quantity-leadership configuration."""

    leader: str = "OPEC"
    compare_all_leaders: bool = True


@dataclass
class StochasticParams:
    """Parameters for stochastic demand simulations.

    AR(1) demand shock:  ε_t = ρ·ε_{t-1} + η_t,  η_t ~ N(0, σ²)
    Jump-diffusion:      add Poisson jumps for supply disruptions / OPEC shocks.
    """

    sigma: float = 8.0
    rho: float = 0.6
    n_paths: int = 300
    T: int = 50
    shock_sizes: List[float] = field(
        default_factory=lambda: [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
    )
    # Jump-diffusion parameters (set jump_intensity>0 to activate)
    jump_intensity: float = 0.08   # avg jumps per period (≈ 1 shock per year at quarterly step)
    jump_mu: float = 0.0           # mean jump size ($/bbl); 0 = symmetric shocks
    jump_sigma: float = 18.0       # std dev of jump (captures ±$18 supply shock, e.g. 2022 war)


@dataclass
class EvoGameParams:
    """Parameters for Evolutionary Game Dynamics simulations.

    Models producers as a population choosing between strategies
    (Cooperate / Defect / Punish). Replicator dynamics track how
    strategy shares evolve as players imitate more successful peers.
    """

    T: int = 400                 # number of replicator time-steps
    dt: float = 0.05             # integration step size (5 units of 'evolutionary time')
    n_trajectories: int = 12     # number of initial conditions to explore
    grid_resolution: int = 50    # phase-diagram grid points


@dataclass
class LambdaSweepParams:
    """Adjustment speed (λ) sensitivity sweep for the repeated game.

    λ = 0 : full inertia (never adjusts — output stays at previous level)
    λ = 1 : no inertia (pure best response each period)
    Baseline λ=0.2 means each player moves 20% toward its best response
    per period.  Target range covers the empirically plausible region for
    OPEC quota adjustment speeds (quarterly ramp ≈ 0.1-0.3 per period).
    """

    lambda_values: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    )


@dataclass
class RLSweepParams:
    """Hyperparameter sweep for the Q-learning agent.

    Justification for baseline values (alpha=0.15, gamma=0.95, epsilon=0.10):
      alpha=0.15 : moderate learning rate; fast enough to converge within 300
                   episodes yet stable (standard recommendation: 0.1-0.2).
      gamma=0.95 : matches the economic discount factor δ, so the agent
                   implicitly optimises the same objective as the game-theoretic model.
      epsilon=0.1: low exploration noise; agent has already identified the
                   action space from early episodes.
    """

    alpha_values: List[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.10, 0.15, 0.25, 0.40]
    )
    gamma_values: List[float] = field(
        default_factory=lambda: [0.70, 0.80, 0.90, 0.95, 0.99]
    )
    epsilon_values: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.20, 0.40]
    )
    eval_last_episodes: int = 50   # episodes used to compute converged average reward


@dataclass
class GreenPorterParams:
    """Green-Porter (1984) repeated game under imperfect public monitoring.

    Players observe the market price but not individual quantities.
    When the observed price falls below a trigger level p_bar, all players
    switch to Nash (punishment) for L periods, then revert to cooperation.
    This happens even when nobody deviated --- a negative demand shock alone
    can trigger punishment ("false positive").

    trigger_price_offset : p_bar = P_coop - offset.  Smaller offset means
        tighter monitoring (more false triggers but faster detection).
        Default 12 $/bbl with P_coop ~ 80 gives p_bar ~ 68, roughly
        1.2 sigma below the cooperative price (sigma_eps ~ 10).
    """

    trigger_price_offset: float = 12.0
    punishment_length: int = 10
    n_paths: int = 500
    T: int = 100


@dataclass
class BertrandParams:
    """Bertrand price-competition parameters.

    In differentiated Bertrand, each player sets a price; demand for player i
    depends on own price and rivals' prices through a substitution parameter σ.
    σ = 0 : independent goods (each is a monopoly)
    σ → 1 : perfect substitutes (converges to classic Bertrand = marginal cost pricing)

    The baseline σ = 0.6 reflects medium differentiation — crude oil is fairly
    homogeneous but not perfectly substitutable (grade, location, sulfur content).
    """

    sigma: float = 0.6
    sigma_sweep: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.4, 0.6, 0.8, 0.95]
    )
    base_demand_scale: float = 40.0


@dataclass
class CapacitySweepParams:
    """Sweep OPEC's capacity from tight to slack to quantify capacity power."""

    opec_cap_values: List[float] = field(
        default_factory=lambda: [25.0, 30.0, 35.0, 40.0, 50.0, 60.0]
    )


@dataclass
class WelfareParams:
    """Welfare analysis parameters.

    carbon_tax_values : list of $/bbl carbon taxes added to every producer's
                        marginal cost; used to study how an environmental tax
                        interacts with collusion incentives.
    """

    carbon_tax_values: List[float] = field(
        default_factory=lambda: [0.0, 5.0, 10.0, 20.0, 30.0, 50.0]
    )


@dataclass
class NPlayerParams:
    """Sensitivity to the number of producers.

    Additional players are calibrated with costs between c_RUS (35) and c_US (45),
    representing plausible new entrants (e.g., Brazil pre-salt, Canada oil sands,
    Norway, Iraq breakaway).
    """

    max_players: int = 6
    extra_player_costs: List[float] = field(
        default_factory=lambda: [38.0, 42.0, 48.0]
    )
    extra_player_names: List[str] = field(
        default_factory=lambda: ["BRA", "CAN", "NOR"]
    )


@dataclass
class CorrelatedEqParams:
    """Correlated equilibrium parameters.

    action_grid_size : number of discrete output levels per player for the LP.
    objectives       : which CE objectives to compute and compare.
    """

    action_grid_size: int = 10
    objectives: List[str] = field(
        default_factory=lambda: ["max_welfare", "max_joint_profit", "max_min_profit"]
    )


@dataclass
class EmpiricalParams:
    """Empirical-validation parameters: which historical episodes to compare."""

    episodes: List[str] = field(
        default_factory=lambda: [
            "1985_opec_price_war",
            "2014_opec_market_share",
            "2020_russia_saudi_price_war",
        ]
    )


@dataclass
class SimulationParams:
    """Top-level configuration object for the project."""

    demand: DemandParams = field(default_factory=DemandParams)
    costs: CostParams = field(default_factory=CostParams)
    capacities: CapacityParams = field(default_factory=CapacityParams)
    adjustment: AdjustmentCostParams = field(default_factory=AdjustmentCostParams)
    repeated: RepeatedGameParams = field(default_factory=RepeatedGameParams)
    rl: RLParams = field(default_factory=RLParams)
    multi_rl: MultiAgentRLParams = field(default_factory=MultiAgentRLParams)
    stackelberg: StackelbergParams = field(default_factory=StackelbergParams)
    stochastic: StochasticParams = field(default_factory=StochasticParams)
    evo: EvoGameParams = field(default_factory=EvoGameParams)
    lambda_sweep: LambdaSweepParams = field(default_factory=LambdaSweepParams)
    rl_sweep: RLSweepParams = field(default_factory=RLSweepParams)
    green_porter: GreenPorterParams = field(default_factory=GreenPorterParams)
    bertrand: BertrandParams = field(default_factory=BertrandParams)
    capacity_sweep: CapacitySweepParams = field(default_factory=CapacitySweepParams)
    welfare: WelfareParams = field(default_factory=WelfareParams)
    n_player: NPlayerParams = field(default_factory=NPlayerParams)
    correlated_eq: CorrelatedEqParams = field(default_factory=CorrelatedEqParams)
    empirical: EmpiricalParams = field(default_factory=EmpiricalParams)

    players: List[str] = field(default_factory=lambda: PLAYERS.copy())
    seed: int = 123


def default_simulation_params() -> SimulationParams:
    """Return baseline parameters used throughout the thesis project."""

    return SimulationParams()
