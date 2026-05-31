"""Class-based API wrapper for the oil market game theory model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import SimulationParams
from .simulations import (
    run_repeated_mode,
    run_rl_mode,
    run_static_mode,
    run_stackelberg_mode,
    run_coalition_mode,
    run_market_power_mode,
    run_stochastic_mode,
    run_evolutionary_mode,
    run_multiagent_rl_mode,
    run_marl_stress_tests,
    run_bertrand_mode,
    run_capacity_mode,
    run_welfare_mode,
    run_n_player_mode,
    run_correlated_mode,
    run_empirical_mode,
)


@dataclass
class OilGameTheory:
    """Unified class interface for running Cournot oil market models.

    Supports the following modes:
    - static        : one-shot Cournot equilibrium (duopoly / triopoly)
    - repeated      : myopic best-response dynamics and tacit punishment
                      (+ inertia lambda sensitivity sweep)
    - rl            : Q-learning agent (OPEC) vs best-responding rivals
    - multi_rl      : two simultaneous Q-learning agents (OPEC + US) under
                      Green-Porter price-only information (Section 8b)
    - stackelberg   : quantity-leadership (all three leader configurations)
    - coalition     : Shapley values and Folk-theorem delta* analysis
    - market_power  : Lerner index, HHI, market shares across structures
    - stochastic    : Monte Carlo demand shocks (AR-1 + jump-diffusion)
    - evolutionary  : Evolutionary game dynamics (C/D/P, phase diagrams)
    - bertrand      : differentiated price competition robustness check
    - capacity      : capacity-constraint activation analysis
    - welfare       : welfare decomposition and deadweight loss
    - n_player      : sensitivity to the number of producers (2..6)
    - correlated    : correlated-equilibrium (mediator) analysis
    - empirical     : validation against historical price-war episodes
    """

    params: SimulationParams

    def run_static(self, output_dir: str) -> Dict[str, str]:
        return run_static_mode(self.params, output_dir)

    def run_repeated(self, output_dir: str) -> Dict[str, str]:
        return run_repeated_mode(self.params, output_dir)

    def run_rl(self, output_dir: str) -> Dict[str, str]:
        return run_rl_mode(self.params, output_dir)

    def run_multi_rl(self, output_dir: str) -> Dict[str, str]:
        return run_multiagent_rl_mode(self.params, output_dir)

    def run_marl_stress(self, output_dir: str) -> Dict[str, str]:
        return run_marl_stress_tests(self.params, output_dir)

    def run_stackelberg(self, output_dir: str) -> Dict[str, str]:
        return run_stackelberg_mode(self.params, output_dir)

    def run_coalition(self, output_dir: str) -> Dict[str, str]:
        return run_coalition_mode(self.params, output_dir)

    def run_market_power(self, output_dir: str) -> Dict[str, str]:
        return run_market_power_mode(self.params, output_dir)

    def run_stochastic(self, output_dir: str) -> Dict[str, str]:
        return run_stochastic_mode(self.params, output_dir)

    def run_evolutionary(self, output_dir: str) -> Dict[str, str]:
        return run_evolutionary_mode(self.params, output_dir)

    def run_bertrand(self, output_dir: str) -> Dict[str, str]:
        return run_bertrand_mode(self.params, output_dir)

    def run_capacity(self, output_dir: str) -> Dict[str, str]:
        return run_capacity_mode(self.params, output_dir)

    def run_welfare(self, output_dir: str) -> Dict[str, str]:
        return run_welfare_mode(self.params, output_dir)

    def run_n_player(self, output_dir: str) -> Dict[str, str]:
        return run_n_player_mode(self.params, output_dir)

    def run_correlated(self, output_dir: str) -> Dict[str, str]:
        return run_correlated_mode(self.params, output_dir)

    def run_empirical(self, output_dir: str) -> Dict[str, str]:
        return run_empirical_mode(self.params, output_dir)

    def run_all(self, output_dir: str) -> Dict[str, str]:
        artifacts: Dict[str, str] = {}
        artifacts.update(self.run_static(output_dir))
        artifacts.update(self.run_bertrand(output_dir))
        artifacts.update(self.run_repeated(output_dir))
        artifacts.update(self.run_rl(output_dir))
        artifacts.update(self.run_multi_rl(output_dir))
        artifacts.update(self.run_marl_stress(output_dir))
        artifacts.update(self.run_stackelberg(output_dir))
        artifacts.update(self.run_coalition(output_dir))
        artifacts.update(self.run_market_power(output_dir))
        artifacts.update(self.run_capacity(output_dir))
        artifacts.update(self.run_welfare(output_dir))
        artifacts.update(self.run_n_player(output_dir))
        artifacts.update(self.run_correlated(output_dir))
        artifacts.update(self.run_stochastic(output_dir))
        artifacts.update(self.run_evolutionary(output_dir))
        artifacts.update(self.run_empirical(output_dir))
        return artifacts
