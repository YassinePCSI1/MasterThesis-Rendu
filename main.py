"""Entry point for Cournot oil market thesis simulations."""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.config import SimulationParams, default_simulation_params
from src.oil_game import OilGameTheory
from src.report import generate_report
from src.simulations import (
    run_repeated_mode,
    run_rl_mode,
    run_static_mode,
    run_stackelberg_mode,
    run_coalition_mode,
    run_market_power_mode,
    run_stochastic_mode,
    run_evolutionary_mode,
)

ALL_MODES = [
    "static", "repeated", "rl", "multi_rl", "marl_stress",
    "stackelberg", "coalition",
    "market_power", "stochastic", "evolutionary",
    "bertrand", "capacity", "welfare", "n_player", "correlated", "empirical",
    "all",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cournot oil market thesis model")
    parser.add_argument("--mode", choices=ALL_MODES, default="all")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--punishment_length", type=int, default=None)
    parser.add_argument("--capacities", choices=["on", "off"], default="off")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_outputs", choices=["on", "off"], default="on")
    parser.add_argument("--stochastic_sigma", type=float, default=None,
                        help="Std dev of demand shock ($/bbl). Default 8.0.")
    parser.add_argument("--stochastic_rho", type=float, default=None,
                        help="AR-1 autocorrelation for demand shock. Default 0.6.")
    parser.add_argument("--stochastic_paths", type=int, default=None,
                        help="Number of Monte Carlo paths. Default 300.")
    parser.add_argument("--marl_episodes", type=int, default=None,
                        help="Override MultiAgentRLParams.episodes (full MARL training).")
    parser.add_argument("--marl_robustness_seeds", type=int, default=None,
                        help="Override MultiAgentRLParams.robustness_n_seeds (headline triopoly + duopoly).")
    parser.add_argument("--marl_learner_seeds", type=int, default=None,
                        help="Override learner-comparison n_seeds per regime (1/2/3 learners).")
    parser.add_argument("--marl_stress_seeds", type=int, default=None,
                        help="Override stress-test n_seeds (γ-sweep, shock, deviation, Stackelberg, capacity).")
    parser.add_argument("--marl_stress_episode_fraction", type=float, default=None,
                        help="Stress-test training budget as a fraction of marl_episodes (default 0.25).")
    return parser


def apply_overrides(params: SimulationParams, args: argparse.Namespace) -> None:
    if args.T is not None:
        params.repeated.T = args.T
    if args.delta is not None:
        params.repeated.delta = args.delta
    if args.punishment_length is not None:
        params.repeated.punishment_length = args.punishment_length
    params.capacities.enabled = args.capacities == "on"
    params.seed = args.seed
    if args.stochastic_sigma is not None:
        params.stochastic.sigma = args.stochastic_sigma
    if args.stochastic_rho is not None:
        params.stochastic.rho = args.stochastic_rho
    if args.stochastic_paths is not None:
        params.stochastic.n_paths = args.stochastic_paths
    if args.marl_episodes is not None:
        params.multi_rl.episodes = args.marl_episodes
    if args.marl_robustness_seeds is not None:
        params.multi_rl.robustness_n_seeds = args.marl_robustness_seeds
    if args.marl_learner_seeds is not None:
        params.multi_rl.learner_comparison_n_seeds = args.marl_learner_seeds
    if args.marl_stress_seeds is not None:
        params.multi_rl.stress_n_seeds = args.marl_stress_seeds
    if args.marl_stress_episode_fraction is not None:
        params.multi_rl.stress_episodes_fraction = args.marl_stress_episode_fraction


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    params = default_simulation_params()
    apply_overrides(params, args)

    np.random.seed(params.seed)

    output_dir = os.path.join(os.getcwd(), "outputs")
    if args.save_outputs == "on":
        os.makedirs(output_dir, exist_ok=True)

    game = OilGameTheory(params=params)
    artifacts = {}

    mode = args.mode
    if mode in ["static", "all"]:
        artifacts.update(game.run_static(output_dir))
    if mode in ["repeated", "all"]:
        artifacts.update(game.run_repeated(output_dir))
    if mode in ["rl", "all"]:
        artifacts.update(game.run_rl(output_dir))
    if mode in ["multi_rl", "all"]:
        artifacts.update(game.run_multi_rl(output_dir))
    if mode in ["marl_stress", "all"]:
        artifacts.update(game.run_marl_stress(output_dir))
    if mode in ["stackelberg", "all"]:
        artifacts.update(game.run_stackelberg(output_dir))
    if mode in ["coalition", "all"]:
        artifacts.update(game.run_coalition(output_dir))
    if mode in ["market_power", "all"]:
        artifacts.update(game.run_market_power(output_dir))
    if mode in ["stochastic", "all"]:
        artifacts.update(game.run_stochastic(output_dir))
    if mode in ["evolutionary", "all"]:
        artifacts.update(game.run_evolutionary(output_dir))
    if mode in ["bertrand", "all"]:
        artifacts.update(game.run_bertrand(output_dir))
    if mode in ["capacity", "all"]:
        artifacts.update(game.run_capacity(output_dir))
    if mode in ["welfare", "all"]:
        artifacts.update(game.run_welfare(output_dir))
    if mode in ["n_player", "all"]:
        artifacts.update(game.run_n_player(output_dir))
    if mode in ["correlated", "all"]:
        artifacts.update(game.run_correlated(output_dir))
    if mode in ["empirical", "all"]:
        artifacts.update(game.run_empirical(output_dir))

    if args.save_outputs == "on":
        report_path = generate_report(output_dir, artifacts, params)
        artifacts["report"] = report_path


if __name__ == "__main__":
    main()
