"""Simulation orchestrators for static, repeated, RL, Stackelberg, coalition, and stochastic modes."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import SimulationParams
from .cooperation_punishment import cartel_quotas, cooperative_output
from .cournot_repeated import simulate_myopic, simulate_tacit_punishment
from .cournot_static import CournotResult, cournot_equilibrium, duopoly_vs_triopoly
from .plotting import (
    plot_action_distribution,
    plot_comparative_statics,
    plot_duopoly_vs_triopoly,
    plot_learning_curve,
    plot_profit_comparison,
    plot_quantity_comparison,
    plot_repeated_price_quantity,
    plot_repeated_profits,
    plot_punishment_regimes,
    plot_rl_rolling_outputs,
    plot_time_series,
    plot_stackelberg_comparison,
    plot_market_power,
    plot_stochastic_price_bands,
    plot_shapley_values,
    plot_folk_theorem,
    plot_demand_shock_scenarios,
    plot_myopic_with_nash_convergence,
    plot_lambda_sensitivity,
    plot_rl_hyperparameter_sweep,
    plot_stochastic_comparison,
    plot_evo_phase_diagram_2strategy,
    plot_evo_phase_diagram_3strategy,
    plot_evo_payoff_matrix,
    plot_evo_punishment_sweep,
    plot_green_porter_regimes,
    plot_cooperation_survival,
    plot_delta_star_comparison,
    plot_convergence_after_shock,
    plot_multiagent_price_convergence,
    plot_multiagent_outputs,
    plot_multiagent_vs_singleagent,
    plot_multiagent_collusion_decomposition,
    plot_multiagent_robustness_distribution,
    plot_multiagent_robustness_prices,
    plot_punishment_detection,
    plot_punishment_anatomy,
    plot_learner_comparison_collusion,
    plot_learner_comparison_prices,
    plot_marl_gamma_sweep,
    plot_marl_shock_response,
    plot_marl_forced_deviation,
    plot_marl_capacity_comparison,
    plot_marl_stackelberg_comparison,
    plot_marl_policy_heatmaps,
)
from .rl_agent import train_q_learning, RLOutcome
from .rl_multiagent import (
    train_multiagent_ql,
    MultiAgentRLOutcome,
    run_multiagent_robustness,
    run_random_policy_baseline,
    detect_punishment_episodes,
    compute_punishment_statistics,
    run_multiagent_comparison,
    run_marl_gamma_sweep,
    run_marl_shock_experiment,
    run_marl_forced_deviation,
    run_marl_capacity_experiment,
    run_marl_stackelberg_comparison,
    policy_heatmap,
)
from .stackelberg import stackelberg_equilibrium
from .coalition import shapley_values, folk_theorem_delta_star
from .market_power import market_power_metrics, market_power_table
from .stochastic import simulate_stochastic_cournot, simulate_stochastic_jump_diffusion, demand_shock_scenario
from .evolutionary import run_evolutionary_game, punishment_multiplier_sweep


def _summary_row(players: List[str], result: CournotResult) -> Dict[str, float]:
    row = {f"q_{p}": result.quantities[p] for p in players}
    row.update(
        {
            "Q": result.total_quantity,
            "P": result.price,
            "consumer_surplus": result.consumer_surplus,
            "total_welfare": result.total_welfare,
        }
    )
    for p in players:
        row[f"profit_{p}"] = result.profits[p]
    return row


def run_comparative_statics(params: SimulationParams,
                             param_name: str,
                             values: List[float]) -> pd.DataFrame:
    """Vary one parameter and recompute equilibrium outputs/prices."""

    records = []
    original = {
        "a": params.demand.a,
        "b": params.demand.b,
        "c_opec": params.costs.c_opec,
    }
    for v in values:
        if param_name == "a":
            params.demand.a = v
        elif param_name == "b":
            params.demand.b = v
        elif param_name == "c_opec":
            params.costs.c_opec = v
        else:
            raise ValueError("Unsupported parameter for comparative statics")

        res = cournot_equilibrium(params.players, params.demand, params.costs, params.capacities)
        records.append(
            {
                "param": param_name,
                "param_value": v,
                "price": res.price,
                "total_quantity": res.total_quantity,
            }
        )

    params.demand.a = original["a"]
    params.demand.b = original["b"]
    params.costs.c_opec = original["c_opec"]
    return pd.DataFrame(records)


def run_static_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Run static Cournot model and produce outputs."""

    duo, tri = duopoly_vs_triopoly(params.demand, params.costs, params.capacities)
    fig_path = f"{output_dir}/duopoly_vs_triopoly.png"
    plot_duopoly_vs_triopoly(duo, tri, fig_path)

    quantity_path = f"{output_dir}/quantity_comparison.png"
    plot_quantity_comparison(duo, tri, quantity_path)

    profit_path = f"{output_dir}/profit_comparison.png"
    plot_profit_comparison(duo, tri, profit_path)

    summary = pd.DataFrame([
        _summary_row(["US", "OPEC"], duo),
        _summary_row(["US", "OPEC", "RUS"], tri),
    ], index=["duopoly", "triopoly"])
    summary.index.name = "model"
    summary = summary.reset_index()
    summary_path = f"{output_dir}/static_equilibrium.csv"
    summary.to_csv(summary_path, index=False)

    statics = run_comparative_statics(
        params, "a", list(np.linspace(params.demand.a * 0.8, params.demand.a * 1.2, 8))
    )
    statics_path = f"{output_dir}/comparative_statics.csv"
    statics.to_csv(statics_path, index=False)
    plot_comparative_statics(statics.to_dict(orient="records"), f"{output_dir}/comparative_statics.png")

    return {
        "summary": summary_path,
        "duopoly_plot": fig_path,
        "comparative_csv": statics_path,
        "quantity_plot": quantity_path,
        "profit_plot": profit_path,
    }


def run_repeated_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Run repeated-game simulations (myopic and cooperative with punishment)."""

    myopic = simulate_myopic(
        params.players,
        params.demand,
        params.costs,
        params.capacities,
        params.repeated,
        params.adjustment,
    )

    cartel = cartel_quotas(params.players, params.demand, params.costs, params.capacities)
    nash = cournot_equilibrium(params.players, params.demand, params.costs, params.capacities)
    punishment = simulate_tacit_punishment(
        params.players,
        params.demand,
        params.costs,
        params.capacities,
        params.repeated,
        cooperative_outputs=cartel.quotas,
        punishment_outputs=nash.quantities,
        deviation_period=5,
        deviator="OPEC",
    )

    time_path = f"{output_dir}/repeated_time_series.png"
    plot_time_series(myopic.prices, myopic.quantities, time_path,
                     nash_price=nash.price)

    price_q_path = f"{output_dir}/repeated_price_quantity.png"
    plot_repeated_price_quantity(myopic.prices, myopic.total_quantities, price_q_path,
                                  nash_price=nash.price, nash_Q=nash.total_quantity)

    profit_ts_path = f"{output_dir}/repeated_profit_time_series.png"
    plot_repeated_profits(
        myopic.profits,
        profit_ts_path,
        nash_profits=nash.profits,
        coop_profits=cartel.quota_profits,
        nash_price=nash.price,
        prices=myopic.prices,
    )

    # Wrap cartel quotas in a CooperationResult-compatible object for plotting
    from .cooperation_punishment import CooperationResult
    cartel_for_plot = CooperationResult(
        target_outputs=cartel.quotas,
        target_price=cartel.quota_price,
        target_profit=cartel.quota_profits,
    )
    nash_convergence_path = f"{output_dir}/repeated_nash_convergence.png"
    plot_myopic_with_nash_convergence(myopic, nash, cartel_for_plot, nash_convergence_path)

    # NEW: dedicated punishment-regime plot with coloured phase backgrounds
    deviation_period = 5
    punishment_plot_path = f"{output_dir}/repeated_punishment_regimes.png"
    plot_punishment_regimes(
        punishment,
        deviation_period=deviation_period,
        punishment_length=params.repeated.punishment_length,
        nash_result=nash,
        cartel_result=cartel,
        path=punishment_plot_path,
    )

    df_myopic = _build_time_series_df(params.players, myopic)
    df_myopic.to_csv(f"{output_dir}/repeated_myopic.csv", index=False)

    df_punish = _build_time_series_df(params.players, punishment)
    df_punish.to_csv(f"{output_dir}/repeated_punishment.csv", index=False)

    volatility = {
        "price_volatility": float(np.std(myopic.prices)),
        "quantity_volatility": float(np.std(myopic.total_quantities)),
    }

    volatility_path = f"{output_dir}/repeated_volatility.csv"
    pd.DataFrame([volatility]).to_csv(volatility_path, index=False)

    discounted = _discounted_profit_table(params.players, myopic, punishment, params.repeated.delta)
    discounted_path = f"{output_dir}/repeated_discounted_profits.csv"
    discounted.to_csv(discounted_path, index=False)

    # NEW: lambda sensitivity sweep
    lambda_artifacts = run_lambda_sensitivity(params, output_dir)

    artifacts = {
        "time_series_plot": time_path,
        "price_quantity_plot": price_q_path,
        "profit_time_series_plot": profit_ts_path,
        "nash_convergence_plot": nash_convergence_path,
        "punishment_regimes_plot": punishment_plot_path,
        "myopic_csv": f"{output_dir}/repeated_myopic.csv",
        "punishment_csv": f"{output_dir}/repeated_punishment.csv",
        "volatility_csv": volatility_path,
        "discounted_csv": discounted_path,
    }
    artifacts.update(lambda_artifacts)
    return artifacts


def run_lambda_sensitivity(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Sweep inertia λ values to validate the calibration choice.

    Computes convergence speed and final price for each λ, then produces
    a three-panel comparison plot.
    """
    nash = cournot_equilibrium(params.players, params.demand, params.costs, params.capacities)
    lambda_results: Dict[float, object] = {}

    for lam in params.lambda_sweep.lambda_values:
        outcome = simulate_myopic(
            params.players,
            params.demand,
            params.costs,
            params.capacities,
            params.repeated,
            params.adjustment,
            inertia=lam,
        )
        lambda_results[lam] = outcome

    n = len(params.players)
    lam_star = 2.0 / (n + 1)
    nash_price = nash.price

    csv_rows = []
    for lam, outcome in lambda_results.items():
        prices = np.array(outcome.prices)
        eigenvalue = 1.0 - 2.0 * lam
        abs_ev = abs(eigenvalue)
        half_life = np.log(2) / (-np.log(abs_ev)) if 0 < abs_ev < 1 else (
            0.0 if abs_ev == 0 else float("inf"))
        within = np.where(np.abs(prices - nash_price) / nash_price < 0.02)[0]
        conv_period = int(within[0]) if len(within) > 0 else len(prices)
        regime = ("critical" if abs(lam - lam_star) < 0.01
                  else "monotone" if lam < lam_star
                  else "oscillatory")
        csv_rows.append({
            "lambda": lam,
            "eigenvalue_mu": round(eigenvalue, 4),
            "abs_eigenvalue": round(abs_ev, 4),
            "regime": regime,
            "half_life_periods": round(half_life, 2),
            "convergence_period_2pct": conv_period,
            "final_price": round(float(prices[-1]), 2),
            "price_volatility": round(float(np.std(prices)), 2),
        })
    lambda_csv = f"{output_dir}/lambda_sensitivity.csv"
    pd.DataFrame(csv_rows).to_csv(lambda_csv, index=False)

    lambda_plot = f"{output_dir}/lambda_sensitivity.png"
    plot_lambda_sensitivity(lambda_results, nash, lambda_plot, n_players=n)

    return {"lambda_csv": lambda_csv, "lambda_plot": lambda_plot}


def _build_time_series_df(players: List[str], outcome) -> pd.DataFrame:
    rows = []
    for t, q in enumerate(outcome.quantities):
        row = {"t": t, "P": outcome.prices[t], "Q": outcome.total_quantities[t]}
        row.update({f"q_{p}": q[p] for p in players})
        row.update({f"profit_{p}": outcome.profits[t][p] for p in players})
        rows.append(row)
    return pd.DataFrame(rows)


def _discounted_profit_table(players: List[str], myopic, punishment, delta: float) -> pd.DataFrame:
    def discounted_sum(outcome) -> Dict[str, float]:
        totals = {p: 0.0 for p in players}
        for t, profits in enumerate(outcome.profits):
            for p in players:
                totals[p] += (delta ** t) * profits[p]
        return totals

    return pd.DataFrame(
        [
            {"scenario": "myopic", **discounted_sum(myopic)},
            {"scenario": "punishment", **discounted_sum(punishment)},
        ]
    )


def run_rl_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Train and evaluate a Q-learning agent, plus hyperparameter sweep."""

    outcome = train_q_learning(
        params.players,
        params.demand,
        params.costs,
        params.capacities,
        params.rl,
        learning_player="OPEC",
        seed=params.seed,
    )

    learning_curve_path = f"{output_dir}/rl_learning_curve.png"
    plot_learning_curve(outcome.reward_history, params.rl.steps_per_episode, learning_curve_path)

    action_dist_path = f"{output_dir}/rl_action_distribution.png"
    plot_action_distribution(outcome.q_history, "OPEC", action_dist_path)

    rolling_path = f"{output_dir}/rl_rolling_outputs.png"
    plot_rl_rolling_outputs(outcome.q_history, window=50, path=rolling_path)

    df_rl = pd.DataFrame(outcome.q_history)
    df_rl["P"] = outcome.price_history
    df_rl["reward"] = outcome.reward_history
    df_rl.to_csv(f"{output_dir}/rl_time_series.csv", index=False)

    nash = cournot_equilibrium(params.players, params.demand, params.costs, params.capacities)
    cartel_rl = cartel_quotas(params.players, params.demand, params.costs, params.capacities)

    tail = df_rl.tail(200)
    rl_avg = {p: float(tail[p].mean()) for p in params.players}
    rl_Q = sum(rl_avg.values())
    from .demand import price_from_quantity
    rl_P = price_from_quantity(rl_Q, params.demand)

    comparison = pd.DataFrame(
        [
            {"benchmark": "nash", **{f"q_{p}": nash.quantities[p] for p in params.players}, "Q": nash.total_quantity, "P": nash.price},
            {"benchmark": "cooperative", **{f"q_{p}": cartel_rl.quotas[p] for p in params.players}, "Q": cartel_rl.total_output, "P": cartel_rl.quota_price},
            {"benchmark": "rl_avg", **{f"q_{p}": rl_avg[p] for p in params.players}, "Q": rl_Q, "P": rl_P},
        ]
    )
    comparison_path = f"{output_dir}/rl_benchmark_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    # NEW: hyperparameter sensitivity sweep
    sweep_artifacts = run_rl_hyperparameter_sweep(params, output_dir)

    artifacts = {
        "learning_curve": learning_curve_path,
        "action_distribution": action_dist_path,
        "rl_csv": f"{output_dir}/rl_time_series.csv",
        "rl_benchmark_csv": comparison_path,
        "rl_rolling_outputs": rolling_path,
    }
    artifacts.update(sweep_artifacts)
    return artifacts


def _avg_reward_last_episodes(outcome: RLOutcome, steps_per_episode: int, n_episodes: int) -> float:
    """Compute average reward over the last n_episodes of training."""
    rewards = np.array(outcome.reward_history)
    total_steps = len(rewards)
    start = max(0, total_steps - n_episodes * steps_per_episode)
    return float(rewards[start:].mean())


def run_rl_hyperparameter_sweep(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Sweep alpha, gamma, and epsilon to justify the baseline hyperparameter choices.

    Each sweep varies one parameter while holding the others at their baseline values.
    The metric is the average per-step reward over the last `eval_last_episodes` episodes
    — a standard measure of converged performance in Q-learning.

    Scientific justification for baselines (see config.py for full references):
      alpha=0.15, gamma=0.95, epsilon=0.10
    """
    sw = params.rl_sweep
    n_eval = sw.eval_last_episodes
    steps = params.rl.steps_per_episode
    np.random.seed(params.seed)

    def _run(alpha, gamma, epsilon) -> float:
        from copy import deepcopy
        rl_p = deepcopy(params.rl)
        rl_p.alpha = alpha
        rl_p.gamma = gamma
        rl_p.epsilon = epsilon
        out = train_q_learning(
            params.players, params.demand, params.costs, params.capacities,
            rl_p, learning_player="OPEC", seed=params.seed,
        )
        return _avg_reward_last_episodes(out, steps, n_eval)

    alpha_results = {a: _run(a, params.rl.gamma, params.rl.epsilon) for a in sw.alpha_values}
    gamma_results = {g: _run(params.rl.alpha, g, params.rl.epsilon) for g in sw.gamma_values}
    epsilon_results = {e: _run(params.rl.alpha, params.rl.gamma, e) for e in sw.epsilon_values}

    baseline = {"alpha": params.rl.alpha, "gamma": params.rl.gamma, "epsilon": params.rl.epsilon}

    sweep_plot = f"{output_dir}/rl_hyperparameter_sweep.png"
    plot_rl_hyperparameter_sweep(alpha_results, gamma_results, epsilon_results, baseline, sweep_plot)

    rows = (
        [{"param": "alpha", "value": k, "avg_reward": v} for k, v in alpha_results.items()]
        + [{"param": "gamma", "value": k, "avg_reward": v} for k, v in gamma_results.items()]
        + [{"param": "epsilon", "value": k, "avg_reward": v} for k, v in epsilon_results.items()]
    )
    sweep_csv = f"{output_dir}/rl_hyperparameter_sweep.csv"
    pd.DataFrame(rows).to_csv(sweep_csv, index=False)

    return {"rl_hyperparam_plot": sweep_plot, "rl_hyperparam_csv": sweep_csv}


# ---------------------------------------------------------------------------
# Multi-agent RL mode (Section 8b)
# ---------------------------------------------------------------------------

def run_multiagent_rl_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Section 8b: Multi-agent Q-learning under Green-Porter information.

    Trains two simultaneous Q-learning agents (default: OPEC and US) that
    observe ONLY the market price and their own previous output.  RUS plays
    a myopic Cournot best-response, exactly as in the existing single-agent
    Section 8.

    Scientific question
    -------------------
    Without communication and without knowing the model, do two reciprocally
    learning agents converge towards the cartel benchmark?  This is a direct
    empirical test of the Folk-Theorem prediction (Section 5) under the
    Green-Porter (1984) imperfect-monitoring information structure already
    used in Section 7 for the trigger-price punishment regime.

    Outputs
    -------
    - ``multiagent_rl_price_convergence.png``
    - ``multiagent_rl_outputs.png``
    - ``multiagent_rl_vs_singleagent.png``
    - ``multiagent_rl_collusion_decomposition.png``
    - ``multiagent_rl_robustness_distribution.png`` (multi-seed)
    - ``multiagent_rl_robustness_prices.png`` (multi-seed)
    - ``multiagent_rl_punishment_detection.png`` (Folk-Theorem signature)
    - ``multiagent_rl_punishment_anatomy.png`` (mean episode shape)
    - ``multiagent_rl_learner_comparison_collusion.png`` (1 vs 2 vs 3 learners)
    - ``multiagent_rl_learner_comparison_prices.png`` (1 vs 2 vs 3 learners)
    - ``multiagent_rl_learner_comparison.csv``
    - ``multiagent_rl_summary.csv``
    """
    # --- 1. Benchmarks (re-using existing project helpers) ---
    nash = cournot_equilibrium(params.players, params.demand, params.costs, params.capacities)
    cartel = cartel_quotas(params.players, params.demand, params.costs, params.capacities)
    nash_price = float(nash.price)
    cartel_price = float(cartel.quota_price)
    nash_q = {p: float(nash.quantities[p]) for p in params.players}
    cartel_q = {p: float(cartel.quotas[p]) for p in params.players}

    # --- 2. Train multi-agent Q-learning ---
    ma_outcome = train_multiagent_ql(
        learning_players=list(params.multi_rl.learning_players),
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=params.multi_rl,
        nash_price=nash_price,
        cartel_price=cartel_price,
        seed=params.seed,
    )

    # --- 3. Re-train the existing single-agent Q-learner for direct comparison ---
    sa_outcome = train_q_learning(
        params.players, params.demand, params.costs, params.capacities,
        params.rl, learning_player="OPEC", seed=params.seed,
    )

    # --- 4. Plots ---
    # NB. plotted curves use the *multi-seed* trajectories (computed in §5
    # below) so the bands are 95% empirical CIs across seeds, not single-seed
    # variability.  We defer the plot calls until robustness has populated
    # ``all_seed_outcomes``; the legacy single-seed paths are kept as a
    # fallback when robustness is disabled (e.g. quick smoke tests).
    price_path = f"{output_dir}/multiagent_rl_price_convergence.png"
    outputs_path = f"{output_dir}/multiagent_rl_outputs.png"

    vs_path = f"{output_dir}/multiagent_rl_vs_singleagent.png"
    plot_multiagent_vs_singleagent(sa_outcome, ma_outcome, nash_price, cartel_price, vs_path)

    decomp_path = f"{output_dir}/multiagent_rl_collusion_decomposition.png"
    plot_multiagent_collusion_decomposition(
        ma_outcome, nash_q, cartel_q, params.players, decomp_path,
    )

    # --- 5. Robustness analysis (multi-seed) ---
    #
    # Two parallel multi-seed batches are run when the budget allows:
    #
    # * **headline (triopoly)** — every player learns, so the figures and
    #   tables in §18.1 are drawn from this batch.  This is the experiment
    #   we will scale up in the upcoming large-budget run.
    # * **secondary (duopoly)** — the legacy configuration where only the
    #   two players listed in ``params.multi_rl.learning_players`` learn
    #   and the remaining player(s) act as myopic best-responders.  Kept
    #   as a secondary check (§18.2) because a *bimodal* CI distribution
    #   here would suggest a learning-driven Folk-Theorem signature.
    #
    # When robustness is disabled the figures fall back to the single-seed
    # ``ma_outcome`` and a warning is annotated on the plot.
    robustness_path = None
    robustness_prices_path = None
    robustness_csv_path = None
    robustness_stats: Dict[str, float] = {}
    all_seed_outcomes: List[MultiAgentRLOutcome] = []
    triopoly_robustness: Optional[Dict] = None
    triopoly_headline_stats: Dict[str, float] = {}

    if params.multi_rl.robustness_n_seeds and params.multi_rl.robustness_n_seeds > 0:
        # ── 5.a  HEADLINE: triopoly (every player learns) — feeds §18.1
        triopoly_robustness = run_multiagent_robustness(
            learning_players=list(params.players),
            all_players=params.players,
            demand=params.demand,
            costs=params.costs,
            capacities=params.capacities,
            params=params.multi_rl,
            nash_price=nash_price,
            cartel_price=cartel_price,
            n_seeds=params.multi_rl.robustness_n_seeds,
            base_seed=200,
        )
        all_seed_outcomes = list(triopoly_robustness["all_outcomes"])
        triopoly_headline_stats = {
            "headline_regime": "triopoly",
            "headline_n_seeds": int(triopoly_robustness["n_seeds"]),
            "headline_mean_collusion_index": round(
                float(triopoly_robustness["mean_collusion_index"]), 4),
            "headline_std_collusion_index": round(
                float(triopoly_robustness["std_collusion_index"]), 4),
            "headline_ci_95_low": round(float(triopoly_robustness["ci_95_low"]), 4),
            "headline_ci_95_high": round(float(triopoly_robustness["ci_95_high"]), 4),
            "headline_mean_converged_price": round(
                float(triopoly_robustness["mean_converged_price"]), 4),
            "headline_std_converged_price": round(
                float(triopoly_robustness["std_converged_price"]), 4),
            # Honest greedy (ε=0) evaluation — audit Option B.
            "headline_mean_greedy_price": round(
                float(triopoly_robustness["mean_greedy_price"]), 4),
            "headline_std_greedy_price": round(
                float(triopoly_robustness["std_greedy_price"]), 4),
            "headline_mean_greedy_collusion_index": round(
                float(triopoly_robustness["mean_greedy_collusion_index"]), 4),
            "headline_greedy_ci_95_low": round(
                float(triopoly_robustness["greedy_ci_95_low"]), 4),
            "headline_greedy_ci_95_high": round(
                float(triopoly_robustness["greedy_ci_95_high"]), 4),
            # Q-coverage diagnostics.
            "headline_mean_q_undervisited_pct": round(
                float(triopoly_robustness["mean_q_undervisited_pct"]), 4),
            "headline_mean_q_visit_mean": round(
                float(triopoly_robustness["mean_q_visit_mean"]), 2),
            "headline_min_q_visit_min": int(
                float(triopoly_robustness["min_q_visit_min"])),
        }
        # Per-seed CSV for the headline batch (§18.1 transparency).
        triopoly_rows = []
        for s, p, ci, outs, gp, gci, uv in zip(
            triopoly_robustness["seeds"],
            triopoly_robustness["converged_prices"],
            triopoly_robustness["collusion_indices"],
            triopoly_robustness["converged_outputs"],
            triopoly_robustness["greedy_prices"],
            triopoly_robustness["greedy_collusion_indices"],
            triopoly_robustness["q_undervisited_pcts"],
        ):
            triopoly_rows.append({
                "seed": s,
                "converged_price": round(float(p), 4),
                "collusion_index": round(float(ci), 4),
                "greedy_price": round(float(gp), 4),
                "greedy_collusion_index": round(float(gci), 4),
                "q_undervisited_pct": round(float(uv), 4),
                **{f"q_{k}": round(float(v), 4) for k, v in outs.items()},
            })
        pd.DataFrame(triopoly_rows).to_csv(
            f"{output_dir}/multiagent_rl_robustness_triopoly_per_seed.csv",
            index=False,
        )

        # ── 5.b  SECONDARY: legacy duopoly batch — feeds §18.2
        robustness = run_multiagent_robustness(
            learning_players=list(params.multi_rl.learning_players),
            all_players=params.players,
            demand=params.demand,
            costs=params.costs,
            capacities=params.capacities,
            params=params.multi_rl,
            nash_price=nash_price,
            cartel_price=cartel_price,
            n_seeds=params.multi_rl.robustness_n_seeds,
        )

        # Single-agent collusion index (computed below for the summary) is
        # also passed in so the violin plot shows the bar to beat.
        sa_tail_for_robust = np.asarray(
            sa_outcome.price_history[-min(200, len(sa_outcome.price_history)):],
            dtype=float,
        )
        sa_price_for_robust = float(sa_tail_for_robust.mean())
        denom_for_robust = (cartel_price - nash_price)
        sa_collusion_for_robust = (
            (sa_price_for_robust - nash_price) / denom_for_robust
            if abs(denom_for_robust) > 1e-9 else 0.0
        )

        robustness_path = f"{output_dir}/multiagent_rl_robustness_distribution.png"
        plot_multiagent_robustness_distribution(
            robustness, robustness_path,
            single_agent_collusion=sa_collusion_for_robust,
        )

        robustness_prices_path = f"{output_dir}/multiagent_rl_robustness_prices.png"
        plot_multiagent_robustness_prices(
            robustness, nash_price, cartel_price, robustness_prices_path,
        )

        # Per-seed CSV for full transparency
        per_seed_rows = []
        for s, p, ci, outs, gp, gci, uv in zip(
            robustness["seeds"],
            robustness["converged_prices"],
            robustness["collusion_indices"],
            robustness["converged_outputs"],
            robustness["greedy_prices"],
            robustness["greedy_collusion_indices"],
            robustness["q_undervisited_pcts"],
        ):
            per_seed_rows.append({
                "seed": s,
                "converged_price": round(float(p), 4),
                "collusion_index": round(float(ci), 4),
                "greedy_price": round(float(gp), 4),
                "greedy_collusion_index": round(float(gci), 4),
                "q_undervisited_pct": round(float(uv), 4),
                **{f"q_{k}": round(float(v), 4) for k, v in outs.items()},
            })
        robustness_csv_path = f"{output_dir}/multiagent_rl_robustness_per_seed.csv"
        pd.DataFrame(per_seed_rows).to_csv(robustness_csv_path, index=False)

        robustness_stats = {
            "robustness_n_seeds": int(robustness["n_seeds"]),
            "mean_collusion_index": round(float(robustness["mean_collusion_index"]), 4),
            "std_collusion_index": round(float(robustness["std_collusion_index"]), 4),
            "ci_95_low": round(float(robustness["ci_95_low"]), 4),
            "ci_95_high": round(float(robustness["ci_95_high"]), 4),
            "mean_converged_price": round(float(robustness["mean_converged_price"]), 4),
            "std_converged_price": round(float(robustness["std_converged_price"]), 4),
            "duopoly_mean_greedy_price": round(float(robustness["mean_greedy_price"]), 4),
            "duopoly_mean_greedy_collusion_index": round(
                float(robustness["mean_greedy_collusion_index"]), 4),
            "duopoly_mean_q_undervisited_pct": round(
                float(robustness["mean_q_undervisited_pct"]), 4),
        }
        # NB. do NOT overwrite ``all_seed_outcomes`` here — it holds the
        # **headline triopoly** trajectories (see §16.1), which is what the
        # price-convergence and outputs figures need to display.  The
        # duopoly trajectories from this batch feed §16.2 only.

    # ── Multi-seed price + outputs plots — uses the *headline triopoly*
    # outcomes when robustness is enabled (so the figures match §16.1's
    # headline table); falls back to single-seed ``ma_outcome`` otherwise.
    outcomes_for_plot = all_seed_outcomes if all_seed_outcomes else ma_outcome
    plot_multiagent_price_convergence(
        outcomes_for_plot, nash_price, cartel_price, price_path,
    )
    plot_multiagent_outputs(
        outcomes_for_plot, nash_q, cartel_q, params.players, outputs_path,
    )

    # --- 6. Emergent punishment detection (Folk-Theorem / Green-Porter signature) ---
    episodes = detect_punishment_episodes(ma_outcome)
    total_steps = len(ma_outcome.price_history)
    punishment_stats = compute_punishment_statistics(episodes, total_steps=total_steps)

    punish_detect_path = f"{output_dir}/multiagent_rl_punishment_detection.png"
    plot_punishment_detection(
        ma_outcome, episodes, nash_price, cartel_price, punish_detect_path,
    )

    punish_anatomy_path = f"{output_dir}/multiagent_rl_punishment_anatomy.png"
    plot_punishment_anatomy(ma_outcome, episodes, punish_anatomy_path)

    punishment_summary = {
        "punishment_n_episodes": int(punishment_stats["n_episodes"]),
        "punishment_mean_duration": round(float(punishment_stats["mean_duration"]), 3),
        "punishment_mean_recovery_time": round(float(punishment_stats["mean_recovery_time"]), 3),
        "punishment_mean_drop": round(float(punishment_stats["mean_drop"]), 3),
        "punishment_mean_trough_price": round(float(punishment_stats["mean_trough_price"]), 3),
        "punishment_mean_pre_price": round(float(punishment_stats["mean_pre_price"]), 3),
        "punishment_frequency_per_1000_steps": round(
            float(punishment_stats["punishment_frequency"]), 4,
        ),
    }

    # Per-episode CSV (always written, even if empty, for transparency)
    if episodes:
        episodes_rows = [
            {
                "episode_id": i,
                "start_step": ep.start_step,
                "trough_step": ep.trough_step,
                "recovery_step": ep.recovery_step,
                "cycle_length": ep.recovery_step - ep.start_step,
                "duration_recovery": ep.duration,
                "price_drop": round(ep.price_drop, 4),
                "pre_price": round(ep.pre_price, 4),
                "trough_price": round(ep.trough_price, 4),
                "post_price": round(ep.post_price, 4),
            }
            for i, ep in enumerate(episodes)
        ]
    else:
        episodes_rows = []
    punish_csv_path = f"{output_dir}/multiagent_rl_punishment_episodes.csv"
    pd.DataFrame(episodes_rows).to_csv(punish_csv_path, index=False)

    # --- 7. Learner-count comparison (1 vs 2 vs 3 learning agents) ---
    comparison = run_multiagent_comparison(
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=params.multi_rl,
        nash_price=nash_price,
        cartel_price=cartel_price,
        nash_quantities=nash_q,
        cartel_quantities=cartel_q,
        n_seeds=int(params.multi_rl.learner_comparison_n_seeds),
    )

    comparison_collusion_path = f"{output_dir}/multiagent_rl_learner_comparison_collusion.png"
    plot_learner_comparison_collusion(comparison, comparison_collusion_path)

    comparison_prices_path = f"{output_dir}/multiagent_rl_learner_comparison_prices.png"
    plot_learner_comparison_prices(
        comparison, nash_price, cartel_price, comparison_prices_path,
    )

    comparison_rows = []
    for label in ("single", "duopoly", "triopoly"):
        block = comparison[label]
        row = {
            "regime": label,
            "n_learners": block["n_learners"],
            "learning_players": "+".join(block["learning_players"]),
            "n_seeds": block["n_seeds"],
            "mean_collusion_index": round(block["mean_collusion_index"], 4),
            "std_collusion_index": round(block["std_collusion_index"], 4),
            "mean_converged_price": round(block["mean_converged_price"], 4),
            "std_converged_price": round(block["std_converged_price"], 4),
        }
        for p in params.players:
            row[f"mean_q_{p}"] = round(block["mean_outputs"][p], 4)
            row[f"std_q_{p}"] = round(block["std_outputs"][p], 4)
        comparison_rows.append(row)
    comparison_csv_path = f"{output_dir}/multiagent_rl_learner_comparison.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_csv_path, index=False)

    # --- 8. CSV summary ---
    sa_tail = np.asarray(sa_outcome.price_history[-min(200, len(sa_outcome.price_history)):],
                         dtype=float)
    sa_price = float(sa_tail.mean())
    denom = (cartel_price - nash_price)
    sa_collusion = (sa_price - nash_price) / denom if abs(denom) > 1e-9 else 0.0

    # ── Random-policy baseline (audit Option B) ──────────────────────────
    # Sanity-check anchor for the collusion index: agents that sample
    # actions uniformly should land near CI ≈ 0 if the action grid is
    # unbiased.  Cheap (1 episode-worth of steps per seed, no learning).
    random_baseline_params = replace(
        params.multi_rl,
        episodes=1,
        steps_per_episode=int(params.multi_rl.steps_per_episode),
    )
    rb = run_random_policy_baseline(
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=random_baseline_params,
        nash_price=nash_price,
        cartel_price=cartel_price,
        n_seeds=20,
    )

    summary_rows = [
        {
            "regime": "nash_benchmark",
            "P": round(nash_price, 4),
            "q_OPEC": round(nash_q["OPEC"], 4),
            "q_US": round(nash_q["US"], 4),
            "q_RUS": round(nash_q["RUS"], 4),
            "collusion_index": 0.0,
        },
        {
            "regime": "random_policy_baseline",
            "P": round(rb["mean_price"], 4),
            "q_OPEC": float("nan"),
            "q_US": float("nan"),
            "q_RUS": float("nan"),
            "collusion_index": round(rb["mean_collusion_index"], 4),
            "random_baseline_n_seeds": int(rb["n_seeds"]),
            "random_baseline_std_collusion_index": round(rb["std_collusion_index"], 4),
            "random_baseline_ci_95_low": round(rb["ci_95_low"], 4),
            "random_baseline_ci_95_high": round(rb["ci_95_high"], 4),
        },
        {
            "regime": "single_agent_rl",
            "P": round(sa_price, 4),
            "q_OPEC": round(float(np.mean([q["OPEC"] for q in sa_outcome.q_history[-200:]])), 4),
            "q_US": round(float(np.mean([q["US"] for q in sa_outcome.q_history[-200:]])), 4),
            "q_RUS": round(float(np.mean([q["RUS"] for q in sa_outcome.q_history[-200:]])), 4),
            "collusion_index": round(sa_collusion, 4),
        },
        {
            "regime": "multi_agent_rl",
            "P": round(ma_outcome.converged_price, 4),
            "q_OPEC": round(ma_outcome.converged_outputs["OPEC"], 4),
            "q_US": round(ma_outcome.converged_outputs["US"], 4),
            "q_RUS": round(ma_outcome.converged_outputs["RUS"], 4),
            "collusion_index": round(ma_outcome.collusion_index, 4),
            **robustness_stats,
            **triopoly_headline_stats,
            **punishment_summary,
        },
        {
            "regime": "cartel_benchmark",
            "P": round(cartel_price, 4),
            "q_OPEC": round(cartel_q["OPEC"], 4),
            "q_US": round(cartel_q["US"], 4),
            "q_RUS": round(cartel_q["RUS"], 4),
            "collusion_index": 1.0,
        },
    ]
    summary_path = f"{output_dir}/multiagent_rl_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    artifacts = {
        "multiagent_rl_price_plot": price_path,
        "multiagent_rl_outputs_plot": outputs_path,
        "multiagent_rl_vs_singleagent_plot": vs_path,
        "multiagent_rl_decomposition_plot": decomp_path,
        "multiagent_rl_summary_csv": summary_path,
        "multiagent_rl_punishment_detection_plot": punish_detect_path,
        "multiagent_rl_punishment_anatomy_plot": punish_anatomy_path,
        "multiagent_rl_punishment_episodes_csv": punish_csv_path,
        "multiagent_rl_learner_comparison_collusion_plot": comparison_collusion_path,
        "multiagent_rl_learner_comparison_prices_plot": comparison_prices_path,
        "multiagent_rl_learner_comparison_csv": comparison_csv_path,
    }
    if robustness_path is not None:
        artifacts["multiagent_rl_robustness_distribution_plot"] = robustness_path
    if robustness_prices_path is not None:
        artifacts["multiagent_rl_robustness_prices_plot"] = robustness_prices_path
    if robustness_csv_path is not None:
        artifacts["multiagent_rl_robustness_per_seed_csv"] = robustness_csv_path
    return artifacts


# ---------------------------------------------------------------------------
# Multi-agent RL stress-tests (audit Part III — option C)
# ---------------------------------------------------------------------------

def run_marl_stress_tests(
    params: SimulationParams,
    output_dir: str,
    *,
    gamma_values: Optional[List[float]] = None,
    n_seeds: Optional[int] = None,
    episodes_override: Optional[int] = None,
) -> Dict[str, str]:
    """Run the audit Part III stress-tests around the triopoly MARL baseline.

    Five experiments — each calibrated to a hypothesis from Parts I–II of the
    report — are executed, plotted, and serialised to CSV:

    1. **γ-sweep** (patience / Folk Theorem) — re-trains MARL across a grid
       of discount factors.
    2. **Demand shocks during training** (Green-Porter) — injects two
       negative shocks of −20 $/bbl on the demand intercept.
    3. **Forced deviation** — pins OPEC's output at the Cournot Nash level
       for 100 steps mid-training and measures the price response.
    4. **Stackelberg-MARL** — re-runs the single-learner mode for each
       possible leader and compares to the static Stackelberg equilibrium.
    5. **Capacity activation** — trains MARL with and without
       :class:`CapacityParams` enabled.

    Plus: policy heatmaps for each learning agent of the baseline triopoly.

    To keep the CPU budget reasonable the function uses a *reduced training
    schedule* by default: ``episodes_override`` defaults to
    ``stress_episodes_fraction * params.multi_rl.episodes`` and ``n_seeds``
    to ``params.multi_rl.stress_n_seeds`` (both lifted from
    :class:`MultiAgentRLParams`).  Pass explicit kwargs to override.
    """
    if n_seeds is None:
        n_seeds = int(params.multi_rl.stress_n_seeds)

    # --- Benchmarks ---
    nash = cournot_equilibrium(params.players, params.demand, params.costs, params.capacities)
    cartel = cartel_quotas(params.players, params.demand, params.costs, params.capacities)
    nash_price = float(nash.price)
    cartel_price = float(cartel.quota_price)

    # Capacity-on benchmarks (for the capacity stress-test)
    caps_on = replace(params.capacities, enabled=True)
    nash_cap = cournot_equilibrium(params.players, params.demand, params.costs, caps_on)
    cartel_cap = cartel_quotas(params.players, params.demand, params.costs, caps_on)

    # Stackelberg benchmarks (static): leader quantity AND clearing price.
    # Both are needed for an honest MARL-vs-static comparison: the leader
    # quantity tells us whether the learner reproduces the *commitment*
    # decision, while the clearing price tells us whether the *market
    # outcome* matches Stackelberg — these can diverge under repeated play.
    stack_static_q: Dict[str, float] = {}
    stack_static_p: Dict[str, float] = {}
    for leader in params.players:
        sr = stackelberg_equilibrium(
            leader, params.players, params.demand, params.costs, params.capacities,
            nash_result=nash,
        )
        stack_static_q[leader] = float(sr.quantities.get(leader, 0.0))
        stack_static_p[leader] = float(sr.price)

    # Reduce training budget to keep stress-tests tractable.
    if episodes_override is None:
        frac = float(params.multi_rl.stress_episodes_fraction)
        frac = max(0.05, min(1.0, frac))
        episodes_override = max(50, int(round(params.multi_rl.episodes * frac)))
    stress_params = replace(params.multi_rl, episodes=int(episodes_override))

    triopoly_learners = list(params.players)
    total_steps = int(stress_params.episodes * stress_params.steps_per_episode)

    artifacts: Dict[str, str] = {}

    # ------------------------------------------------------------------ 1. γ
    if gamma_values is None:
        gamma_values = [0.50, 0.70, 0.85, 0.95, 0.99]
    gamma_result = run_marl_gamma_sweep(
        learning_players=triopoly_learners,
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=stress_params,
        nash_price=nash_price,
        cartel_price=cartel_price,
        gamma_values=gamma_values,
        n_seeds=n_seeds,
        return_per_seed=True,
    )
    gamma_rows = gamma_result["aggregated"]
    gamma_per_seed = gamma_result["per_seed"]
    gamma_csv = f"{output_dir}/marl_gamma_sweep.csv"
    pd.DataFrame([{k: round(float(v), 4) if isinstance(v, (int, float)) else v
                   for k, v in row.items()} for row in gamma_rows]).to_csv(
        gamma_csv, index=False,
    )
    gamma_per_seed_csv = f"{output_dir}/marl_gamma_sweep_per_seed.csv"
    pd.DataFrame([{k: round(float(v), 4) if isinstance(v, (int, float)) else v
                   for k, v in row.items()} for row in gamma_per_seed]).to_csv(
        gamma_per_seed_csv, index=False,
    )
    gamma_plot = f"{output_dir}/marl_gamma_sweep.png"
    plot_marl_gamma_sweep(gamma_rows, gamma_plot)
    artifacts["marl_gamma_sweep_csv"] = gamma_csv
    artifacts["marl_gamma_sweep_per_seed_csv"] = gamma_per_seed_csv
    artifacts["marl_gamma_sweep_plot"] = gamma_plot

    # --------------------------------------------------------------- 2. shocks
    shock_window = max(200, total_steps // 10)
    shock_schedule = [
        (total_steps // 3, total_steps // 3 + shock_window, -20.0),
        (2 * total_steps // 3, 2 * total_steps // 3 + shock_window, -20.0),
    ]
    shock_result = run_marl_shock_experiment(
        learning_players=triopoly_learners,
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=stress_params,
        nash_price=nash_price,
        cartel_price=cartel_price,
        shock_schedule=shock_schedule,
        n_seeds=n_seeds,
    )
    shock_rows = [{
        "experiment": "marl_demand_shock",
        "shock_size_delta_a": -20.0,
        "shock_window": shock_window,
        "shocks": ";".join(f"{s[0]}-{s[1]}" for s in shock_schedule),
        **{k: round(float(v), 4) for k, v in shock_result.items()
           if isinstance(v, (int, float))},
    }]
    shock_csv = f"{output_dir}/marl_shock_experiment.csv"
    pd.DataFrame(shock_rows).to_csv(shock_csv, index=False)
    shock_plot = f"{output_dir}/marl_shock_response.png"
    plot_marl_shock_response(
        shock_result["headline_outcome"], shock_schedule,
        nash_price, cartel_price, shock_plot,
    )
    artifacts["marl_shock_csv"] = shock_csv
    artifacts["marl_shock_plot"] = shock_plot

    # -------------------------------------------------------- 3. forced deviation
    deviation_start = int(0.75 * total_steps)
    deviation_duration = max(50, total_steps // 40)
    nash_q_opec = float(nash.quantities["OPEC"])
    deviation_result = run_marl_forced_deviation(
        learning_players=triopoly_learners,
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=stress_params,
        nash_price=nash_price,
        cartel_price=cartel_price,
        deviator="OPEC",
        deviation_q=nash_q_opec,
        deviation_start=deviation_start,
        deviation_duration=deviation_duration,
        n_seeds=n_seeds,
    )
    dev_csv = f"{output_dir}/marl_forced_deviation.csv"
    pd.DataFrame([{
        "experiment": "marl_forced_deviation",
        "deviator": deviation_result["deviator"],
        "deviation_q": deviation_result["deviation_q"],
        "deviation_start": deviation_result["deviation_start"],
        "deviation_duration": deviation_result["deviation_duration"],
        "mean_pre_price": round(deviation_result["mean_pre_price"], 4),
        "mean_during_price": round(deviation_result["mean_during_price"], 4),
        "mean_post_price": round(deviation_result["mean_post_price"], 4),
        **{k: round(float(v), 4) for k, v in deviation_result.items()
           if isinstance(v, (int, float)) and k not in {
               "deviation_q", "deviation_start", "deviation_duration",
               "mean_pre_price", "mean_during_price", "mean_post_price"
           }},
    }]).to_csv(dev_csv, index=False)
    dev_plot = f"{output_dir}/marl_forced_deviation.png"
    plot_marl_forced_deviation(
        deviation_result["all_outcomes"], deviation_result["deviator"],
        deviation_result["deviation_start"], deviation_result["deviation_duration"],
        nash_price, cartel_price,
        deviation_result["mean_pre_price"],
        deviation_result["mean_during_price"],
        deviation_result["mean_post_price"],
        dev_plot,
    )
    artifacts["marl_deviation_csv"] = dev_csv
    artifacts["marl_deviation_plot"] = dev_plot

    # ----------------------------------------------- 4. Stackelberg-MARL comparison
    stack_rows = run_marl_stackelberg_comparison(
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=stress_params,
        nash_price=nash_price,
        cartel_price=cartel_price,
        static_leader_outputs=stack_static_q,
        static_leader_prices=stack_static_p,
        n_seeds=n_seeds,
    )
    stack_csv = f"{output_dir}/marl_stackelberg_comparison.csv"
    pd.DataFrame([{k: round(float(v), 4) if isinstance(v, (int, float)) else v
                   for k, v in row.items()} for row in stack_rows]).to_csv(stack_csv, index=False)
    stack_plot = f"{output_dir}/marl_stackelberg_comparison.png"
    plot_marl_stackelberg_comparison(stack_rows, stack_plot)
    artifacts["marl_stackelberg_csv"] = stack_csv
    artifacts["marl_stackelberg_plot"] = stack_plot

    # --------------------------------------------------------- 5. capacities
    cap_result = run_marl_capacity_experiment(
        learning_players=triopoly_learners,
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=stress_params,
        nash_price_unconstrained=nash_price,
        cartel_price_unconstrained=cartel_price,
        nash_price_constrained=float(nash_cap.price),
        cartel_price_constrained=float(cartel_cap.quota_price),
        n_seeds=n_seeds,
    )
    cap_rows = []
    for label, block in cap_result.items():
        row = {"regime": label}
        row.update({k: round(float(v), 4) if isinstance(v, (int, float)) else v
                    for k, v in block.items()})
        cap_rows.append(row)
    cap_csv = f"{output_dir}/marl_capacity_experiment.csv"
    pd.DataFrame(cap_rows).to_csv(cap_csv, index=False)
    cap_plot = f"{output_dir}/marl_capacity_experiment.png"
    plot_marl_capacity_comparison(
        cap_result["unconstrained"], cap_result["constrained"], cap_plot,
    )
    artifacts["marl_capacity_csv"] = cap_csv
    artifacts["marl_capacity_plot"] = cap_plot

    # --------------------------------------------- 6. Triopoly policy heatmaps
    triopoly_outcome = train_multiagent_ql(
        learning_players=triopoly_learners,
        all_players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=stress_params,
        nash_price=nash_price,
        cartel_price=cartel_price,
        seed=params.seed,
    )
    matrices: Dict[str, np.ndarray] = {}
    q_centres_per_player: Dict[str, np.ndarray] = {}
    price_centres_ref: np.ndarray = np.array([])
    for p in triopoly_learners:
        mat, pc, qc = policy_heatmap(triopoly_outcome, p, metric="argmax")
        matrices[p] = mat
        q_centres_per_player[p] = qc
        price_centres_ref = pc
    heatmap_plot = f"{output_dir}/marl_policy_heatmaps.png"
    plot_marl_policy_heatmaps(matrices, price_centres_ref, q_centres_per_player, heatmap_plot)
    artifacts["marl_policy_heatmaps_plot"] = heatmap_plot

    # ----------------------------------------------------------- summary CSV
    summary_rows = [
        {"experiment": "gamma_sweep",       "headline_value": round(gamma_rows[-1]["mean_collusion_index"], 4),
         "headline_metric": "mean_CI at γ_max"},
        {"experiment": "demand_shock",      "headline_value": round(shock_result["mean_collusion_index"], 4),
         "headline_metric": "mean_CI under shocks"},
        {"experiment": "forced_deviation",  "headline_value": round(deviation_result["mean_during_price"], 4),
         "headline_metric": "mean price during forced deviation"},
        {"experiment": "marl_stackelberg",  "headline_value": round(np.mean([r["marl_leader_q_mean"] for r in stack_rows]), 4),
         "headline_metric": "avg MARL leader q (mbd)"},
        {"experiment": "capacities",        "headline_value": round(cap_result["constrained"]["mean_collusion_index"], 4),
         "headline_metric": "mean_CI under capacity constraints"},
    ]
    summary_csv = f"{output_dir}/marl_stress_tests_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    artifacts["marl_stress_tests_summary_csv"] = summary_csv

    return artifacts


# ---------------------------------------------------------------------------
# Stackelberg mode
# ---------------------------------------------------------------------------

def run_stackelberg_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Compute Stackelberg equilibria for all three possible leaders."""

    nash = cournot_equilibrium(params.players, params.demand, params.costs, params.capacities)
    leaders = params.players if params.stackelberg.compare_all_leaders else [params.stackelberg.leader]

    stack_results = {}
    rows = []
    for leader in leaders:
        sr = stackelberg_equilibrium(leader, params.players, params.demand, params.costs,
                                     params.capacities, nash_result=nash)
        stack_results[leader] = sr
        row = {"leader": leader, "P": round(sr.price, 4), "Q": round(sr.total_quantity, 4),
               "leader_advantage": round(sr.leader_advantage, 4),
               "consumer_surplus": round(sr.consumer_surplus, 4),
               "total_welfare": round(sr.total_welfare, 4)}
        for p in params.players:
            row[f"q_{p}"] = round(sr.quantities.get(p, 0.0), 4)
            row[f"profit_{p}"] = round(sr.profits.get(p, 0.0), 4)
        rows.append(row)

    # Add Nash row for comparison
    nash_row = {"leader": "Nash (no leader)", "P": round(nash.price, 4),
                "Q": round(nash.total_quantity, 4), "leader_advantage": 0.0,
                "consumer_surplus": round(nash.consumer_surplus, 4),
                "total_welfare": round(nash.total_welfare, 4)}
    for p in params.players:
        nash_row[f"q_{p}"] = round(nash.quantities.get(p, 0.0), 4)
        nash_row[f"profit_{p}"] = round(nash.profits.get(p, 0.0), 4)
    rows.insert(0, nash_row)

    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/stackelberg_comparison.csv"
    df.to_csv(csv_path, index=False)

    fig_path = f"{output_dir}/stackelberg_comparison.png"
    plot_stackelberg_comparison(rows, params.players, fig_path)

    return {"stackelberg_csv": csv_path, "stackelberg_plot": fig_path}


# ---------------------------------------------------------------------------
# Market power mode
# ---------------------------------------------------------------------------

def run_market_power_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Compute market power metrics across all market structures."""

    duo_res, tri_res = duopoly_vs_triopoly(params.demand, params.costs, params.capacities)
    nash_tri = tri_res

    leaders = params.players
    stack_results = []
    stack_labels = []
    for leader in leaders:
        sr = stackelberg_equilibrium(leader, params.players, params.demand,
                                     params.costs, params.capacities)
        # wrap into a CournotResult-like for market_power_metrics
        from .cournot_static import CournotResult as CR
        cr = CR(
            quantities=sr.quantities,
            price=sr.price,
            total_quantity=sr.total_quantity,
            profits=sr.profits,
            consumer_surplus=sr.consumer_surplus,
            total_welfare=sr.total_welfare,
        )
        stack_results.append(cr)
        stack_labels.append(f"Stackelberg ({leader} leads)")

    all_results = [duo_res, nash_tri] + stack_results
    all_labels = ["Duopoly (US+OPEC)", "Cournot Nash (triopoly)"] + stack_labels

    rows = market_power_table(all_labels, all_results, params.costs)
    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/market_power.csv"
    df.to_csv(csv_path, index=False)

    fig_path = f"{output_dir}/market_power.png"
    plot_market_power(df, fig_path)

    return {"market_power_csv": csv_path, "market_power_plot": fig_path}


# ---------------------------------------------------------------------------
# Coalition / Shapley mode
# ---------------------------------------------------------------------------

def run_coalition_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Compute Shapley values and Folk-theorem critical discount factors."""

    # Shapley values
    shap = shapley_values(params.players, params.demand, params.costs, params.capacities)

    shap_rows = [{"player": p, "shapley_value": round(shap.values[p], 4)} for p in params.players]
    shap_rows.append({"player": "grand_coalition", "shapley_value": round(shap.grand_coalition_value, 4)})
    shap_rows.append({"player": "core_stable", "shapley_value": float(shap.core_stable)})
    df_shap = pd.DataFrame(shap_rows)
    shap_csv = f"{output_dir}/shapley_values.csv"
    df_shap.to_csv(shap_csv, index=False)

    char_rows = [{"coalition": "+".join(k) if k else "empty", "value": round(v, 4)}
                 for k, v in shap.characteristic.items()]
    char_csv = f"{output_dir}/characteristic_function.csv"
    pd.DataFrame(char_rows).to_csv(char_csv, index=False)

    shap_fig = f"{output_dir}/shapley_values.png"
    plot_shapley_values(shap, shap_fig)

    # Folk theorem
    folk = folk_theorem_delta_star(
        params.players, params.demand, params.costs, params.capacities,
        delta_actual=params.repeated.delta,
    )

    folk_rows = [{"player": p,
                  "delta_star": round(folk.delta_star[p], 4),
                  "pi_cooperative": round(folk.pi_cooperative[p], 4),
                  "pi_nash": round(folk.pi_nash[p], 4),
                  "pi_deviation": round(folk.pi_deviation[p], 4)}
                 for p in params.players]
    folk_rows.append({"player": "BINDING",
                      "delta_star": round(folk.delta_binding, 4),
                      "pi_cooperative": float("nan"),
                      "pi_nash": float("nan"),
                      "pi_deviation": float("nan")})
    folk_csv = f"{output_dir}/folk_theorem.csv"
    pd.DataFrame(folk_rows).to_csv(folk_csv, index=False)

    folk_fig = f"{output_dir}/folk_theorem.png"
    plot_folk_theorem(folk, folk_fig)

    return {
        "shapley_csv": shap_csv,
        "characteristic_csv": char_csv,
        "shapley_plot": shap_fig,
        "folk_csv": folk_csv,
        "folk_plot": folk_fig,
    }


# ---------------------------------------------------------------------------
# Stochastic mode
# ---------------------------------------------------------------------------

def run_stochastic_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Run Monte Carlo stochastic Cournot and demand-shock scenario analysis.

    Produces two sets of MC paths:
    1. Pure AR(1) demand shocks (baseline specification).
    2. Jump-diffusion (AR(1) + Poisson supply jumps) — more realistic,
       reproducing the spikes/crashes of real Brent crude oil price series.
    """
    sp = params.stochastic

    stoch_ar1 = simulate_stochastic_cournot(
        players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        T=sp.T,
        n_paths=sp.n_paths,
        sigma=sp.sigma,
        rho=sp.rho,
        seed=params.seed,
    )

    stoch_jump = simulate_stochastic_jump_diffusion(
        players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        T=sp.T,
        n_paths=sp.n_paths,
        sigma=sp.sigma,
        rho=sp.rho,
        jump_intensity=sp.jump_intensity,
        jump_mu=sp.jump_mu,
        jump_sigma=sp.jump_sigma,
        seed=params.seed,
    )

    # Baseline (AR1) outputs — kept for backward compatibility
    bands_fig = f"{output_dir}/stochastic_price_bands.png"
    plot_stochastic_price_bands(stoch_ar1, bands_fig)

    df_stoch = pd.DataFrame({
        "t": range(sp.T),
        "price_mean": stoch_ar1.price_mean,
        "price_p10": stoch_ar1.price_p10,
        "price_p25": stoch_ar1.price_p25,
        "price_p75": stoch_ar1.price_p75,
        "price_p90": stoch_ar1.price_p90,
    })
    stoch_csv = f"{output_dir}/stochastic_price_bands.csv"
    df_stoch.to_csv(stoch_csv, index=False)

    # Jump-diffusion outputs
    jump_fig = f"{output_dir}/stochastic_jump_diffusion.png"
    plot_stochastic_price_bands(stoch_jump, jump_fig)

    df_jump = pd.DataFrame({
        "t": range(sp.T),
        "price_mean": stoch_jump.price_mean,
        "price_p10": stoch_jump.price_p10,
        "price_p25": stoch_jump.price_p25,
        "price_p75": stoch_jump.price_p75,
        "price_p90": stoch_jump.price_p90,
    })
    jump_csv = f"{output_dir}/stochastic_jump_diffusion.csv"
    df_jump.to_csv(jump_csv, index=False)

    # Comparison figure
    comparison_fig = f"{output_dir}/stochastic_comparison.png"
    plot_stochastic_comparison(stoch_ar1, stoch_jump, comparison_fig)

    # Demand shock scenarios
    scenarios = demand_shock_scenario(
        params.players, params.demand, params.costs, params.capacities,
        sp.shock_sizes,
    )
    shock_rows = []
    for s in scenarios:
        row = {"label": s.label, "shock_size": s.shock_size,
               "price_before": round(s.price_before, 4),
               "price_after": round(s.price_after, 4),
               "price_change_pct": s.price_change_pct}
        for p in params.players:
            row[f"q_after_{p}"] = round(s.quantities_after.get(p, 0.0), 4)
            row[f"profit_after_{p}"] = round(s.profit_after.get(p, 0.0), 4)
        shock_rows.append(row)
    df_shock = pd.DataFrame(shock_rows)
    shock_csv = f"{output_dir}/demand_shock_scenarios.csv"
    df_shock.to_csv(shock_csv, index=False)

    shock_fig = f"{output_dir}/demand_shock_scenarios.png"
    plot_demand_shock_scenarios(df_shock, params.players, shock_fig)

    # ------------------------------------------------------------------
    # Green-Porter repeated game under stochastic demand
    # ------------------------------------------------------------------
    from .stochastic import (
        simulate_green_porter,
        green_porter_delta_star,
        convergence_after_shock as _convergence_after_shock,
    )

    gp = params.green_porter

    gp_ar1 = simulate_green_porter(
        params.players, params.demand, params.costs, params.capacities,
        stoch=sp, gp=gp, delta=params.repeated.delta,
        use_jumps=False, seed=params.seed,
    )
    gp_jump = simulate_green_porter(
        params.players, params.demand, params.costs, params.capacities,
        stoch=sp, gp=gp, delta=params.repeated.delta,
        use_jumps=True, seed=params.seed + 1,
    )

    gp_regimes_fig = f"{output_dir}/green_porter_regimes.png"
    plot_green_porter_regimes(gp_ar1, gp_regimes_fig)

    coop_survival_fig = f"{output_dir}/green_porter_coop_survival.png"
    plot_cooperation_survival(gp_ar1, gp_jump, coop_survival_fig)

    delta_ar1 = green_porter_delta_star(
        params.players, params.demand, params.costs, params.capacities,
        stoch=sp, gp=gp, delta_actual=params.repeated.delta,
    )
    delta_jump = green_porter_delta_star(
        params.players, params.demand, params.costs, params.capacities,
        stoch=sp, gp=gp, delta_actual=params.repeated.delta,
    )

    from .coalition import folk_theorem_delta_star as _folk_det
    det_folk = _folk_det(params.players, params.demand, params.costs,
                         params.capacities, params.repeated.delta)

    delta_fig = f"{output_dir}/delta_star_comparison.png"
    plot_delta_star_comparison(det_folk, delta_ar1, delta_jump,
                              params.repeated.delta, delta_fig)

    conv_data = _convergence_after_shock(gp_ar1)
    conv_fig = f"{output_dir}/convergence_after_shock.png"
    plot_convergence_after_shock(conv_data, conv_fig)

    # CSVs
    gp_summary_rows = []
    for label, gp_out in [("AR1", gp_ar1), ("Jump", gp_jump)]:
        row = {
            "specification": label,
            "trigger_price": round(gp_out.trigger_price, 2),
            "coop_fraction_steady": round(float(gp_out.coop_fraction[-20:].mean()), 4),
            "mean_coop_spell": round(gp_out.mean_coop_spell, 2),
            "mean_punish_spell": round(gp_out.mean_punish_spell, 2),
            "total_triggers": gp_out.total_trigger_count,
            "false_triggers": gp_out.false_trigger_count,
        }
        for p in params.players:
            row[f"E_disc_profit_{p}"] = round(gp_out.expected_discounted_profit[p], 2)
        gp_summary_rows.append(row)
    gp_csv = f"{output_dir}/green_porter_summary.csv"
    pd.DataFrame(gp_summary_rows).to_csv(gp_csv, index=False)

    delta_rows = []
    for p in params.players:
        delta_rows.append({
            "player": p,
            "delta_star_deterministic": round(det_folk.delta_star[p], 4),
            "delta_star_stochastic_AR1": delta_ar1.delta_star_stochastic,
            "alpha_AR1": delta_ar1.alpha,
            "delta_actual": params.repeated.delta,
            "sustainable": delta_ar1.cooperation_sustainable,
        })
    delta_csv = f"{output_dir}/delta_star_comparison.csv"
    pd.DataFrame(delta_rows).to_csv(delta_csv, index=False)

    conv_csv = f"{output_dir}/convergence_after_shock.csv"
    if conv_data is not None:
        pd.DataFrame({
            "t_after_shock": range(conv_data.horizon),
            "mean_price": conv_data.mean_price_path,
            "std_price": conv_data.std_price_path,
            "mean_quantity": conv_data.mean_quantity_path,
        }).to_csv(conv_csv, index=False)

    return {
        "stochastic_bands_csv": stoch_csv,
        "stochastic_bands_plot": bands_fig,
        "stochastic_jump_csv": jump_csv,
        "stochastic_jump_plot": jump_fig,
        "stochastic_comparison_plot": comparison_fig,
        "shock_scenarios_csv": shock_csv,
        "shock_scenarios_plot": shock_fig,
        "gp_regimes_plot": gp_regimes_fig,
        "gp_coop_survival_plot": coop_survival_fig,
        "gp_delta_plot": delta_fig,
        "gp_convergence_plot": conv_fig,
        "gp_summary_csv": gp_csv,
        "gp_delta_csv": delta_csv,
        "gp_convergence_csv": conv_csv,
    }


# ---------------------------------------------------------------------------
# Evolutionary game mode
# ---------------------------------------------------------------------------

def run_evolutionary_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Run evolutionary game dynamics analysis.

    Models OPEC-member producers choosing between Cooperate / Defect / Punish.
    Replicator dynamics identify evolutionarily stable strategies (ESS) and
    produce phase diagrams on the 2-D barycentric simplex.

    Key outputs:
    - Payoff matrices (2×2 and 3×3) derived from calibrated Cournot model.
    - 1-D phase portrait for 2-strategy game (validates PD structure).
    - Triangular phase diagram for 3-strategy game (novel policy insight).
    - Trajectories showing convergence or cycling from multiple ICs.
    """
    evo = run_evolutionary_game(
        players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=params.evo,
        focal_player="OPEC",
        seed=params.seed,
    )

    phase_2_path = f"{output_dir}/evo_phase_2strategy.png"
    plot_evo_phase_diagram_2strategy(evo, phase_2_path)

    phase_3_path = f"{output_dir}/evo_phase_3strategy.png"
    plot_evo_phase_diagram_3strategy(evo, phase_3_path)

    payoff_matrix_path = f"{output_dir}/evo_payoff_matrix.png"
    plot_evo_payoff_matrix(evo, payoff_matrix_path)

    # Save payoff matrices to CSV
    rows_2 = []
    for i, s_i in enumerate(evo.payoff_2.strategies):
        for j, s_j in enumerate(evo.payoff_2.strategies):
            rows_2.append({"row_strategy": s_i, "col_strategy": s_j,
                           "payoff": round(float(evo.payoff_2.A[i, j]), 4)})
    payoff_2_csv = f"{output_dir}/evo_payoff_2x2.csv"
    pd.DataFrame(rows_2).to_csv(payoff_2_csv, index=False)

    rows_3 = []
    for i, s_i in enumerate(evo.payoff_3.strategies):
        for j, s_j in enumerate(evo.payoff_3.strategies):
            rows_3.append({"row_strategy": s_i, "col_strategy": s_j,
                           "payoff": round(float(evo.payoff_3.A[i, j]), 4)})
    payoff_3_csv = f"{output_dir}/evo_payoff_3x3.csv"
    pd.DataFrame(rows_3).to_csv(payoff_3_csv, index=False)

    # ESS summary
    ess_rows = [{"label": lbl} for lbl in evo.ess_labels_2]
    if evo.interior_eq_2 is not None:
        ess_rows.append({"label": f"interior_x_star={evo.interior_eq_2:.4f}"})
    ess_csv = f"{output_dir}/evo_ess_summary.csv"
    pd.DataFrame(ess_rows).to_csv(ess_csv, index=False)

    # --- Punishment multiplier sensitivity sweep (conditional model) ---
    sweep_cond = punishment_multiplier_sweep(
        players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=params.evo,
        focal_player="OPEC",
        conditional=True,
        seed=params.seed,
    )
    sweep_cond_path = f"{output_dir}/evo_punishment_sweep.png"
    plot_evo_punishment_sweep(sweep_cond, evo.t_grid, sweep_cond_path)

    sweep_rows = []
    for i, m in enumerate(sweep_cond.multipliers):
        sweep_rows.append({
            "multiplier": m,
            "model": "conditional",
            "share_C": round(float(sweep_cond.convergence[i][0]), 4),
            "share_D": round(float(sweep_cond.convergence[i][1]), 4),
            "share_P": round(float(sweep_cond.convergence[i][2]), 4),
            "dominant": sweep_cond.dominant_strategy[i],
            "interior_eq": sweep_cond.has_interior_eq[i],
        })
    # Also run unconditional for comparison
    sweep_uncond = punishment_multiplier_sweep(
        players=params.players,
        demand=params.demand,
        costs=params.costs,
        capacities=params.capacities,
        params=params.evo,
        focal_player="OPEC",
        conditional=False,
        seed=params.seed,
    )
    for i, m in enumerate(sweep_uncond.multipliers):
        sweep_rows.append({
            "multiplier": m,
            "model": "unconditional",
            "share_C": round(float(sweep_uncond.convergence[i][0]), 4),
            "share_D": round(float(sweep_uncond.convergence[i][1]), 4),
            "share_P": round(float(sweep_uncond.convergence[i][2]), 4),
            "dominant": sweep_uncond.dominant_strategy[i],
            "interior_eq": sweep_uncond.has_interior_eq[i],
        })
    sweep_csv = f"{output_dir}/evo_punishment_sweep.csv"
    pd.DataFrame(sweep_rows).to_csv(sweep_csv, index=False)

    return {
        "evo_phase_2_plot": phase_2_path,
        "evo_phase_3_plot": phase_3_path,
        "evo_payoff_matrix_plot": payoff_matrix_path,
        "evo_payoff_2_csv": payoff_2_csv,
        "evo_payoff_3_csv": payoff_3_csv,
        "evo_ess_csv": ess_csv,
        "evo_punishment_sweep_plot": sweep_cond_path,
        "evo_punishment_sweep_csv": sweep_csv,
    }


# ---------------------------------------------------------------------------
# Section A — Bertrand competition orchestrator
# ---------------------------------------------------------------------------

def run_bertrand_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Section A: differentiated-products Bertrand price competition.

    Acts as a robustness check on the Cournot baseline: do the qualitative
    OPEC-dominance and cooperation-premium findings survive when firms
    compete on price rather than quantity?  Computes Bertrand-Nash and
    cooperative outcomes at the calibrated sigma, plus a sigma sweep.
    """
    from .bertrand import (
        bertrand_nash,
        bertrand_cooperative,
        bertrand_sigma_sweep,
        bertrand_vs_cournot_comparison,
    )
    from .plotting import (
        plot_bertrand_nash_equilibrium,
        plot_bertrand_vs_cournot,
        plot_bertrand_sigma_sweep,
        plot_bertrand_cooperation_gap,
    )

    nash = bertrand_nash(params.players, params.costs, params.bertrand, params.demand)
    coop = bertrand_cooperative(params.players, params.costs, params.bertrand,
                                params.demand)
    sweep = bertrand_sigma_sweep(params.players, params.costs, params.bertrand,
                                 params.demand)
    comparison = bertrand_vs_cournot_comparison(
        params.players, params.demand, params.costs, params.capacities,
        params.bertrand,
    )

    nash_path = f"{output_dir}/bertrand_nash_equilibrium.png"
    plot_bertrand_nash_equilibrium(nash, nash_path)

    vs_path = f"{output_dir}/bertrand_vs_cournot.png"
    plot_bertrand_vs_cournot(comparison, params.players, vs_path)

    sweep_path = f"{output_dir}/bertrand_sigma_sweep.png"
    plot_bertrand_sigma_sweep(sweep, params.players, sweep_path)

    coop_gap_path = f"{output_dir}/bertrand_cooperation_gap.png"
    plot_bertrand_cooperation_gap(sweep, params.players, coop_gap_path)

    nash_rows = [{
        "regime": "bertrand_nash",
        "sigma": nash.sigma,
        "average_price": round(nash.average_price, 4),
        "total_quantity": round(nash.total_quantity, 4),
        "consumer_surplus": round(nash.consumer_surplus, 4),
        "total_welfare": round(nash.total_welfare, 4),
        **{f"price_{p}": round(nash.prices[p], 4) for p in params.players},
        **{f"q_{p}":     round(nash.quantities[p], 4) for p in params.players},
        **{f"profit_{p}": round(nash.profits[p], 4)  for p in params.players},
    }, {
        "regime": "bertrand_cooperative",
        "sigma": coop.sigma,
        "average_price": round(coop.average_price, 4),
        "total_quantity": round(coop.total_quantity, 4),
        "consumer_surplus": round(coop.consumer_surplus, 4),
        "total_welfare": round(coop.total_welfare, 4),
        **{f"price_{p}": round(coop.prices[p], 4) for p in params.players},
        **{f"q_{p}":     round(coop.quantities[p], 4) for p in params.players},
        **{f"profit_{p}": round(coop.profits[p], 4)  for p in params.players},
    }]
    nash_csv = f"{output_dir}/bertrand_nash.csv"
    pd.DataFrame(nash_rows).to_csv(nash_csv, index=False)

    sweep_rows = []
    for sigma, n_res, c_res in zip(sweep.sigma_values,
                                   sweep.nash_results, sweep.coop_results):
        row = {
            "sigma": sigma,
            "nash_avg_price": round(n_res.average_price, 4),
            "nash_total_q":   round(n_res.total_quantity, 4),
            "nash_total_profit": round(sum(n_res.profits.values()), 4),
            "coop_avg_price": round(c_res.average_price, 4),
            "coop_total_q":   round(c_res.total_quantity, 4),
            "coop_total_profit": round(sum(c_res.profits.values()), 4),
        }
        for p in params.players:
            row[f"nash_price_{p}"] = round(n_res.prices[p], 4)
            row[f"coop_price_{p}"] = round(c_res.prices[p], 4)
            row[f"nash_q_{p}"]     = round(n_res.quantities[p], 4)
            row[f"coop_q_{p}"]     = round(c_res.quantities[p], 4)
        sweep_rows.append(row)
    sweep_csv = f"{output_dir}/bertrand_sigma_sweep.csv"
    pd.DataFrame(sweep_rows).to_csv(sweep_csv, index=False)

    return {
        "bertrand_nash_plot": nash_path,
        "bertrand_vs_cournot_plot": vs_path,
        "bertrand_sigma_sweep_plot": sweep_path,
        "bertrand_cooperation_gap_plot": coop_gap_path,
        "bertrand_nash_csv": nash_csv,
        "bertrand_sigma_sweep_csv": sweep_csv,
    }


# ---------------------------------------------------------------------------
# Section B — Capacity-constraint analysis orchestrator
# ---------------------------------------------------------------------------

def run_capacity_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Section B: activate the capacity constraints and compare to the baseline.

    Re-runs Nash, cartel, Folk-theorem delta* and Shapley with
    ``capacities.enabled = True`` and contrasts each metric against the
    unconstrained baseline.  Also sweeps OPEC's capacity to show how
    swing-producer capacity translates into market power and into the
    Folk-theorem threshold.
    """
    from .capacity_analysis import (
        run_constrained_vs_unconstrained,
        opec_capacity_sweep,
        comparison_to_dataframe,
        opec_sweep_to_dataframe,
    )
    from .plotting import (
        plot_capacity_constrained_vs_unconstrained,
        plot_capacity_opec_sweep,
        plot_capacity_binding_analysis,
        plot_capacity_folk_theorem,
    )

    comparison = run_constrained_vs_unconstrained(
        params.players, params.demand, params.costs, params.capacities,
    )
    sweep = opec_capacity_sweep(
        params.players, params.demand, params.costs, params.capacities,
        params.capacity_sweep,
    )

    cmp_path = f"{output_dir}/capacity_constrained_vs_unconstrained.png"
    plot_capacity_constrained_vs_unconstrained(comparison, params.players, cmp_path)

    sweep_df = opec_sweep_to_dataframe(sweep)
    sweep_path = f"{output_dir}/capacity_opec_sweep.png"
    plot_capacity_opec_sweep(sweep_df, sweep_path)

    binding_path = f"{output_dir}/capacity_binding_analysis.png"
    plot_capacity_binding_analysis(comparison, params.players, binding_path)

    folk_path = f"{output_dir}/capacity_folk_theorem.png"
    plot_capacity_folk_theorem(comparison, params.players, folk_path)

    cmp_df = comparison_to_dataframe(comparison, params.players)
    cmp_csv = f"{output_dir}/capacity_comparison.csv"
    cmp_df.to_csv(cmp_csv, index=False)

    sweep_csv = f"{output_dir}/capacity_opec_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    return {
        "capacity_comparison_plot": cmp_path,
        "capacity_opec_sweep_plot": sweep_path,
        "capacity_binding_plot": binding_path,
        "capacity_folk_plot": folk_path,
        "capacity_comparison_csv": cmp_csv,
        "capacity_opec_sweep_csv": sweep_csv,
    }


# ---------------------------------------------------------------------------
# Section C — Welfare and deadweight-loss orchestrator
# ---------------------------------------------------------------------------

def run_welfare_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Section C: welfare decomposition and carbon-tax interaction.

    Computes consumer surplus, producer surplus, total welfare and DWL
    under perfect competition, Nash, three Stackelberg leaders and the
    cartel.  Then sweeps a carbon tax to see whether it shrinks or grows
    the collusion premium.
    """
    from .welfare import (
        welfare_decomposition,
        carbon_tax_interaction,
        carbon_results_to_lite,
        decomposition_to_dataframe,
        carbon_results_to_dataframe,
    )
    from .plotting import (
        plot_welfare_decomposition,
        plot_welfare_dwl_comparison,
        plot_welfare_carbon_tax,
        plot_welfare_surplus_distribution,
    )

    bundle = welfare_decomposition(
        params.players, params.demand, params.costs, params.capacities,
    )
    decompositions = bundle["decompositions"]
    carbon_results = carbon_tax_interaction(
        params.players, params.demand, params.costs, params.capacities,
        params.welfare.carbon_tax_values,
    )

    decomp_path = f"{output_dir}/welfare_decomposition.png"
    plot_welfare_decomposition(decompositions, decomp_path)

    dwl_path = f"{output_dir}/welfare_dwl_comparison.png"
    plot_welfare_dwl_comparison(decompositions, dwl_path)

    carbon_path = f"{output_dir}/welfare_carbon_tax.png"
    plot_welfare_carbon_tax(carbon_results_to_lite(carbon_results), carbon_path)

    surplus_path = f"{output_dir}/welfare_surplus_distribution.png"
    plot_welfare_surplus_distribution(decompositions, params.players, surplus_path)

    decomp_csv = f"{output_dir}/welfare_decomposition.csv"
    decomposition_to_dataframe(decompositions, params.players).to_csv(
        decomp_csv, index=False
    )
    carbon_csv = f"{output_dir}/welfare_carbon_tax.csv"
    carbon_results_to_dataframe(carbon_results).to_csv(carbon_csv, index=False)

    return {
        "welfare_decomposition_plot": decomp_path,
        "welfare_dwl_plot": dwl_path,
        "welfare_carbon_plot": carbon_path,
        "welfare_surplus_plot": surplus_path,
        "welfare_decomposition_csv": decomp_csv,
        "welfare_carbon_csv": carbon_csv,
    }


# ---------------------------------------------------------------------------
# Section D — N-player sensitivity orchestrator
# ---------------------------------------------------------------------------

def run_n_player_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Section D: sweep n in {2, ..., max_players} and study the comparative statics.

    Adds plausible new entrants (Brazil pre-salt, Canada oil sands, Norway)
    one at a time and records Nash/cartel prices, Folk-theorem delta*, HHI,
    and OPEC's profit/Shapley/market-share dilution.
    """
    from .n_player import n_player_sweep, n_player_comparison_table
    from .plotting import (
        plot_n_player_price_quantity,
        plot_n_player_cooperation,
        plot_n_player_opec_power,
    )

    sweep = n_player_sweep(
        params.players, params.demand, params.costs, params.capacities,
        params.n_player,
    )

    pq_path = f"{output_dir}/n_player_price_quantity.png"
    plot_n_player_price_quantity(sweep, pq_path)

    coop_path = f"{output_dir}/n_player_cooperation.png"
    plot_n_player_cooperation(sweep, coop_path)

    opec_path = f"{output_dir}/n_player_opec_power.png"
    plot_n_player_opec_power(sweep, opec_path)

    sweep_csv = f"{output_dir}/n_player_sweep.csv"
    n_player_comparison_table(sweep).to_csv(sweep_csv, index=False)

    return {
        "n_player_price_quantity_plot": pq_path,
        "n_player_cooperation_plot": coop_path,
        "n_player_opec_power_plot": opec_path,
        "n_player_sweep_csv": sweep_csv,
    }


# ---------------------------------------------------------------------------
# Section E — Correlated equilibrium orchestrator
# ---------------------------------------------------------------------------

def run_correlated_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Section E: correlated equilibrium via linear programming.

    Tests whether a *mediator* (e.g. the OPEC Secretariat) could
    Pareto-improve on Nash by recommending production levels privately
    and incentive-compatibly.  Reports the CE under three objectives —
    welfare, joint profit and max-min profit.
    """
    from .correlated_eq import ce_vs_nash_comparison, comparison_to_dataframe
    from .plotting import (
        plot_correlated_eq_comparison,
        plot_correlated_eq_welfare,
        plot_correlated_eq_support,
    )

    comparison = ce_vs_nash_comparison(
        params.players, params.demand, params.costs, params.capacities,
        params.correlated_eq,
    )

    cmp_path = f"{output_dir}/correlated_eq_comparison.png"
    plot_correlated_eq_comparison(comparison, params.players, cmp_path)

    welfare_path = f"{output_dir}/correlated_eq_welfare.png"
    plot_correlated_eq_welfare(comparison, welfare_path)

    # Pick the welfare CE for the support heatmap (most economically interesting)
    support_obj = "max_welfare" if "max_welfare" in comparison.ce_results \
        else next(iter(comparison.ce_results))
    support_path = f"{output_dir}/correlated_eq_support.png"
    plot_correlated_eq_support(
        comparison.ce_results[support_obj],
        comparison.players,
        comparison.action_grid,
        support_path,
    )

    cmp_csv = f"{output_dir}/correlated_eq_comparison.csv"
    comparison_to_dataframe(comparison).to_csv(cmp_csv, index=False)

    return {
        "correlated_eq_comparison_plot": cmp_path,
        "correlated_eq_welfare_plot": welfare_path,
        "correlated_eq_support_plot": support_path,
        "correlated_eq_comparison_csv": cmp_csv,
    }


# ---------------------------------------------------------------------------
# Section F — Empirical validation orchestrator
# ---------------------------------------------------------------------------

def run_empirical_mode(params: SimulationParams, output_dir: str) -> Dict[str, str]:
    """Section F: validate the model qualitatively against three historical
    oil-market price wars (1985, 2014, 2020).
    """
    from .empirical import run_empirical_validation, validation_to_dataframe
    from .plotting import (
        plot_empirical_price_wars,
        plot_empirical_mechanism_match,
        plot_empirical_model_vs_history,
    )

    result = run_empirical_validation(
        params.players, params.demand, params.costs, params.capacities,
        params.repeated, params.stackelberg, params.empirical.episodes,
    )

    if not result.episodes:
        return {}

    pw_path = f"{output_dir}/empirical_price_wars.png"
    plot_empirical_price_wars(result, pw_path)

    mech_path = f"{output_dir}/empirical_mechanism_match.png"
    plot_empirical_mechanism_match(result, mech_path)

    sc_path = f"{output_dir}/empirical_model_vs_history.png"
    plot_empirical_model_vs_history(result, sc_path)

    csv_path = f"{output_dir}/empirical_validation.csv"
    validation_to_dataframe(result).to_csv(csv_path, index=False)

    return {
        "empirical_price_wars_plot": pw_path,
        "empirical_mechanism_match_plot": mech_path,
        "empirical_model_vs_history_plot": sc_path,
        "empirical_validation_csv": csv_path,
    }
