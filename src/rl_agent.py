"""Simple Q-learning agent for output decisions in repeated Cournot."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import AdjustmentCostParams, CapacityParams, CostParams, DemandParams, RLParams
from .cournot_repeated import best_response
from .demand import price_from_quantity


@dataclass
class RLOutcome:
    q_history: List[Dict[str, float]]
    price_history: List[float]
    reward_history: List[float]
    q_table: np.ndarray
    action_grid: np.ndarray


def _cost_map(costs: CostParams) -> Dict[str, float]:
    return {"US": costs.c_us, "OPEC": costs.c_opec, "RUS": costs.c_rus}


def _cap_map(caps: CapacityParams) -> Dict[str, float]:
    return {"US": caps.cap_us, "OPEC": caps.cap_opec, "RUS": caps.cap_rus}


def _state_bins(params: RLParams, demand: DemandParams, capacities: CapacityParams) -> Dict[str, np.ndarray]:
    Q_max = sum(_cap_map(capacities).values()) if capacities.enabled else demand.a / demand.b
    price_min = demand.price_floor if demand.price_floor is not None else 0.0
    price_max = demand.a
    bins = {
        "price": np.linspace(price_min, price_max, params.state_bins["price"] + 1)[1:-1],
        "Q": np.linspace(0.0, Q_max, params.state_bins["Q"] + 1)[1:-1],
        "own_q": np.linspace(0.0, Q_max, params.state_bins["own_q"] + 1)[1:-1],
        "others_q": np.linspace(0.0, Q_max, params.state_bins["others_q"] + 1)[1:-1],
    }
    return bins


def _discretize_state(state: Dict[str, float], bins: Dict[str, np.ndarray]) -> Tuple[int, ...]:
    return tuple(int(np.digitize(state[key], bins[key])) for key in ["price", "Q", "own_q", "others_q"])


def _action_grid(cap: float, params: RLParams) -> np.ndarray:
    return np.linspace(0.0, cap, params.action_grid_size)


def train_q_learning(players: List[str],
                     demand: DemandParams,
                     costs: CostParams,
                     capacities: CapacityParams,
                     params: RLParams,
                     learning_player: str = "OPEC",
                     penalty_volatility: float = 0.0,
                     seed: int = 123) -> RLOutcome:
    """Train a Q-learning agent, others follow myopic best responses."""

    rng = np.random.default_rng(seed)
    cost_map = _cost_map(costs)
    cap_map = _cap_map(capacities)
    if capacities.enabled:
        cap = cap_map[learning_player]
    else:
        cap = (demand.a - cost_map[learning_player]) / (2 * demand.b)

    action_grid = _action_grid(cap, params)
    bins = _state_bins(params, demand, capacities)

    q_table = np.zeros(
        (
            params.state_bins["price"] + 1,
            params.state_bins["Q"] + 1,
            params.state_bins["own_q"] + 1,
            params.state_bins["others_q"] + 1,
            params.action_grid_size,
        )
    )
    assert q_table.shape[-1] == params.action_grid_size

    q_history: List[Dict[str, float]] = []
    price_history: List[float] = []
    reward_history: List[float] = []

    for _ in range(params.episodes):
        outputs = {p: 0.0 for p in players}
        last_price = price_from_quantity(0.0, demand)
        last_Q = 0.0

        for _ in range(params.steps_per_episode):
            others_q = sum(outputs[p] for p in players if p != learning_player)
            state = {
                "price": last_price,
                "Q": last_Q,
                "own_q": outputs[learning_player],
                "others_q": others_q,
            }
            state_idx = _discretize_state(state, bins)

            if rng.random() < params.epsilon:
                action_idx = rng.integers(0, params.action_grid_size)
            else:
                action_idx = int(np.argmax(q_table[state_idx]))

            outputs[learning_player] = float(action_grid[action_idx])

            for p in players:
                if p == learning_player:
                    continue
                q_others = sum(outputs[k] for k in players if k != p)
                outputs[p] = best_response(
                    q_others,
                    outputs[p],
                    demand,
                    cost_map[p],
                    capacities,
                    p,
                    adjustment=AdjustmentCostParams(enabled=False, k=0.0),
                )

            Q = sum(outputs.values())
            price = price_from_quantity(Q, demand)
            profit = (price - cost_map[learning_player]) * outputs[learning_player]
            volatility_penalty = penalty_volatility * (outputs[learning_player] - state["own_q"]) ** 2
            reward = profit - volatility_penalty

            next_state = {
                "price": price,
                "Q": Q,
                "own_q": outputs[learning_player],
                "others_q": sum(outputs[p] for p in players if p != learning_player),
            }
            next_idx = _discretize_state(next_state, bins)

            td_target = reward + params.gamma * np.max(q_table[next_idx])
            td_error = td_target - q_table[state_idx + (action_idx,)]
            q_table[state_idx + (action_idx,)] += params.alpha * td_error

            last_price = price
            last_Q = Q

            q_history.append(outputs.copy())
            price_history.append(price)
            reward_history.append(reward)

    return RLOutcome(
        q_history=q_history,
        price_history=price_history,
        reward_history=reward_history,
        q_table=q_table,
        action_grid=action_grid,
    )
