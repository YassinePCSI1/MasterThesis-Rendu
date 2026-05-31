"""Demand and surplus functions for the global crude oil market."""
from __future__ import annotations

from dataclasses import dataclass

from .config import DemandParams


@dataclass
class DemandResult:
    price: float
    quantity: float


def price_from_quantity(Q: float, params: DemandParams) -> float:
    """Compute price from total quantity under linear demand.

    P(Q) = a - b Q, optionally floored at price_floor.
    """

    price = params.a - params.b * Q
    if params.price_floor is not None:
        price = max(params.price_floor, price)
    return price


def consumer_surplus(Q: float, price: float, params: DemandParams) -> float:
    """Consumer surplus for linear demand.

    For P(Q) = a - bQ, inverse demand intercept is a.
    CS = 0.5 * (a - price) * Q.
    """

    return 0.5 * (params.a - price) * Q
