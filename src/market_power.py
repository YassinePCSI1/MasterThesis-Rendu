"""Market power metrics for the global crude oil market.

Provides per-player and aggregate statistics that economists and regulators use
to assess competitive intensity.  All metrics are dimensionless or in $/bbl.

Lerner Index  L_i = (P - c_i) / P
    Measures the price-cost mark-up as a fraction of price.
    L = 0 → perfect competition; L → 1 → monopoly.

Herfindahl-Hirschman Index  HHI = Σ_i s_i² × 10 000
    where s_i = q_i / Q is player i's market share.
    DOJ thresholds: < 1 500 unconcentrated, 1 500–2 500 moderate,
    > 2 500 highly concentrated.

Price-cost margin  PCM_i = (P - c_i) × q_i  (absolute, in $M/day)

Concentration ratio  CR_k = Σ_{top-k} s_i

Relative market power index  RMP_i = (s_i × L_i)  (Dansby-Willig)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import CostParams
from .cournot_static import CournotResult


@dataclass
class MarketPowerResult:
    lerner_index: Dict[str, float]
    market_share: Dict[str, float]
    hhi: float
    price_cost_margin: Dict[str, float]
    cr1: float
    cr2: float
    relative_market_power: Dict[str, float]
    average_lerner: float


def market_power_metrics(result: CournotResult, costs: CostParams) -> MarketPowerResult:
    """Compute all market-power metrics from a Cournot (or Stackelberg) result."""
    P = result.price
    Q = result.total_quantity
    cost_map = {"US": costs.c_us, "OPEC": costs.c_opec, "RUS": costs.c_rus}

    players = list(result.quantities.keys())

    lerner: Dict[str, float] = {}
    share: Dict[str, float] = {}
    pcm: Dict[str, float] = {}
    rmp: Dict[str, float] = {}

    for p in players:
        q_i = result.quantities[p]
        c_i = cost_map.get(p, 0.0)
        L_i = (P - c_i) / P if P > 0 else 0.0
        s_i = q_i / Q if Q > 0 else 0.0
        lerner[p] = round(L_i, 6)
        share[p] = round(s_i, 6)
        pcm[p] = round((P - c_i) * q_i, 4)
        rmp[p] = round(s_i * L_i, 6)

    hhi = round(sum(s ** 2 for s in share.values()) * 10_000, 2)

    sorted_shares = sorted(share.values(), reverse=True)
    cr1 = round(sorted_shares[0], 6) if len(sorted_shares) >= 1 else 0.0
    cr2 = round(sum(sorted_shares[:2]), 6) if len(sorted_shares) >= 2 else cr1

    avg_lerner = round(sum(share[p] * lerner[p] for p in players), 6)

    return MarketPowerResult(
        lerner_index=lerner,
        market_share=share,
        hhi=hhi,
        price_cost_margin=pcm,
        cr1=cr1,
        cr2=cr2,
        relative_market_power=rmp,
        average_lerner=avg_lerner,
    )


def market_power_table(
    labels: List[str],
    results: List[CournotResult],
    costs: CostParams,
) -> List[Dict]:
    """Build a list of dicts suitable for pd.DataFrame across multiple market structures."""
    rows = []
    for label, result in zip(labels, results):
        mp = market_power_metrics(result, costs)
        row: Dict = {"model": label, "P": round(result.price, 4),
                     "Q": round(result.total_quantity, 4), "HHI": mp.hhi,
                     "CR1": round(mp.cr1 * 100, 2), "CR2": round(mp.cr2 * 100, 2),
                     "avg_lerner": round(mp.average_lerner, 4)}
        for p in result.quantities:
            row[f"share_{p}"] = round(mp.market_share.get(p, 0.0) * 100, 2)
            row[f"lerner_{p}"] = round(mp.lerner_index.get(p, 0.0), 4)
        rows.append(row)
    return rows
