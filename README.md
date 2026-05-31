# Game Theory in Oil Producing Countries
### Master Thesis — Quantitative Model

A fully self-contained Python simulation of strategic interaction in the global
crude oil market, using game theory, reinforcement learning, and evolutionary
dynamics. All outputs are auto-generated into `outputs/`.

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all simulations (recommended)
python main.py --mode all

# 4. Run a single mode
python main.py --mode static
python main.py --mode repeated
python main.py --mode rl
python main.py --mode multiagent_rl
python main.py --mode stackelberg
python main.py --mode coalition
python main.py --mode market_power
python main.py --mode stochastic
python main.py --mode evolutionary

# Extended robustness / policy modes (Sections A–F)
python main.py --mode bertrand        # Price competition with differentiated goods
python main.py --mode capacity        # Capacity-constrained equilibria + sweeps
python main.py --mode welfare         # Consumer/producer surplus + DWL + carbon tax
python main.py --mode n_player        # Sensitivity to 2–6 producers (entry/fragmentation)
python main.py --mode correlated      # Correlated equilibrium via LP (mediator design)
python main.py --mode empirical       # Validation against three historical price wars
```

Outputs are written to `outputs/`. The auto-generated report is at `outputs/report.md`.

---

## Project structure

```
Thesis/
├── main.py                  ← Entry point (CLI)
├── requirements.txt
├── src/
│   ├── config.py            ← All dataclass parameters (fully documented)
│   ├── calibration.py       ← Empirical calibration rationale + validation
│   ├── demand.py            ← Inverse demand function P(Q) = a − bQ
│   ├── costs.py             ← Variable and adjustment cost functions
│   ├── cournot_static.py    ← One-shot Cournot Nash equilibrium
│   ├── cournot_repeated.py  ← Repeated game: myopic BR + inertia
│   ├── cooperation_punishment.py ← Cooperative output + tacit punishment
│   ├── stackelberg.py       ← Stackelberg quantity leadership
│   ├── market_power.py      ← Lerner index, HHI, market shares
│   ├── coalition.py         ← Shapley values, characteristic function, Folk theorem
│   ├── stochastic.py        ← AR(1) + jump-diffusion Monte Carlo
│   ├── rl_agent.py          ← Q-learning agent (OPEC)
│   ├── rl_multiagent.py     ← Independent multi-agent Q-learning
│   ├── evolutionary.py      ← Evolutionary game dynamics (replicator ODE)
│   ├── bertrand.py          ← Differentiated Bertrand price competition
│   ├── capacity_analysis.py ← Capacity-constrained equilibria + sweeps
│   ├── welfare.py           ← Consumer/producer surplus, DWL, carbon tax
│   ├── n_player.py          ← N-player Cournot sensitivity (n = 2..6)
│   ├── correlated_eq.py     ← Correlated equilibrium via linear programming
│   ├── empirical.py         ← Validation against historical price wars
│   ├── simulations.py       ← Orchestrators for each mode
│   ├── plotting.py          ← All matplotlib figures
│   ├── report.py            ← Auto-generates outputs/report.md
│   └── oil_game.py          ← OilGameTheory class API
└── outputs/                 ← Generated figures, CSVs, report.md
```

---

## Model summary

| Framework | Key result |
|---|---|
| Static Cournot | Nash: P=60 $/bbl, Q=80 mbd |
| Stackelberg (OPEC leads) | OPEC gains first-mover advantage |
| Repeated game | Cooperation sustainable iff δ ≥ δ* |
| Coalition / Shapley | OPEC has highest marginal contribution |
| Folk theorem | δ*=0.XX < 0.95 → cooperation rational |
| Stochastic (AR1 + jumps) | ±30 $/bbl demand shock → ±10 $/bbl price |
| Q-learning | OPEC converges between Nash and cooperative |
| Multi-agent RL (Green-Porter) | Independent learners self-organise punishment regimes |
| Evolutionary dynamics | Defection is ESS in 2-strategy game; Punish strategy stabilises cooperation in 3-strategy game |
| Bertrand (price competition) | OPEC dominance and collusion premium survive; collapses to MC pricing as σ → 1 |
| Capacity constraints | Binding caps shrink Nash output and raise δ\* (cooperation easier when constrained) |
| Welfare / DWL | Cartel triples DWL vs Nash; carbon tax shrinks the collusion premium |
| N-player sensitivity | Adding entrants dilutes OPEC's profit and Shapley value monotonically |
| Correlated equilibrium | A welfare-oriented mediator improves on Nash without binding agreements |
| Empirical validation | Model captures direction & mechanism of 1985, 2014, 2020 price wars (depth: under-predicted) |

---

## CLI options

| Flag | Description | Default |
|---|---|---|
| `--mode` | Simulation mode (see table below) | `all` |
| `--T` | Periods for repeated/stochastic | 50 |
| `--delta` | Discount factor δ | 0.95 |
| `--punishment_length` | Grim-trigger punishment periods | 10 |
| `--capacities` | Enable capacity constraints | `off` |
| `--seed` | Random seed | 123 |
| `--stochastic_sigma` | AR(1) shock std dev ($/bbl) | 8.0 |
| `--stochastic_rho` | AR(1) autocorrelation | 0.6 |
| `--stochastic_paths` | Monte Carlo paths | 300 |
| `--save_outputs` | Save outputs to disk | `on` |

### Available `--mode` values

| Mode | What it does |
|---|---|
| `static` | Duopoly vs triopoly Cournot Nash |
| `stackelberg` | Quantity leadership (each player as leader) |
| `market_power` | Lerner index, HHI, market shares |
| `repeated` | Myopic best-response convergence + λ inertia sweep |
| `cooperation` | Cartel quotas, tacit punishment, Folk-theorem δ\* |
| `coalition` | Shapley values + characteristic function |
| `stochastic` | AR(1) + jump-diffusion Monte Carlo, Green-Porter |
| `rl` | Single-agent Q-learning (OPEC vs static rivals) |
| `multiagent_rl` | Independent multi-agent Q-learning (Green-Porter info) |
| `evolutionary` | Replicator dynamics + ESS analysis |
| `bertrand` | **NEW.** Differentiated Bertrand price competition + σ sweep |
| `capacity` | **NEW.** Constrained vs unconstrained + OPEC capacity sweep |
| `welfare` | **NEW.** CS/PS/DWL across structures + carbon-tax interaction |
| `n_player` | **NEW.** Sweep n from 2 to 6 producers (entry / fragmentation) |
| `correlated` | **NEW.** Correlated equilibria (max-welfare / max-joint / max-min) |
| `empirical` | **NEW.** Validation against 1985, 2014, 2020 price wars |
| `all` | Run every mode end-to-end |

---

## Calibration

All parameters are empirically grounded:

| Parameter | Value | Source |
|---|---|---|
| a = 140 $/bbl | Choke price | IEA WEO 2019; Brent peak 2022 |
| b = 1.0 | Demand slope | Normalised; η ≈ −0.75 at baseline |
| c_US = 45 $/bbl | US shale all-in breakeven | Dallas Fed Survey (2020), EIA AEO 2023 |
| c_OPEC = 20 $/bbl | Gulf lifting + overhead | Saudi Aramco IPO (2019), BP Stat. Review 2022 |
| c_RUS = 35 $/bbl | Siberia lifting + transport | IMF Russia Art. IV (2021), Rystad Energy |
| δ = 0.95 | Discount factor | Griffin (1985), Gülen (1996) |

Run `python -c "from src.calibration import validate_calibration, empirically_grounded_params; d,c=empirically_grounded_params(); print(validate_calibration(d,c))"` to check all calibration invariants.

---

## Key outputs

| File | Description |
|---|---|
| `outputs/report.md` | Full auto-generated thesis report |
| `outputs/static_equilibrium.csv` | Duopoly and triopoly Nash equilibria |
| `outputs/repeated_nash_convergence.png` | Myopic BR convergence vs Nash benchmarks |
| `outputs/lambda_sensitivity.png` | Inertia λ validation sweep |
| `outputs/stochastic_comparison.png` | AR(1) vs jump-diffusion price paths |
| `outputs/rl_hyperparameter_sweep.png` | α/γ/ε sensitivity analysis |
| `outputs/evo_phase_2strategy.png` | 1-D phase portrait: C vs D |
| `outputs/evo_phase_3strategy.png` | Triangular simplex: C/D/P |
| `outputs/shapley_values.png` | Shapley value allocation |
| `outputs/folk_theorem.png` | δ* critical discount factors |
| `outputs/bertrand_sigma_sweep.png` | Bertrand-Nash response to substitutability σ |
| `outputs/capacity_opec_sweep.png` | Sensitivity to OPEC's capacity cap |
| `outputs/welfare_decomposition.png` | CS + PS + DWL by market structure |
| `outputs/welfare_carbon_tax.png` | Carbon tax × collusion premium |
| `outputs/n_player_opec_power.png` | OPEC's profit/Shapley/share vs n |
| `outputs/correlated_eq_comparison.png` | Nash vs CE vs cartel (3 objectives) |
| `outputs/empirical_price_wars.png` | Model vs history (1985 / 2014 / 2020) |

---

## References

- Cournot, A. (1838). *Recherches sur les principes mathématiques de la théorie des richesses.*
- Weibull, J.W. (1995). *Evolutionary Game Theory.* MIT Press.
- Friedman, D. (1991). Evolutionary games in economics. *Econometrica*, 59(3).
- Griffin, J.M. (1985). OPEC behavior: A test of alternative hypotheses. *AER*, 75(5).
- Gülen, S.G. (1996). Is OPEC a cartel? Evidence from smooth transition models. *Energy Journal*.
- Muthoo, A. (1999). *Bargaining Theory with Applications.* Cambridge University Press.
- IEA (2019). *World Energy Outlook.* International Energy Agency.
- BP (2022). *Statistical Review of World Energy.*
