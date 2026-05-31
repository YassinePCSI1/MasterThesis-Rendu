## 1. Model Overview & Calibration

This study models the global crude oil market as a **strategic oligopoly** among
three major production blocs: the **United States (US shale)**, the **OPEC cartel**,
and **Russia (RUS)**. Rather than applying a single game-theoretic model, we
progressively construct a multi-layered analysis — each framework addresses a
limitation of the previous one:

| Step | Framework | Builds on | New question addressed |
|---|---|---|---|
| 1 | **Static Cournot** | — | What is the one-shot Nash equilibrium? |
| 2 | **Market Power + Stackelberg** | Step 1 | Does OPEC have structural or first-mover power? |
| 3 | **Repeated Games** | Step 2 | Do dynamics change the equilibrium? |
| 4 | **Cartel + Folk Theorem** | Step 3 | Can cooperation survive deviation and punishment? |
| 5 | **Coalitions + Shapley** | Step 4 | How should cartel gains be shared fairly? |
| 6 | **Stochastic Demand** | Steps 1–5 | Are results robust to real-world uncertainty? |
| 7 | **Reinforcement Learning** | Step 6 | What if OPEC learns without knowing the model? |
| 8 | **Evolutionary Dynamics** | Step 7 | Which strategies survive natural selection? |

### 1.1 Calibration

The model uses a linear inverse demand function **P(Q) = a − bQ** with parameters
chosen to reflect stylised oil-market ranges:

| Parameter | Value | Interpretation |
|---|---|---|
| a (demand intercept) | 140 $/bbl | Brent choke price; consistent with IEA (2019) high-end scenarios |
| b (demand slope) | 1.0 | Normalised slope; implied elasticity ≈ −0.75 at baseline (medium-run; EIA SR: −0.06 to −0.10) |
| c_US (US marginal cost) | 45 $/bbl | Shale break-even; Dallas Fed Pulse survey average (2020) |
| c_OPEC (OPEC marginal cost) | 20 $/bbl | Gulf weighted average lifting + fiscal overhead (BP Stat. Review 2022) |
| c_RUS (Russia marginal cost) | 35 $/bbl | Siberian lifting + Urals export transport (IMF Russia Art. IV, 2021) |
| δ (discount factor) | 0.95 | Quarterly discount; captures geopolitical impatience beyond pure time preference |

### 1.2 Reading Guide

Each section ends with a **transition paragraph** that motivates the next step.
The logic is: *establish a benchmark → show its limits → add a new layer → repeat*.
By Section 10 all frameworks converge toward a unified set of conclusions.

---

## 2. The Benchmark: Static Cournot Equilibrium

We begin with the simplest possible model: a **one-shot Cournot game**
where each producer simultaneously chooses an output level to maximise
its own profit, taking rivals' outputs as given. This provides the
**competitive benchmark** against which all other structures are measured.

### 2.1 Duopoly (US vs OPEC) vs Triopoly (US, OPEC, Russia)

| Metric | Duopoly | Triopoly | Change |
|---|---|---|---|
| Total output Q (mbd) | 71.67 | 80.00 | +8.33 |
| Market price P ($/bbl) | 68.33 | 60.00 | -8.33 |
| q_US (mbd) | 23.33 | 15.00 | -8.33 |
| q_OPEC (mbd) | 48.33 | 40.00 | -8.33 |
| q_RUS (mbd) | — | 25.00 | — |
| Profit US | 544.44 | 225.00 | -319.44 |
| Profit OPEC | 2,336.11 | 1,600.00 | -736.11 |
| Profit RUS | — | 625.00 | — |
| Consumer surplus | 2,568.06 | 3,200.00 | +631.94 |
| Total welfare | 5,448.61 | 5,650.00 | +201.39 |

**Key findings:**

- Russia's entry **increases total supply by 8.33 mbd** and **lowers price by 8.33 $/bbl**.
- OPEC's profit falls by **736.11** — Russia's entry directly erodes OPEC's market power.
- Consumer surplus rises by **631.94**, confirming the pro-competitive effect.
- Total welfare is **higher** in the triopoly: more competition raises efficiency.

![Duopoly vs Triopoly](duopoly_vs_triopoly.png)
![Quantity Comparison](quantity_comparison.png)
![Profit Comparison](profit_comparison.png)

### 2.2 Comparative Statics: demand sensitivity

Varying the demand intercept *a* from 112.00 to 168.00 (±20% of baseline):

- Price ranges from **53.00** to **67.00 $/bbl**.
- Total output ranges from **59.00** to **101.00 mbd**.
- Every 1-unit increase in demand intercept raises equilibrium price by 1/3
  and output by 1/3 (standard triopoly result).

![Comparative Statics](comparative_statics.png)

> **Transition →** The static Nash gives us a benchmark (P=60, Q=80), but it treats
> all producers as symmetric decision-makers. In reality, OPEC has a **structural cost
> advantage** (c=20 vs 35–45) and a **first-mover advantage** (it announces quotas
> before others respond). The next section quantifies these asymmetries.

---

## 3. Static Market Structure: Market Power & First-Mover Advantage

### 3.1 Market Power Analysis

Market power is quantified via the **Lerner Index** L_i = (P − c_i)/P and the
**Herfindahl-Hirschman Index** HHI = Σ s_i² × 10,000.

| Market structure | Price | HHI | Lerner OPEC | Lerner US | Lerner RUS |
|---|---|---|---|---|---|
| Duopoly (US+OPEC) | 68.33 | 5,608 | 0.71 | 0.34 | nan |
| Cournot Nash (triopoly) | 60.00 | 3,828 | 0.67 | 0.25 | 0.42 |
| Stackelberg (US leads) | 55.00 | 3,495 | 0.64 | 0.18 | 0.36 |
| Stackelberg (OPEC leads) | 46.67 | 7,506 | 0.57 | 0.04 | 0.25 |
| Stackelberg (RUS leads) | 51.67 | 4,546 | 0.61 | 0.13 | 0.32 |

**Interpretation:**

- OPEC consistently achieves the **highest Lerner index** across all structures,
  reflecting its lowest marginal cost (20 $/bbl).
- HHI > 2,500 in all scenarios confirms the market is **highly concentrated**.

![Market Power Analysis](market_power.png)

### 3.2 Stackelberg Leadership

OPEC historically announces production targets *before* other producers respond —
this is the Stackelberg leader role. How much does this first-mover advantage
add to the structural cost advantage identified above?

| Scenario | Price ($/bbl) | Q (mbd) | q_OPEC | q_US | q_RUS | OPEC profit | Leader advantage |
|---|---|---|---|---|---|---|---|
| Nash (no leader) | 60.00 | 80.00 | 40.00 | 15.00 | 25.00 | 1,600.00 | 0.00 |
| US | 55.00 | 85.00 | 35.00 | 30.00 | 20.00 | 1,225.00 | 75.00 |
| OPEC | 46.67 | 93.33 | 80.00 | 1.67 | 11.67 | 2,133.33 | 533.33 |
| RUS | 51.67 | 88.33 | 31.67 | 6.67 | 50.00 | 1,002.78 | 208.33 |

**Key findings:**

- When OPEC leads, the first-mover advantage is **533.33** profit units.
- OPEC's market power is therefore **both structural (cost) and strategic (timing)**.
- A US-led Stackelberg (shale revolution) compresses OPEC's profits significantly.

![Stackelberg Comparison](stackelberg_comparison.png)

> **Transition →** So far, all results are static (one-shot). But the oil market is a
> **quarterly repeated game**: producers observe each other's output and adjust over
> time. Do the dynamics converge to the same Nash equilibrium, or does repetition
> change the outcome?

---

## 4. From Static to Dynamic: Repeated Game Dynamics

We now model production as a **T=50 period repeated game** with inertia (λ=0.2):
each period, producers move 20% toward their best response to rivals' previous output.

### 4.1 Myopic Best-Response Convergence

| Metric | Period 0 | Period 49 | Δ |
|---|---|---|---|
| Price ($/bbl) | 108.00 | 60.00 | -48.00 |
| Total output Q | 32.00 | 80.00 | 48.00 |
| q_OPEC | 12.00 | 39.93 | 27.93 |
| q_US | 9.50 | 15.06 | 5.56 |
| q_RUS | 10.50 | 25.01 | 14.51 |

**Result:** The repeated game **converges to the static Nash equilibrium** (P≈60, Q≈80)
within ~8 periods. The one-shot benchmark from Section 2 is therefore the
long-run attractor of the dynamic system. Inertia slows convergence but does
not alter the destination.

- **Price volatility** (std dev): **8.14 $/bbl** — **Quantity volatility**: **8.14 mbd**

### 4.2 Inertia (λ) Sensitivity Analysis

λ=0 means no adjustment (output stays unchanged); λ=1 means instant best response.
The baseline λ=0.2 is validated by sweeping the full range.

**Mathematical structure.** For n=3 players with linear demand, the aggregate
output evolves as Q(t+1) = Q(t)·μ + const, where the eigenvalue is
**μ = 1 − 2λ**. Two qualitative regimes emerge:

| Regime | λ range | μ | Behaviour |
| --- | --- | --- | --- |
| Monotone | 0 < λ < 0.5 | 0 < μ < 1 | Price descends smoothly toward Nash |
| Critical | λ* = 0.5 | μ = 0 | **Instant convergence** (Q = Q_Nash from t=1) |
| Oscillatory | 0.5 < λ < 1 | −1 < μ < 0 | Price overshoots and zigzags around Nash |
| Divergent | λ = 1 | μ = −1 | Permanent oscillation (cobweb instability) |

The critical value λ* = 2/(n+1) = 0.5 is specific to n=3 players;
convergence speed depends on |μ| = |1 − 2λ|, which is **symmetric around λ=0.5**.
This explains the U-shaped convergence-period curve.

- **λ=0.2 (baseline)**: |μ|=0.6, half-life ≈ 1.4 periods, convergence in ~9 periods.
  This matches OPEC's quarterly adjustment cycle (2–3 quarters in practice).
- All λ ∈ (0, 1) converge to the same Nash price — speed changes, destination does not.

![Repeated Time Series](repeated_time_series.png)
![Nash Convergence](repeated_nash_convergence.png)
![Repeated Price & Quantity](repeated_price_quantity.png)
![Repeated Profits](repeated_profit_time_series.png)
![Lambda Sensitivity](lambda_sensitivity.png)

> **Transition →** The myopic dynamics confirm that Nash is the attractor. But OPEC
> earns only **1,600/period** at Nash, while a cartel restricting output could earn
> **1,800/period** (Section 2 already hinted that joint-profit maximisation raises
> the price to 80 $/bbl). Can cooperation be sustained despite each player's
> short-term incentive to overproduce?

---

## 5. Can Cooperation Be Sustained? Cartel, Punishment & Folk Theorem

This is the central question of the thesis. We tackle it in three stages:

1. **Define cooperation**: proportional cartel quotas (OPEC-style pro-rata cuts from Nash).
2. **Simulate deviation + punishment**: one player deviates, then all revert to Nash.
3. **Formalise sustainability**: the Folk Theorem gives the exact patience threshold δ*.

### 5.1 Cooperative Equilibrium vs Nash

Under proportional quotas, each player cuts production by the same percentage
so that total output equals the joint-profit-maximising quantity. The price
rises from 60 to 80 $/bbl, and **every player earns more than at Nash** —
guaranteeing individual rationality without side payments.

### 5.2 Tacit Punishment Simulation

In the simulation, all players maintain cartel quotas until **OPEC deviates at t=5**
(best-responding to the others' cooperative outputs). A **punishment phase** (10 periods
of Nash reversion) follows, after which cooperation resumes.

| Scenario | OPEC discounted profit (δ=0.95) |
|---|---|
| Myopic Nash convergence | 26,855.55 |
| Cooperative (with deviation + punishment) | 32,224.22 |

**Even after deviating and being punished**, OPEC still earns **5,368.67
more** under the cooperative regime than under pure Nash play. The cartel is
unambiguously profitable.

The graph below shows the three regimes clearly: **green** (cooperation) → **red**
(punishment after OPEC's deviation at t=5) → **blue** (recovery/return to cooperation).

![Punishment Regimes](repeated_punishment_regimes.png)

### 5.3 Folk Theorem: Formal Sustainability Condition

The Folk Theorem formalises the above: cooperation is an equilibrium of the
repeated game if and only if the discount factor δ exceeds a critical threshold δ*.

**IC constraint (grim trigger):** δ ≥ δ* = (π_dev − π_coop) / (π_dev − π_nash)

| Player | δ* (critical) | π_deviation | π_cooperative | π_nash |
|---|---|---|---|---|
| US | **0.455** | 534.77 | 393.75 | 225.00 |
| OPEC | **0.529** | 2,025.00 | 1,800.00 | 1,600.00 |
| RUS | **0.441** | 1,016.02 | 843.75 | 625.00 |

**Binding constraint: δ* = 0.529** (OPEC has the highest temptation).

With δ = 0.95 ≥ δ* = 0.529,
**cooperation is sustainable ✓**.

**Interpretation:** OPEC is the binding player because it has the largest gain
from deviating (π_dev − π_coop) relative to the punishment cost (π_dev − π_nash).
A δ* of ~0.53 means the cartel is viable as long as members plan at least ~2 quarters
ahead — a low bar, confirming that OPEC-style cooperation is theoretically robust.

![Folk Theorem](folk_theorem.png)

> **Transition →** The Folk Theorem tells us *that* cooperation is sustainable.
> But it does not say *how to split the gains*. If OPEC, US, and RUS form a
> grand coalition (OPEC+), which allocation is fair and stable?

---

## 6. Fair Sharing: Coalition Formation & Shapley Values

Shapley values provide the **unique fair allocation** satisfying the axioms
of efficiency, symmetry, null-player, and additivity. Each player's value
equals its average marginal contribution across all coalition orderings.

| Player | Shapley value | Interpretation |
|---|---|---|
| US | 477.95 | Meaningful contributor — competitive pressure |
| OPEC | 2,202.43 | Largest contributor — low cost + dominant share |
| RUS | 919.62 | Moderate contributor — swing producer role |
| **Grand coalition** | **3,600.00** | Total joint profit v(N) |

**Core stability: YES ✓**

The grand coalition is **in the core**: no subset can profitably break away.
This provides a theoretical foundation for **OPEC+ (OPEC-Russia) coordination**.

### Characteristic function v(S)

| Coalition | v(S) |
|---|---|
| empty | 0.00 |
| US | 300.00 |
| OPEC | 2,133.33 |
| OPEC+US | 2,278.12 |
| RUS | 833.33 |
| RUS+US | 1,012.50 |
| OPEC+RUS | 2,628.12 |
| OPEC+RUS+US | 3,600.00 |

![Shapley Values](shapley_values.png)

> **Transition →** Sections 2–6 assumed **deterministic demand**. Real oil markets
> face persistent demand shocks (business cycles, energy transition), supply disruptions
> (geopolitics, pandemics), and Poisson jumps. Do our equilibria survive uncertainty?

---

## 7. Robustness to Uncertainty: Stochastic Demand

Sections 2–6 assumed **deterministic demand**.  Real oil markets face
persistent demand shocks (business cycles, energy transition), supply
disruptions (geopolitics, pandemics), and Poisson jumps.  This section
asks: *do our equilibria survive uncertainty?*

### 7.1 Benchmark: Static Cournot Under Demand Shocks

As a first pass, players solve the **static Nash** each period under
a shocked demand intercept P(Q,t) = (a + ε_t) − bQ, where
ε_t = ρ·ε_{t-1} + η_t (AR-1, σ=8, ρ=0.6, 300 MC paths).

| Statistic | Value |
|---|---|
| Mean terminal price | 59.77 $/bbl |
| 10th percentile | 56.62 $/bbl |
| 90th percentile | 62.89 $/bbl |
| P90−P10 spread | 6.27 $/bbl |

The **6.27 $/bbl** spread shows that demand uncertainty
alone generates substantial price volatility.  But this benchmark is
**incomplete**: it ignores the repeated-game structure entirely.  If the
Folk Theorem tells us cooperation is sustainable (Section 5), what
happens to the cartel when prices drop — is it a deviation or bad luck?

### 7.2 Demand Shock Scenarios (Comparative Statics)

| Shock Δa | Price before | Price after | Change (%) | OPEC profit after |
|---|---|---|---|---|
| Δa=-30 | 60.00 | 52.50 | -12.50% | 1,056.25 |
| Δa=-20 | 60.00 | 55.00 | -8.33% | 1,225.00 |
| Δa=-10 | 60.00 | 57.50 | -4.17% | 1,406.25 |
| Δa=+0 | 60.00 | 60.00 | 0.00% | 1,600.00 |
| Δa=+10 | 60.00 | 62.50 | 4.17% | 1,806.25 |
| Δa=+20 | 60.00 | 65.00 | 8.33% | 2,025.00 |
| Δa=+30 | 60.00 | 67.50 | 12.50% | 2,256.25 |

### 7.3 The Monitoring Problem: Green-Porter (1984)

The deterministic Folk Theorem (Section 5) assumes players can **observe**
whether a rival deviated.  Under stochastic demand, a low price could mean
either cheating or bad luck — this is **imperfect public monitoring**.

The **Green-Porter trigger-price strategy** resolves this:
- **Cooperative phase**: all produce at cartel quotas (Section 5.1).
- **Trigger rule**: if the observed price falls below p̄ (set near the
  Nash price), *all* players switch to Nash punishment for L periods —
  regardless of whether anyone actually deviated.
- **Recovery**: after L periods, cooperation resumes.

This means punishment is sometimes triggered by **demand shocks alone**
(false positives).  The cost of uncertainty is periodic price wars
even when everyone cooperates.

**Simulation results (AR-1 specification):**

| Metric | Value |
|---|---|
| Trigger price p̄ | 68.00 $/bbl |
| Steady-state cooperation fraction | 94% |
| Mean cooperation spell | 61.79 periods |
| Mean punishment spell | 9.62 periods |
| Total punishment triggers | 291 |

**With jump-diffusion shocks**, the cooperation fraction drops to
**85%** (vs 94% under AR-1), reflecting the higher probability of large negative shocks
triggering false punishment phases.

**Connection to the Punish ESS (Section 9):** the evolutionary analysis
showed that Punish is the surviving strategy.  Green-Porter explains *why*:
under uncertainty, the only credible enforcement mechanism is based on
**observables** (market price), not on detecting individual deviations.
The Punish strategy IS the Green-Porter trigger in population form.

![Green-Porter Regimes](green_porter_regimes.png)
![Cooperation Survival](green_porter_coop_survival.png)

### 7.4 Stochastic Folk Theorem: δ* Under Uncertainty

The Green-Porter IC constraint accounts for false triggers (probability α
per cooperative period).  The stochastic δ* is higher than the deterministic
one because players must be more patient to tolerate periodic price wars.

| | Deterministic | Stochastic (AR-1) |
|---|---|---|
| δ* (binding) | 0.53 | 0.60 |
| False-trigger α | 0 | 11.5% |
| Sustainable (δ=0.95)? | Yes | Yes |

Cooperation remains sustainable (δ*=0.60 < δ=0.95)
but the **price of uncertainty** is a higher patience requirement and periodic
punishment phases even when no one deviates.

![Delta Star Comparison](delta_star_comparison.png)

### 7.5 Convergence After Shocks

After a large negative shock, the Green-Porter mechanism triggers
punishment, then recovers to cooperation.  The convergence path depends
on the same inertia eigenvalue |1−2λ| from Section 4.2.

![Convergence After Shock](convergence_after_shock.png)

### 7.6 Jump-Diffusion Extension (Compound Poisson Process)

The AR(1) specification captures demand-side persistence well but misses the
sudden **supply-side disruptions** that characterise real oil markets.
We extend the model with a **compound Poisson process** (Merton, 1976):

  ε_t = ρ·ε_{t-1} + η_t + J_t

where the jump component J_t is:
- **Arrival**: N_t ~ Poisson(λ_J), λ_J=0.08 ≈ 1 shock/year.
- **Size**: each jump ΔJ ~ N(μ_J, σ_J²), σ_J=18 $/bbl.
- **Persistence**: jumps enter the AR(1) ε_t, decaying at rate ρ=0.6.

![Stochastic Price Bands (AR1)](stochastic_price_bands.png)
![Stochastic Comparison: AR1 vs Jump-Diffusion](stochastic_comparison.png)
![Demand Shock Scenarios](demand_shock_scenarios.png)

> **Transition →** All models so far assume OPEC **knows** the demand function and
> rivals' cost structure. What if OPEC must **learn** the optimal strategy through
> trial and error, with no model of the environment?

---

## 8. Learning Without a Model: Reinforcement Learning

A **Q-learning agent** is trained as OPEC over 300 episodes × 50 steps,
while US and RUS play myopic best responses.

| Benchmark | q_US | q_OPEC | q_RUS | Q | P |
|---|---|---|---|---|---|
| nash | 15.00 | 40.00 | 25.00 | 80.00 | 60.00 |
| cooperative | 11.25 | 30.00 | 18.75 | 60.00 | 80.00 |
| rl_avg | 18.00 | 32.08 | 27.46 | 77.54 | 62.46 |

**Key finding:** The RL agent converges to **P=62.5 $/bbl** — between
Nash (60.00) and cartel (80.00). Without any explicit
communication or knowledge of the demand model, the agent **discovers partial
collusion** through trial-and-error: it learns that restraining output raises
revenue. This is consistent with the algorithmic collusion literature
(Calvano et al., 2020).

### 8.2 Hyperparameter Sensitivity

| Parameter | Baseline | Rationale |
|---|---|---|
| α (learning rate) | 0.15 | Optimal in 0.10–0.20 range; balances speed and stability |
| γ (discount factor) | 0.95 | Matches economic δ; agent has the same patience as the game |
| ε (exploration) | 0.10 | Moderate exploration outperforms pure exploitation |

![RL Learning Curve](rl_learning_curve.png)
![RL Hyperparameter Sweep](rl_hyperparameter_sweep.png)
![RL Action Distribution](rl_action_distribution.png)
![RL Rolling Outputs](rl_rolling_outputs.png)

> **Transition →** The RL agent shows that *individual* learning converges to partial
> collusion. But is this outcome stable at the *population* level? If many producers
> imitate successful strategies, which strategy survives evolutionary selection?

---

## 9. Population Dynamics: Evolutionary Game Theory

Evolutionary game theory models a population of producers choosing among strategies,
with more profitable strategies spreading through imitation (**replicator dynamics**).

### 9.1 Two-Strategy Game (C vs D)

| | vs Cooperate | vs Defect |
|---|---|---|
| **Cooperate** | 1,800.00 (π_coop) | 1,500.00 (π_sucker) |
| **Defect**    | 2,025.00 (π_dev)  | 1,600.00 (π_nash)   |

This is a **Prisoner's Dilemma**: π_dev > π_coop > π_nash > π_sucker.
Without punishment, defection dominates — cooperation unravels.

### ESS Results

- x=1 (All Cooperate) — unstable ✗
- x=0 (All Defect) — ESS ✓

### 9.2 Three-Strategy Game (C, D, P)

Adding a **Punish** strategy (Saudi-style production surges to discipline cheaters)
changes the dynamics fundamentally. Punishers deter defectors, which can
stabilise cooperation — closing the loop with the Folk Theorem in Section 5.

![Evolutionary Payoff Matrices](evo_payoff_matrix.png)
![Evolutionary Phase Diagram (2-Strategy)](evo_phase_2strategy.png)
![Evolutionary Phase Diagram (3-Strategy: C/D/P)](evo_phase_3strategy.png)

> **Transition →** The evolutionary analysis confirms the key insight: **punishment
> is the mechanism that sustains cooperation**. This is consistent across the
> repeated game (Section 4–5), the Folk Theorem (Section 5.3), and the replicator
> dynamics (this section). We now synthesise all seven frameworks.

---

## 10. Cross-Model Synthesis & Conclusions

All seven frameworks converge toward a coherent picture. The table below ranks
market structures from most to least collusive:

| Market structure | Price ($/bbl) | OPEC output | OPEC profit | Consumer welfare |
|---|---|---|---|---|
| Cartel quotas (cooperation) | 80.00 | 30.00 mbd | Highest | Lowest |
| Stackelberg (OPEC leads) | 46.67 | Highest | High | Low |
| RL learned policy | 62.46 | 32.08 mbd | Medium-high | Medium |
| Cournot Nash triopoly | 60.00 | 40.00 mbd | Medium | Medium |
| Perfect competition | ≈c_avg | Max | Zero | Maximum |

### Key findings

1. **Nash is the static attractor** (Section 2, 4): myopic dynamics always converge to it.

2. **OPEC's power is structural + strategic** (Section 3): lowest cost + first-mover = dominant position.

3. **Cooperation beats Nash for everyone** (Section 5): cartel quotas raise OPEC's
   per-period profit from 1,600 to 1,800 (+12.5%).

4. **Cooperation is enforceable** (Section 5.3): δ*=0.53 ≪ δ=0.95, so the cartel
   survives even with considerable impatience.

5. **Punishment is the key mechanism** (Sections 5, 9): it sustains cooperation in
   the repeated game (Folk Theorem) and in evolutionary populations (Punish strategy).

6. **Demand uncertainty raises δ* but doesn't break cooperation** (Section 7):
   the Green-Porter model shows that stochastic shocks trigger periodic
   price wars (false positives), but cooperation survives because δ* remains
   below δ=0.95.  This explains *why* the Punish ESS emerges (Section 9):
   trigger-price punishment is the only credible mechanism under imperfect monitoring.

7. **Learning discovers partial collusion** (Section 8): even without a model,
   OPEC converges between Nash and cartel — algorithmic collusion is a real possibility.

---

## 11. Policy Implications

### For OPEC member states
- Maintaining **quantity leadership** (Stackelberg) is strategically optimal.
- The Folk theorem result (δ* ≈ 0.53) means cartel discipline is enforceable
  whenever members plan more than ~2 quarters ahead.
- Demand shocks disproportionately hurt high-cost producers; OPEC's structural
  hedge (low c) makes it the most resilient bloc.

### For Russia (OPEC+)
- Russia's Shapley value justifies its inclusion: it brings measurable marginal value.
- As Stackelberg follower, its optimal play is to best-respond to OPEC's quota.

### For competing producers (US shale)
- US shale (c=45 $/bbl) is the **most exposed** to price wars and demand collapses.
- Cost reduction toward Gulf levels would shift the Stackelberg leadership equilibrium.

### For regulators
- HHI > 2,500 in all structures confirms **structural high concentration**.
- The welfare gap between Nash and cartel (**1,400.00** consumer surplus units)
  quantifies the social cost of OPEC collusion — a lower-bound for the value of
  antitrust enforcement or pro-competitive policy.

---
