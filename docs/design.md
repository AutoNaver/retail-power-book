# Design notes

These notes say how the pieces fit together, so contributors (human or agent) don't make modelling decisions in isolation. `AGENTS.md` has the coding conventions and domain rules. This document doesn't repeat them.

Anything under **Open questions** is undecided. Don't settle an open question inside a feature PR. Raise it in the issue.

## 1. The analogy

A retail supplier sells power to households at a fixed price, which is the same position a bank holds with non-maturing deposits.

| Non-maturing deposits | Fixed-price retail power book |
|---|---|
| Deposit balance, uncertain over time | Delivered volume in MWh, uncertain per hour and per year |
| Administered client rate, sticky | Fixed tariff price in EUR/MWh |
| Market rate | DE-LU day-ahead price in EUR/MWh |
| Replicating portfolio of fixed-rate tranches | Strip of baseload and peakload forwards |
| Objective: stabilise the net interest margin | Objective: stabilise the gross margin |
| Core balance vs volatile balance | Volume a forward strip can hedge vs residual shape and volume |

The book's gross margin over a delivery period is

```
margin = Σ_h  L_h · (p_tariff − S_h)  +  Σ_k  q_k · Σ_{h ∈ H_k} (S_h − F_k)
         └──── retail book ─────┘       └────────── hedge ──────────┘
```

where `h` indexes delivery hours, `L_h` is load in MWh, `S_h` is the day-ahead price in EUR/MWh, `p_tariff` is the fixed tariff price in EUR/MWh (energy component only), `q_k` is the hedge volume in MW of forward product `k`, `H_k` is its set of delivery hours, and `F_k` is its forward price in EUR/MWh. At hourly resolution, 1 MW held for one hour is 1 MWh.

The whole project is about estimating the distribution of `margin` and choosing `q` well.

## 2. Pipeline

```
data/  ──►  models/load.py   ──► load paths   L[s, h]  (MWh)      ─┐
       └►  models/price.py  ──► price paths  S[s, h]  (EUR/MWh)  ─┼─► hedging/replication.py ──► q_k (MW)
                                                                   │
            calendar / products ──► delivery hours H_k, F_k        ─┘
                                                                          │
                                       risk/metrics.py, risk/scenarios.py ◄┘  margin distribution
```

Each stage only communicates through the array and frame contracts in section 3. No stage reaches into another's internals.

## 3. Data contracts

**Time series** (loaders, fitted inputs): a `pandas.Series` or `DataFrame` with a timezone-aware UTC `DatetimeIndex` at hourly frequency. Put units in the column names: `price_eur_mwh`, `load_mwh`.

**Simulation output**: a `numpy` array of shape `(n_paths, n_hours)`, plus the shared UTC `DatetimeIndex` of length `n_hours` that labels its columns. Load and price paths for the same run share the index and the path dimension, so path `s` of load and path `s` of price belong to the same scenario. Correlation between load and price lives in the simulation, not downstream.

**Gaps**: loaders never forward-fill or interpolate silently. They return NaN for missing hours and expose which hours are missing. Any fill happens explicitly in model code and is documented there.

**Day-ahead resolution**: since delivery day 1 October 2025, the SDAC day-ahead auction clears in 15-minute products. This project works hourly. A loader that receives quarter-hourly prices aggregates each hour to the mean of its four quarter-hour prices, which is the price of a flat 1 MW hour. The loader does this in one place and tests it.

**Products**: a forward product is defined by its type (`base` or `peak`), its delivery period (month or quarter), and therefore a boolean mask over the hourly index. Hour counts always come from that mask, never from `24 × days`. Peak follows the domain rule in `AGENTS.md` (see open question Q1).

## 4. Load model (`models/load.py`)

The book is residential. Below 100,000 kWh per year, German household customers are settled against a standard load profile (SLP), not against metered hourly values.

Structure:

```
L[s, h] = N_customers · E_annual[s] · shape_h · (1 + ε[s, h])
```

- `N_customers`: fixed per run. Churn is out of scope.
- `E_annual[s]`: annual consumption per customer in MWh, stochastic across paths. This is the main volume risk for an SLP book.
- `shape_h`: a normalised profile that sums to 1 over the delivery year, built from the standard household profile including its dynamisation factor. The profile source must be citable and committed as a fixture.
- `ε[s, h]`: optional hourly deviation, zero in the baseline (see open question Q2).

Functions accept a `numpy.random.Generator`. A deterministic path (no noise) must reproduce `N_customers · E_annual · shape_h` exactly, and tests check that.

## 5. Price model (`models/price.py`)

The price is additive, never multiplicative in logs, because prices can be negative:

```
S[s, h] = level_m(h)[s] + shape(hour-of-day, day type, month) + residual[s, h]
```

- `level`: the monthly baseload level. In simulation it is anchored to a forward curve and moves stochastically between paths.
- `shape`: a deterministic hourly shape estimated from history, with zero mean per month, so it doesn't shift the level. Day types are at least weekday, Saturday and Sunday/holiday. Hours are local Europe/Berlin clock hours, computed from the UTC index.
- `residual`: stochastic, mean-zero, with daily persistence. It must be able to produce negative prices and price spikes. Don't truncate it.

Estimation reads historical SMARD or ENTSO-E prices through `data/`. Every estimator has a synthetic test: generate prices from known parameters with a seeded generator, then check that the estimator recovers them within tolerance.

## 6. Replication layer (`hedging/`), interface only

The maintainer writes `hedging/replication.py` by hand. Agents don't modify it. They build what feeds it and test what comes out.

Inputs:

- `load_mwh`: array `(n_paths, n_hours)`
- `price_eur_mwh`: array `(n_paths, n_hours)`
- `index`: UTC `DatetimeIndex`, length `n_hours`
- `products`: a list of product definitions (section 3), with delivery masks over `index`
- `forward_eur_mwh`: one forward price per product
- `tariff_eur_mwh`: scalar

Output: a hedge volume `q_k` in MW per product.

Reference objective (for tests and comparison): minimum variance of `margin` across paths. With `X[s, k] = Σ_{h∈H_k}(S[s,h] − F_k)` and the unhedged margin `M0[s]`, this is a least-squares problem on the centred columns, solved with `lstsq`, not by inverting `XᵀX`. Overlapping products (base and peak for the same month; a month inside a quarter) make `X` collinear, so the product set passed in must be non-redundant (see Q3).

**Forward prices in simulation**: by default `F_k` equals the simulated mean of the average spot price over `H_k` (hour-weighted), so the hedge has zero expected payoff and only reduces variance. Tests can use that as a known result.

## 7. Risk (`risk/`)

- `metrics.py` takes margin samples `(n_paths,)` and returns the mean, standard deviation, and the lower-tail value-at-risk and expected shortfall of margin at 95% and 99% (loss convention stated in the docstring). It also decomposes margin into price, volume and shape components against a reference plan.
- `scenarios.py` defines deterministic stresses as transformations of simulated paths: a price-level shock, an annual volume shock, and a "cold and expensive" joint shock where high volume coincides with high price. Scenarios are data in `configs/`, not constants in code.

**Shape risk** is the part a base/peak strip can't remove: the SLP profile differs from a flat or peak block inside every month. It is expected to remain after hedging and must be reported separately, not hidden in "residual".

## 8. Open questions

- **Q1 Peak and holidays.** Does the EEX Phelix-DE Peak product exclude German public holidays that fall on weekdays? Our current understanding is that it doesn't. Check the EEX contract specification before relying on this and cite the document in the calendar module.
- **Q2 Hourly volume noise.** For an SLP book, the supplier delivers the synthetic profile, and the deviation from metered consumption is settled after the fact as Mehr-/Mindermengen at a regulated price. Should `ε` stay zero, with volume risk only in `E_annual` and settlement as a separate cash flow? Or should we model a smart-metered book where `ε` is weather-driven?
- **Q3 Product set.** Hedge a cascade (months for the front quarter, quarters beyond) or solve with overlapping products under regularisation? This is the maintainer's decision.
- **Q4 Load–price dependence.** With `ε = 0`, load and price are linked only through `E_annual` and the price level. What correlation between them do we assume, and on what evidence?
- **Q5 Tariff.** Is `p_tariff` the energy component only (current assumption) or the full end-customer price minus grid fees, levies and taxes?
