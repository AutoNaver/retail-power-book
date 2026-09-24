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

`data/` holds the loaders: SMARD day-ahead prices and DWD station temperatures.

Each stage only communicates through the array and frame contracts in section 3. No stage reaches into another's internals.

## 3. Data contracts

**Time series** (loaders, fitted inputs): a `pandas.Series` or `DataFrame` with a timezone-aware UTC `DatetimeIndex` at hourly frequency. Put units in the column names: `price_eur_mwh`, `load_mwh`.

**Simulation output**: a `numpy` array of shape `(n_paths, n_hours)`, plus the shared UTC `DatetimeIndex` of length `n_hours` that labels its columns. Load and price paths for the same run share the index and the path dimension, so path `s` of load and path `s` of price belong to the same scenario. Correlation between load and price lives in the simulation, not downstream.

**Gaps**: loaders never forward-fill or interpolate silently. They return NaN for missing hours and expose which hours are missing. Any fill happens explicitly in model code and is documented there.

**Day-ahead resolution**: since delivery day 1 October 2025, the SDAC day-ahead auction clears in 15-minute products. This project works hourly. A loader that receives quarter-hourly prices aggregates each hour to the mean of its four quarter-hour prices, which is the price of a flat 1 MW hour. The loader does this in one place and tests it.

**Products**: a forward product is defined by its type (`base` or `peak`), its delivery period (month or quarter), and therefore a boolean mask over the hourly index. Hour counts always come from that mask, never from `24 × days`. Peak follows the domain rule in `AGENTS.md` (see open question Q1).

## 4. Load model (`models/load.py`)

The book is residential and modelled as **smart-metered**: the supplier buys each customer's actual hourly consumption at the day-ahead price, so hourly deviations from the profile are the supplier's risk. (With standard-load-profile settlement, the supplier would deliver the synthetic profile and settle the difference later at a regulated price; we don't model that case.)

Structure:

```
L[s, h] = N_customers · E_annual[s] · shape_h · (1 + ε[s, h])
```

- `N_customers`: fixed per run. Churn is out of scope.
- `E_annual[s]`: annual consumption per customer in MWh under normal weather, stochastic across paths.
- `shape_h`: the BDEW household profile H25, including its dynamisation, summed from quarter-hours to hours and normalised to sum to 1 over each local calendar year, so a delivery month gets its real share of `E_annual`. It comes from the `demandlib` package (MIT). BDEW publishes the profiles without a licence, so we don't commit copies of the data; tests use the profile from the installed package. Holidays passed to the profile are the nationwide ones from `rpb.holidays`. On DST days, `demandlib` is evaluated on the local quarter-hour clock index of the year: the spring day has 92 quarter-hours (no 02:00 hour), and on the autumn day both 02:00 hours carry the profile's 02:00 values. Each UTC hour gets the sum of the four quarter-hours that start in it, and the normalisation is applied after this mapping, so every UTC delivery hour has exactly one value.
- `ε[s, h]`: the hourly deviation from the profile caused by weather:

  ```
  ε[s, h] = g(ΔT[s, h]) + η[s, h]
  ΔT[s, h] = Σ_i w̃_i(h) · (T_i[s, h] − T_normal,i(h))
  ```

  `T_i` is the hourly temperature in °C at DWD station `i`, with the station list and weights in `configs/`. `ΔT` is a German temperature anomaly: each station's deviation from **its own** normal, averaged with weights `w̃_i`. When some stations are missing an hour, `w̃` are the reporting stations' weights rescaled to sum to 1, as long as they carry at least `weather.min_reporting_weight` of the total weight; otherwise the hour is NaN. Taking anomalies per station first means a missing station changes which stations are averaged, but doesn't create a false anomaly: a missing cold station would otherwise make the average look warm.

  `T_normal,i` is fitted per station and per hour of day in standard time (see below), by least squares on history: a level, a linear trend in years, and three annual harmonics. Because of the trend, a weather year's anomaly is measured against that year's own normal, so a historical weather year doesn't carry decades of warming into the delivery year. `g` is a temperature response with `g(0) = 0`, so normal weather reproduces the profile. `η` is a small mean-zero hourly residual.

**Weather scenarios**: each path gets a historical weather year. This keeps the real persistence of cold spells and heat waves, which a simple noise process wouldn't.

The mapping from delivery hours to weather hours uses **standard time, UTC+1 all year (MEZ, no DST)**, not local clock time. In standard time every day has exactly 24 hours, so there are no DST gaps or repeated hours, and it tracks the sun, which suits temperature. For each delivery hour:

1. Convert its UTC start to UTC+1 and take the calendar month, day and hour.
2. Take the weather year's observation at the same month, day and hour in UTC+1.
3. Leap days: a delivery 29 February uses the weather year's 28 February when the weather year has no 29 February. A weather year's 29 February is unused when the delivery year has none.
4. Multi-year deliveries: the first delivery year (in UTC+1) maps to the chosen weather year, and each following year to the next weather year, so consecutive years stay consecutive.

Every delivery hour gets exactly one weather hour, with nothing dropped, duplicated or filled. Weekdays shift between the two years; that's fine for temperature, because day-type effects live in `shape_h`. Missing weather observations stay NaN, and the load model decides how to handle them.

Functions accept a `numpy.random.Generator`. A deterministic path (normal weather, no residual) must reproduce `N_customers · E_annual · shape_h` exactly, and tests check that.

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
- **Q2 Hourly volume** (decided 2026-09-24). The book is smart-metered: `ε` is non-zero, driven by temperature from DWD, and settled at the day-ahead price. See section 4.
- **Q3 Product set.** Hedge a cascade (months for the front quarter, quarters beyond) or solve with overlapping products under regularisation? This is the maintainer's decision.
- **Q4 Load–price dependence.** Load depends on temperature through `ε`. The price model has no weather term yet, so load and price are currently linked only through `E_annual` and the price level. Do we condition the price residual on the same weather year, and how do we separate weather from fuel-price effects in the price history?
- **Q5 Tariff.** Is `p_tariff` the energy component only (current assumption) or the full end-customer price minus grid fees, levies and taxes?
- **Q6 Temperature response calibration.** We have no metered household data to estimate `g`. Candidates: the temperature sensitivity of total German load (SMARD), scaled to the household share; published household sensitivities; or a stated assumption for scenario analysis. Until this is decided, `g` is a parameter, not an estimate.
