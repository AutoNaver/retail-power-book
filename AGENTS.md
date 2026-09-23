# retail-power-book

Instructions for coding agents (Codex, Claude Code) working in this repository.

## Purpose

A simulation and hedging framework for a German retail power supply book, built with the methods of banking asset-liability management. The core idea: a retail power book and a bank's non-maturing deposit book are the same problem — uncertain volume, customer optionality, no contractual maturity. See `README.md` and `docs/design.md`.

## Commands

- Install: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- All hooks: `uv run pre-commit run --all-files`

Run the tests and the linter before saying a task is done.

## Layout

```
src/rpb/
  data/       # loaders and caching, no analysis
  models/     # load.py, price.py
  hedging/    # replication.py (maintainer-owned, see below)
  risk/       # metrics.py, scenarios.py
  config.py
tests/
  fixtures/   # small committed datasets; tests never use the network
notebooks/    # exploration only, never imported
configs/
docs/
```

## Conventions

- Python 3.12. Type hints on every public function.
- Notebooks import from `src/` and plot. No logic in notebooks.
- Randomness: pass a `numpy.random.Generator` explicitly. Never use global `np.random` state. Tests fix their seeds.
- Every model function has at least one test against a known result (closed form or a synthetic case with a known answer), not only a "runs without error" test.
- No network access in tests. Use `tests/fixtures/`.
- Configuration comes from files in `configs/`. No hard-coded paths.
- Secrets come from environment variables (loaded from `.env`, which is never committed).

## Domain rules

- Market: DE-LU bidding zone, day-ahead, hourly resolution.
- Units: prices in EUR/MWh, energy in MWh, power in MW. Put the unit in the name when it's ambiguous, e.g. `price_eur_mwh`.
- Prices can be negative. Never clip, floor or log-transform prices.
- Time: store timezone-aware UTC. Convert to Europe/Berlin only for display and calendar features. DST days have 23 and 25 hours. Never assume 24 hours per day or 8,760 hours per year.
- Peak: Monday–Friday 08:00–20:00 Europe/Berlin. Check holiday treatment against the EEX contract specification before relying on it.
- Data: SMARD (CC BY 4.0 — credit it) and ENTSO-E (token in `ENTSOE_API_TOKEN`).

## Scope

In scope: residential fixed-price book, day-ahead only, monthly and quarterly baseload and peakload forwards.

Out of scope unless an issue explicitly asks for it: intraday, balancing, gas, ETS, batteries and flexibility, churn, price elasticity, any UI.

## Working rules

- One issue, one branch, one PR. Branch names: `feat/<issue>-<slug>` or `test/<issue>-<slug>`.
- Never push to `main`. Never merge.
- Do not modify `src/rpb/hedging/`. The maintainer writes the replication layer by hand. Agents may review it and write tests for it.
- Never change a test just to make it pass. If a test looks wrong, say so in the PR description.
- Keep PRs small. Aim for under 400 changed lines.

## Review guidelines

Flag as P1:

- Unit errors (EUR/MWh vs EUR/kWh, MW vs MWh).
- Timezone errors: naive datetimes, local time in storage, days assumed to have 24 hours.
- Unseeded randomness, silent NaN propagation, explicit matrix inversion where `solve`/`lstsq` would do.
- Tests that don't assert a known result.
- Logic added to notebooks.
- Any agent change to `src/rpb/hedging/`.
- Secrets in code, or network calls in tests.
- Changes beyond the scope of the linked issue.
