# retail-power-book

A bank's non-maturing deposit book and a retail power supply book are the same problem. In both, the volume is uncertain, the customer holds an option they can exercise at any time (withdraw savings, switch supplier, consume more or less), and nothing has a contractual maturity. Banking asset-liability management has spent decades building tools for this: behavioural models of volume, replicating portfolios that turn a maturity-less position into a hedgeable one, and risk metrics over rate scenarios. This project applies those methods to a German fixed-price residential power book, simulating load and day-ahead prices for the DE-LU zone and hedging the book with baseload and peakload forwards.

## Data sources

- Day-ahead prices (DE-LU): Bundesnetzagentur | SMARD.de, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Hourly air temperature (station observations): Deutscher Wetterdienst (DWD), Climate Data Center, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
