# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wordbadha/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wordbadha/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A cohort-flow forecaster for GTA VI trophy earn rates on Exophase: for every PlayStation 5 trophy it forecasts five quantiles (5/25/50/75/95%) of the percent of tracked players who will hold it one week after launch, from the house's recording of the trophy page before each case's checkpoint.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every `*.in` case in the input directory and writes `<id>.out` per case to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## Method

An earn rate is a ratio — earners over players-tracked — and its two halves move for different reasons, so this entry forecasts them separately and divides:

- **Players tracked** — a launch-week growth curve fitted to the reads: a power law in time since release (with a fitted start offset), blended with a saturating exponential, weighted by a holdout test (fit on the first 70% of reads, scored on the last 30%) so the shape that actually extrapolates forward wins.
- **Earners per trophy** — earners ride a fraction of the tracked-count growth (the dilution exponent, regressed per trophy from the refreshes the history actually shows, shrunk toward a prior that common trophies dilute less than rare ones), plus the existing cohort's grind: the earner growth the tracked-count growth does not explain over the last day, damped over ~2 days.
- **Two more opinions** — a logit-linear trend in log time since release, and the recent logit velocity over the last day, damped over ~48 hours; blended with the flow forecast and the last read's rate in logit space.
- **Story logic** — the chapters run in order, the final mission precedes 100% completion, and the platinum needs every trophy, so the medians along that chain cannot rise (isotonic regression on the logit scale, platinum capped at the lowest other median).
- **Honest spread** — a random-walk term in the horizon, the opinions' disagreement, and a refresh term for Exophase's batch jumps. When the whole history is one frozen page (launch night), a stale-page prior applies the dilution the next refresh brings instead of pretending the frozen rates persist.

Two runs on the same case write the same bytes; a case runs in ~0.1 s, far inside the 8 s cap.

---

<sub>Entered on [make.fun](https://make.fun/?ref=czd88s7a&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
