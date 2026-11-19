"""Cohort-flow forecaster for GTA VI trophy earn rates on Exophase.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles (5/25/50/75/95%) of its Exophase earn rate (percent of tracked
players who hold it) at a case's target time, from the house's recording of
the trophy page before the case's checkpoint.

Why forecast earners and players separately instead of the rate itself?
A rate is a ratio, earners over players-tracked, and the two halves move for
different reasons. Exophase refreshes its counts in batches: at a refresh a
wave of new, mostly fresh players joins the tracked set, so the denominator
jumps while the numerator only grows by the trophies those newcomers have
already earned (early trophies) or barely at all (late ones). A model that
extrapolates the rate directly mistakes that dilution for players losing
trophies. This program therefore forecasts the two halves and divides:

  * players tracked N(t): a launch-week growth curve fitted to the reads
    (a power law in time since release, blended with a saturating
    exponential, whichever fits the recorded growth better);
  * earners E_i(t) per trophy: E grows with N, but only by a fraction
    gamma_i of it -- the dilution exponent, regressed per trophy from the
    refreshes the history actually shows (how much did this trophy's
    earners jump when the tracked count jumped?), shrunk toward a prior
    that common trophies dilute less than rare ones;
  * a logit-linear trend in log time since release as a second opinion,
  * the trophy's recent velocity -- how fast its logit rate actually moved
    over the last day of reads, extrapolated with a ~48-hour damping
    timescale, so a trophy the existing cohort is visibly grinding
    through keeps climbing instead of freezing at the dilution view --
    as a third, and the last read's rate as a fourth, blended in logit
    space.

The median is then cleaned up with the story's own logic: a player who
holds the platinum holds every trophy, one at 100% completion finished the
final mission, and the chapters run in order, so the medians along that
chain cannot rise (pool-adjacent-violators on the logit scale).

The spread is honest about launch week: a random-walk term in the horizon,
the disagreement between the three median opinions, and a refresh term,
because a batch refresh can move a rate by a logit unit or more in one
step. When the whole history is one frozen page -- the launch-night
situation, where Exophase has not refreshed since release -- a stale-page
prior applies the dilution the next refresh brings, instead of pretending
the frozen rates will simply persist.

Standard library only, no network, no randomness: two runs on the same
case write the same bytes. Reads every *.in in the input directory
($MAKEFUN_INPUT_DIR, else /input) and writes <id>.out per case to the
output directory ($MAKEFUN_OUTPUT_DIR, else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE_FALLBACK_HOURS = 0.0  # hours since release; 0 when unparseable


def clamp_pct(x: float) -> float:
    """A percent onto the range the logit is defined on (0.01..99.99)."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0.0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript rounds it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_between(later: datetime, earlier: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, rmse)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return my, 0.0, math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    rmse = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rmse


def tier_prior_rate(tier: str) -> float:
    """A launch-night-scale rate for a trophy never seen in the history."""
    return {"platinum": 0.3, "gold": 0.5, "silver": 0.5}.get(tier, 1.0)


def extract_series(case: dict):
    """Turn a case's history into aligned time / players / rates series.

    Returns (times_h, players, rates, release_dt, target_h) where times_h
    are hours since the game's release, players the tracked count per read,
    rates[i][j] trophy i's rate at read j (nulls filled forward, then
    backward, then by a tier prior), and target_h the target in the same
    clock. Reads with no usable players count are dropped.
    """
    release = case.get("game", {}).get("release_date", "2026-11-19")
    try:
        release_dt = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
    except ValueError:
        release_dt = parse_time(case["history"][0]["at"]) if case.get("history") else datetime(2026, 11, 19, tzinfo=timezone.utc)
    at_dt = parse_time(case["at"])
    target_dt = parse_time(case["target"])
    target_h = hours_between(target_dt, release_dt)

    trophies = case["trophies"]
    n_tr = len(trophies)
    times_h, players, raw_rates = [], [], []
    for read in case.get("history") or []:
        n = read.get("players_tracked")
        rates = read.get("rates") or []
        if not isinstance(n, (int, float)) or n <= 0 or len(rates) < n_tr:
            continue
        times_h.append(hours_between(parse_time(read["at"]), release_dt))
        players.append(float(n))
        raw_rates.append([float(r) if r is not None else None for r in rates[:n_tr]])

    if not times_h:
        return [], [], [], release_dt, target_h
    # Fill nulls: forward, then backward, then the tier prior.
    filled = [[None] * len(times_h) for _ in range(n_tr)]
    for i in range(n_tr):
        last = None
        for j in range(len(times_h)):
            if raw_rates[j][i] is not None:
                last = raw_rates[j][i]
            filled[i][j] = last
        nxt = None
        for j in range(len(times_h) - 1, -1, -1):
            if filled[i][j] is None:
                filled[i][j] = nxt if nxt is not None else tier_prior_rate(trophies[i]["tier"])
            else:
                nxt = filled[i][j]
    return times_h, players, filled, release_dt, target_h


def _power_fit(times_h, players):
    """Power law N ~ (t + c)^b in log space, offset c by grid.

    Launch-week adoption is a power law in time since release, but the
    clock starts at release while the first players arrive a little
    later; the offset lets the curve bend at the start instead of forcing
    the shape through t = 0. The offset with the best in-sample log fit
    wins. Returns a predictor, or None.
    """
    if len(times_h) < 2:
        return None
    log_n = [math.log(n) for n in players]
    best = None
    for c in (0.0, 1.0, 3.0, 6.0, 12.0, 24.0):
        xs = [math.log(max(0.0, t) + c + 1.0) for t in times_h]
        a, b, rmse = ols(xs, log_n)
        if best is None or rmse < best[0]:
            best = (rmse, a, min(2.0, max(0.0, b)), c)
    _, a, b, c = best
    return lambda t: math.exp(a + b * math.log(max(0.0, t) + c + 1.0))


def _sat_fit(times_h, players):
    """Saturating exponential N = Ninf * (1 - exp(-t / tau)), tau by grid.
    Returns a predictor, or None."""
    if len(times_h) < 2:
        return None
    best = None
    for tau in (6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0):
        xs = [1.0 - math.exp(-max(0.0, t) / tau) for t in times_h]
        denom = sum(x * x for x in xs)
        if denom <= 0.0:
            continue
        ninf = sum(x * n for x, n in zip(xs, players)) / denom
        if ninf <= 0.0:
            continue
        errs = [math.log(max(1.0, ninf * x)) - math.log(n) for x, n in zip(xs, players)]
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        if best is None or rmse < best[0]:
            best = (rmse, tau, ninf)
    if best is None:
        return None
    _, tau, ninf = best
    return lambda t: ninf * (1.0 - math.exp(-max(0.0, t) / tau))


def fit_players(times_h, players, target_h):
    """Forecast the players-tracked count at the target.

    Blends a power law in time since release with a saturating exponential.
    The blend weights come from a holdout test, not the in-sample fit: each
    model is fitted on the first 70% of reads and scored on the last 30%,
    so the weight rewards the shape that actually extrapolates forward --
    the saturating curve can hug the history and still go flat too early.
    A flat history forecasts no growth at all.
    """
    last = players[-1]
    if len(times_h) < 2 or min(players) == max(players):
        return last
    pow_full = _power_fit(times_h, players)
    sat_full = _sat_fit(times_h, players)

    def holdout_rmse(fitfn):
        k = int(0.7 * len(times_h))
        f = fitfn(times_h[:k], players[:k])
        if f is None:
            return None
        errs = [math.log(max(1.0, f(t))) - math.log(n)
                for t, n in zip(times_h[k:], players[k:])]
        return math.sqrt(sum(e * e for e in errs) / len(errs))

    if len(times_h) >= 10:
        r_pow = holdout_rmse(_power_fit)
        r_sat = holdout_rmse(_sat_fit)
        w_pow = 1.0 / ((r_pow if r_pow is not None else 1.0) + 0.02)
        w_sat = 1.0 / ((r_sat if r_sat is not None else 1.0) + 0.02) if r_sat is not None else 0.0
    else:
        w_pow, w_sat = 1.0, 1.0 if sat_full is not None else 0.0
    num, den = 0.0, 0.0
    if pow_full is not None:
        num += w_pow * math.log(max(1.0, pow_full(target_h)))
        den += w_pow
    if sat_full is not None and w_sat > 0.0:
        num += w_sat * math.log(max(1.0, sat_full(target_h)))
        den += w_sat
    pred = math.exp(num / den) if den > 0.0 else last
    # Never forecast fewer players than the last read, and never an absurd
    # launch-week explosion: cap the growth over the last read at 60x.
    return min(max(pred, last), last * 60.0)


def dilution_prior(rate_pct: float) -> float:
    """How much of a tracked-count jump becomes earners, a priori.

    Common trophies ride most of a newcomer wave (new players earn them
    quickly); rare ones barely move. 0.05..0.95 on the last rate.
    """
    r = clamp_pct(rate_pct)
    return 0.05 + 0.90 * r / (r + 1.0)


def dilution_exponents(times_h, players, rates, n_tr):
    """Per-trophy dilution exponent gamma_i, regressed from the refreshes.

    Across consecutive reads where the tracked count moved, regress the
    change in log earners on the change in log players through the origin,
    then shrink toward the prior by how many refreshes were seen.
    """
    gammas = []
    for i in range(n_tr):
        num = 0.0
        den = 0.0
        pairs = 0
        for j in range(1, len(times_h)):
            dn = math.log(players[j] / players[j - 1])
            if abs(dn) < 1e-9:
                continue
            e_prev = rates[i][j - 1] / 100.0 * players[j - 1]
            e_now = rates[i][j] / 100.0 * players[j]
            if e_prev <= 0.0 or e_now <= 0.0:
                continue
            de = math.log(e_now / e_prev)
            num += dn * de
            den += dn * dn
            pairs += 1
        prior = dilution_prior(rates[i][-1])
        if den > 0.0 and pairs > 0:
            est = min(1.0, max(0.0, num / den))
            gammas.append((pairs * est + 2.0 * prior) / (pairs + 2.0))
        else:
            gammas.append(prior)
    return gammas


def trend_logit(times_h, rates_i, target_h):
    """Logit-linear trend of one trophy's rate against log time since release.

    Slope shrunk toward zero when the history barely varies, so a frozen
    page forecasts persistence rather than a runaway line.
    """
    xs = [math.log1p(max(0.0, t)) for t in times_h]
    ys = [logit(r) for r in rates_i]
    distinct = len(set(round(y, 6) for y in ys))
    if len(xs) < 3 or distinct < 2:
        return ys[-1]
    a, b, _ = ols(xs, ys)
    shrink = distinct / (distinct + 3.0)
    return a + b * shrink * math.log1p(max(0.0, target_h))


def earner_progress(times_h, players, rates_i, gamma):
    """Per-day log growth of a trophy's earners that the tracked-count
    growth does not explain, over the last day of reads.

    At a refresh, earners jump for two reasons: newcomers who already hold
    the trophy, and the existing cohort ageing into it. The dilution
    exponent gamma carries the first; the leftover log growth per day is
    the second -- the cohort's grind -- floored at zero (a trophy nobody
    new earns) and capped at 0.7/day, so a noisy day cannot promise a
    tenfold army of new holders.
    """
    last_t = times_h[-1]
    j0 = 0
    for j, t in enumerate(times_h):
        if t >= last_t - 24.0:
            j0 = j
            break
    dt_days = (times_h[-1] - times_h[j0]) / 24.0
    if dt_days < 0.25:
        return 0.0
    e0 = rates_i[j0] / 100.0 * players[j0]
    e1 = rates_i[-1] / 100.0 * players[-1]
    if e0 <= 0.0 or e1 <= 0.0:
        return 0.0
    resid = math.log(e1 / e0) - gamma * math.log(players[-1] / players[j0])
    return min(0.7, max(0.0, resid / dt_days))


def velocity_logit(times_h, rates_i, target_h):
    """One trophy's recent logit velocity, damped over the horizon.

    Takes the slope of the trophy's logit rate over the last day of reads
    (or the whole history when it is shorter), then extrapolates it with a
    48-hour persistence timescale: the cohort's grind keeps paying for
    about two days, then fades, so a fast-climbing late trophy is not
    frozen at its dilution level and a fading one is not run away with.
    The move is shrunk by its own size, so a flat or noisy page forecasts
    persistence. A falling rate (dilution winning) extrapolates down.
    """
    last_t = times_h[-1]
    j0 = 0
    for j, t in enumerate(times_h):
        if t >= last_t - 24.0:
            j0 = j
            break
    dt = last_t - times_h[j0]
    base = logit(rates_i[-1])
    if dt < 6.0:
        return base
    move = logit(rates_i[-1]) - logit(rates_i[j0])
    if abs(move) < 1e-9:
        return base
    slope = move / dt  # logit units per hour
    horizon_h = max(0.0, target_h - last_t)
    h_eff = 48.0 * (1.0 - math.exp(-horizon_h / 48.0))
    shrink = abs(move) / (abs(move) + 0.5)
    return base + slope * shrink * h_eff


def stale_page_dilution(case, times_h, players, rates, medians):
    """Dilute the medians when the whole history is one frozen page.

    Exophase refreshes in batches: a page unchanged for hours on launch
    week is stale, and the next refresh admits a wave of fresh players who
    have earned little. The launch-night refresh the rules describe took
    the tracked count about 6x after ~4 hours frozen, so the expected
    multiplier grows about 1.2x per frozen hour, capped at 8x. A trophy
    keeps the share of the wave its own earliness earns it: the
    near-universal one keeps everything, early ones keep about a third,
    rare ones about an eighth. Only fires on a fully flat history, so a
    recording that already shows refreshes is left to the data.
    """
    if len(times_h) < 2:
        return medians
    if min(players) != max(players):
        return medians
    first_rates = tuple(rates[i][0] for i in range(len(rates)))
    if any(tuple(rates[i][j] for i in range(len(rates))) != first_rates for j in range(1, len(times_h))):
        return medians
    stale_h = max(0.0, hours_between(parse_time(case["at"]), parse_time(case["history"][0]["at"])))
    if stale_h < 1.0:
        return medians
    mult = min(8.0, 1.0 + 1.2 * stale_h)
    out = []
    for med, row in zip(medians, rates):
        last_rate = row[-1]
        if last_rate >= 99.0:
            keep = 1.0
        elif last_rate >= 1.0:
            keep = 0.35
        else:
            keep = 0.12
        out.append(med * (1.0 + (mult - 1.0) * keep) / mult)
    return out


def chain_indices(trophies):
    """Indices of the story chain, in earning order, from descriptions.

    Chapter 1..4, then the final mission, then 100% completion, then the
    platinum: each step's holders are a subset of the step before's, so
    their rates cannot rise along the chain.
    """
    by_desc = {}
    for idx, tr in enumerate(trophies):
        by_desc.setdefault(tr["description"].strip().lower(), idx)
    chain = []
    for key in ("complete chapter 1.", "complete chapter 2.", "complete chapter 3.",
                "complete chapter 4.", "complete the final mission.",
                "attain 100% completion."):
        if key in by_desc:
            chain.append(by_desc[key])
    for idx, tr in enumerate(trophies):
        if tr["tier"] == "platinum":
            chain.append(idx)
            break
    return chain


def enforce_chain(medians_logit, chain):
    """Pool-adjacent-violators: medians non-increasing along the chain."""
    if len(chain) < 2:
        return medians_logit
    vals = [medians_logit[i] for i in chain]
    blocks = [[v, 1] for v in vals]  # [mean, weight]
    merged = []
    for mean, w in blocks:
        merged.append([mean, w])
        while len(merged) >= 2 and merged[-2][0] < merged[-1][0]:
            m2, w2 = merged.pop()
            m1, w1 = merged.pop()
            merged.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    out = list(medians_logit)
    pos = 0
    for mean, w in merged:
        for _ in range(w):
            out[chain[pos]] = mean
            pos += 1
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    times_h, players, rates, _release, target_h = extract_series(case)
    if not times_h:
        # No usable history at all: tier priors, very wide.
        rows = []
        for tr in trophies:
            med = tier_prior_rate(tr["tier"])
            rows.append({"id": tr["id"], "q": [round4(sigmoid(logit(med) + z * 2.0)) for z in Z]})
        return {"trophies": rows}

    n_target = fit_players(times_h, players, target_h)
    gammas = dilution_exponents(times_h, players, rates, n_tr)
    n_last = players[-1]
    growth = n_target / n_last

    medians, sigmas = [], []
    for i, tr in enumerate(trophies):
        last_rate = rates[i][-1]
        last_e = last_rate / 100.0 * n_last
        # Opinion 1: the cohort flow -- earners ride a fraction gamma of
        # the tracked-count growth, plus the existing cohort's grind
        # (unexplained earner growth, damped over ~2 days), then divide
        # by the new tracked count. Earners cannot exceed the tracked.
        if last_e > 0.0:
            prog = earner_progress(times_h, players, rates[i], gammas[i])
            h_eff_days = 2.0 * (1.0 - math.exp(-horizon_days / 2.0))
            e_target = last_e * (growth ** gammas[i]) * math.exp(prog * h_eff_days)
            e_target = min(e_target, n_target)
            flow_rate = min(99.5, 100.0 * e_target / n_target)
        else:
            flow_rate = 0.0
        # Opinion 2: the logit-linear trend in log time since release.
        trend_rate = sigmoid(trend_logit(times_h, rates[i], target_h))
        # Opinion 3: the recent velocity, damped over the horizon.
        vel_rate = sigmoid(velocity_logit(times_h, rates[i], target_h))
        # Opinion 4: persistence -- the last read's rate.
        med_logit = (0.35 * logit(flow_rate) + 0.20 * logit(trend_rate)
                     + 0.30 * logit(vel_rate) + 0.15 * logit(last_rate))
        medians.append(sigmoid(med_logit))
        # Spread: a random walk in the horizon, the opinions' disagreement,
        # and a refresh term for the batch jumps Exophase makes.
        opinions = [logit(flow_rate), logit(trend_rate), logit(vel_rate), logit(last_rate)]
        mean_op = sum(opinions) / len(opinions)
        disagree = math.sqrt(sum((o - mean_op) ** 2 for o in opinions) / len(opinions))
        sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2
                          + (0.5 * disagree) ** 2
                          + (0.35 * math.sqrt(min(horizon_days, 3.0))) ** 2)
        sigmas.append(min(2.0, max(0.35, sigma)))

    flat_history = (min(players) == max(players)
                    and all(tuple(rates[i][j] for i in range(n_tr)) == tuple(rates[i][0] for i in range(n_tr))
                            for j in range(len(times_h))))
    if flat_history:
        medians = stale_page_dilution(case, times_h, players, rates, medians)
        sigmas = [max(s, 0.9) for s in sigmas]

    med_logit = [logit(m) for m in medians]
    chain = chain_indices(trophies)
    med_logit = enforce_chain(med_logit, chain)
    # The platinum needs every trophy: it cannot exceed any other trophy.
    plat = next((idx for idx, tr in enumerate(trophies) if tr["tier"] == "platinum"), None)
    if plat is not None and n_tr > 1:
        cap = min(med_logit[i] for i in range(n_tr) if i != plat)
        med_logit[plat] = min(med_logit[plat], cap)

    rows = []
    for i, tr in enumerate(trophies):
        base = med_logit[i]
        # The near-universal trophy stays near-universal: tighten its band.
        sigma = 0.5 if rates[i][-1] >= 99.0 else sigmas[i]
        qs = [round4(sigmoid(base + z * sigma)) for z in Z]
        qs = [min(99.99, max(0.01, q)) for q in qs]
        qs.sort()
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def forecast_case_safe(case: dict, name: str) -> dict:
    """Never crash a case: fall back to persistence with a wide band."""
    try:
        return forecast_case(case, name)
    except Exception:
        rows = []
        last = {}
        try:
            hist = case.get("history") or []
            if hist:
                last = {case["trophies"][i]["id"]: (hist[-1]["rates"][i] or 0.0)
                        for i in range(len(case["trophies"]))}
        except Exception:
            last = {}
        for tr in case["trophies"]:
            rate = last.get(tr["id"])
            if rate is None:
                rate = tier_prior_rate(tr["tier"])
            rows.append({"id": tr["id"],
                         "q": [round4(sigmoid(logit(rate) + z * 1.5)) for z in Z]})
        return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case_safe(case, path.name)
        blob = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(blob.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over the 65,536 byte cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(blob, encoding="utf-8")


if __name__ == "__main__":
    main()
