"""
stats.py — analysis utilities for the TSMOM backtest.

All functions operate on pandas Series/DataFrames indexed by date. The notebook
wires them to the output of tsmom_monthly.backtest(). Typical setup:

    port, position, costs = backtest(prices, ...)
    gross = port + costs.mean(axis=1)                 # portfolio gross return
    turnover = portfolio_turnover(position)           # from tsmom_monthly (counts the first entry)
    spy = prices["SPY"].pct_change(fill_method=None).reindex(port.index)
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

FIG_DIR = "figures"


def _max_drawdown(r):
    """Max drawdown with the starting equity (1.0) counted as a peak."""
    cum = (1 + r).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, cum.values])[1:]
    return (cum.values / peak - 1).min()


def _save_show(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# CORE PERFORMANCE STATS
# ============================================================

def perf(returns, rf_annual=0.0, periods_per_year=252):
    """Standard performance block.

    Sharpe = mean(excess) / std * sqrt(252), arithmetic. With rf_annual=0 it is
    computed on raw returns (no cash subtracted), matching the strategy script.
    """
    r = returns.dropna()
    if len(r) < 2:
        return {}
    excess = r - rf_annual / periods_per_year
    cum = (1 + r).cumprod()
    cagr = cum.iloc[-1] ** (periods_per_year / len(r)) - 1
    max_dd = _max_drawdown(r)
    return {
        "CAGR": cagr,
        "Ann Vol": r.std() * np.sqrt(periods_per_year),
        "Sharpe": excess.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan,
        "Max DD": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "Total Return": cum.iloc[-1] - 1,
        "N days": len(r),
    }


def print_perf(stats, header=""):
    if header:
        print(f"=== {header} ===")
    for k, v in stats.items():
        if k in ("Sharpe", "Calmar"):
            print(f"  {k:14s}: {v:>7.2f}")
        elif k == "N days":
            print(f"  {k:14s}: {v:>7d}")
        else:
            print(f"  {k:14s}: {v:>7.2%}")
    print()


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL ON SHARPE
# ============================================================

def bootstrap_sharpe_ci(returns, n_boot=2000, block=21, seed=0, alpha=0.05):
    """95% CI for Sharpe via circular block bootstrap (blocks keep autocorrelation
    and volatility clustering). Returns (lower, upper)."""
    r = returns.dropna().values
    T = len(r)
    if T < block * 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    off = np.arange(block)
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        sample = r[(starts[:, None] + off[None, :]).ravel()[:T] % T]
        sd = sample.std(ddof=1)
        out[i] = sample.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    return tuple(np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


# ============================================================
# REGRESSION ON A BENCHMARK
# ============================================================

def regress_on_benchmark(strategy_returns, benchmark_returns, rf_annual=0.0):
    """OLS of strategy return on benchmark return with Newey-West (HAC, 5 lags)
    standard errors. Returns annualised alpha, beta, t(alpha), R^2, correlation."""
    df = pd.concat([strategy_returns.rename("strat"),
                    benchmark_returns.rename("bench")], axis=1).dropna()
    rf_daily = rf_annual / 252
    y = df["strat"] - rf_daily
    x = sm.add_constant(df["bench"] - rf_daily)
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "Alpha (ann)": model.params["const"] * 252,
        "Beta": model.params["bench"],
        "t(alpha)": model.tvalues["const"],
        "t(beta)": model.tvalues["bench"],
        "R^2": model.rsquared,
        "Corr": df["strat"].corr(df["bench"]),
        "N days": len(df),
    }


def print_regression(stats, header="Regression on benchmark"):
    print(f"=== {header} ===")
    for k, v in stats.items():
        if k in ("Beta", "t(alpha)", "t(beta)", "R^2", "Corr"):
            print(f"  {k:14s}: {v:>7.3f}")
        elif k == "N days":
            print(f"  {k:14s}: {v:>7d}")
        else:
            print(f"  {k:14s}: {v:>7.2%}")
    print()


# ============================================================
# CONDITIONAL PERFORMANCE IN BENCHMARK DRAWDOWNS
# ============================================================

def conditional_on_drawdown(strategy_returns, benchmark_returns, threshold=-0.10):
    """Strategy vs benchmark inside and outside benchmark drawdowns deeper than
    |threshold|. A day is 'in drawdown' while the benchmark is below its prior peak
    by more than that. Returns are annualised over the stitched-together days, so
    read them as rough regime averages, not literal calendar returns."""
    df = pd.concat([strategy_returns.rename("strat"),
                    benchmark_returns.rename("bench")], axis=1).dropna()
    bench_cum = (1 + df["bench"]).cumprod()
    in_dd = (bench_cum / bench_cum.cummax() - 1) < threshold

    def ann(r):
        if len(r) < 2:
            return np.nan
        return (1 + r).prod() ** (252 / len(r)) - 1

    rows = {}
    for label, mask in (("In drawdown", in_dd), ("Outside drawdown", ~in_dd)):
        rows[label] = {
            "Days": int(mask.sum()),
            "Share": mask.mean(),
            "Strategy Ret": ann(df.loc[mask, "strat"]),
            "Bench Ret": ann(df.loc[mask, "bench"]),
        }
    return rows


def print_conditional(rows, header="Conditional on benchmark drawdown (>10%)"):
    print(f"=== {header} ===")
    print(f"  {'Regime':<20}{'Days':>8}{'Share':>9}{'StratRet':>10}{'BenchRet':>10}")
    for label, r in rows.items():
        print(f"  {label:<20}{r['Days']:>8d}{r['Share']:>9.1%}"
              f"{r['Strategy Ret']:>10.2%}{r['Bench Ret']:>10.2%}")
    print()


# ============================================================
# SUB-PERIOD ANALYSIS
# ============================================================

def subperiod_stats(strategy_returns, benchmark_returns=None, splits=("2020-01-01",)):
    """Perf for each sub-period. `splits` are the start dates of the later periods,
    e.g. ("2020-01-01",) gives [first, 2020) and [2020, last]."""
    s_all = strategy_returns.dropna()
    b_all = None if benchmark_returns is None else benchmark_returns.reindex(s_all.index)
    bounds = [s_all.index[0]] + [pd.Timestamp(x) for x in splits] + [None]

    out = {}
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (s_all.index >= lo) & ((s_all.index < hi) if hi is not None else True)
        sub = s_all[mask]
        if len(sub) < 2:
            continue
        label = f"{sub.index[0].year}-{sub.index[-1].year}"
        out[label] = {"Strategy": perf(sub)}
        if b_all is not None:
            out[label]["Benchmark"] = perf(b_all[mask])
    return out


def print_subperiod(rows, header="Sub-period results"):
    print(f"=== {header} ===")
    for period, d in rows.items():
        s = d["Strategy"]
        print(f"  {period}  Strategy:  CAGR {s['CAGR']:>6.2%}  Vol {s['Ann Vol']:>6.2%}  "
              f"Sharpe {s['Sharpe']:>5.2f}  MaxDD {s['Max DD']:>6.1%}")
        if "Benchmark" in d:
            b = d["Benchmark"]
            print(f"  {'':<9}  Benchmark: CAGR {b['CAGR']:>6.2%}  Vol {b['Ann Vol']:>6.2%}  "
                  f"Sharpe {b['Sharpe']:>5.2f}  MaxDD {b['Max DD']:>6.1%}")
    print()


# ============================================================
# CALENDAR YEARS AND WORST DRAWDOWNS
# ============================================================

def calendar_year_returns(strategy_returns, benchmark_returns=None):
    """Return per calendar year (first and last years may be partial)."""
    def yearly(r):
        r = r.dropna()
        return r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)

    out = pd.DataFrame({"Strategy": yearly(strategy_returns)})
    if benchmark_returns is not None:
        out["Benchmark"] = yearly(benchmark_returns.reindex(strategy_returns.index))
    return out


def print_calendar(df, header="Calendar-year returns"):
    print(f"=== {header} ===")
    print("  " + f"{'Year':<6}" + "".join(f"{c:>12}" for c in df.columns))
    for yr, row in df.iterrows():
        print("  " + f"{yr:<6}" + "".join(f"{v:>12.1%}" for v in row))
    print()


def top_drawdowns(returns, n=5):
    """The n deepest drawdown episodes: peak date, trough date, recovery date
    (None if not yet recovered), depth, and days from peak to recovery/end."""
    r = returns.dropna()
    cum = (1 + r).cumprod()
    dd = cum / np.maximum.accumulate(np.r_[1.0, cum.values])[1:] - 1
    episode = (dd == 0).cumsum()
    rows = []
    for _, sub in dd.groupby(episode):
        if sub.min() >= 0:
            continue
        last = sub.index[-1]
        pos = dd.index.get_loc(last)
        recovery = dd.index[pos + 1] if pos + 1 < len(dd) else None
        end = recovery if recovery is not None else last
        rows.append({"Peak": sub.index[0].date(), "Trough": sub.idxmin().date(),
                     "Recovery": recovery.date() if recovery is not None else None,
                     "Depth": sub.min(), "Days": (end - sub.index[0]).days})
    return pd.DataFrame(rows).sort_values("Depth").head(n).reset_index(drop=True)


def print_top_drawdowns(df, header="Deepest drawdowns"):
    print(f"=== {header} ===")
    print(f"  {'Peak':<12}{'Trough':<12}{'Recovery':<12}{'Depth':>8}{'Days':>7}")
    for _, r in df.iterrows():
        print(f"  {str(r['Peak']):<12}{str(r['Trough']):<12}{str(r['Recovery']):<12}"
              f"{r['Depth']:>8.1%}{r['Days']:>7d}")
    print()


# ============================================================
# PORTFOLIO COMBINATION TEST
# ============================================================

def portfolio_combination(strategy_returns, benchmark_returns, weights=(0.0, 0.2, 0.3, 0.4)):
    """Daily-rebalanced mix: (1-w) * benchmark + w * strategy. Treats the strategy as
    a fully-funded sleeve; the mix is an illustration, not a trading recommendation."""
    df = pd.concat([strategy_returns.rename("strat"),
                    benchmark_returns.rename("bench")], axis=1).dropna()
    return {f"{round(w * 100)}% strategy": perf((1 - w) * df["bench"] + w * df["strat"])
            for w in weights}


def print_portfolio_combination(rows, header="Portfolio combination: benchmark + strategy"):
    print(f"=== {header} ===")
    print(f"  {'Allocation':<16}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>9}{'Calmar':>9}")
    for label, s in rows.items():
        print(f"  {label:<16}{s['CAGR']:>9.2%}{s['Ann Vol']:>9.2%}{s['Sharpe']:>9.2f}"
              f"{s['Max DD']:>9.1%}{s['Calmar']:>9.2f}")
    print()


# ============================================================
# COST SENSITIVITY
# ============================================================

def cost_sensitivity(gross_returns, turnover, bps_list=(0, 1, 3, 5, 10, 20)):
    """perf() for each per-trade cost. `turnover` = portfolio turnover per day
    (mean over assets of |change in position|)."""
    return {bps: perf(gross_returns - turnover * bps / 10000.0) for bps in bps_list}


def print_cost_sensitivity(rows, header="Cost sensitivity"):
    print(f"=== {header} ===")
    print(f"  {'Cost (bps)':<12}{'CAGR':>10}{'Sharpe':>10}{'MaxDD':>10}")
    for bps, s in rows.items():
        print(f"  {bps:<12}{s['CAGR']:>10.2%}{s['Sharpe']:>10.2f}{s['Max DD']:>10.1%}")
    print()


# ============================================================
# PER-ASSET P&L CONTRIBUTION
# ============================================================

def asset_contribution(positions, asset_returns):
    """Gross (pre-cost) P&L per asset, in units of per-asset capital, and each asset's
    share of the total. Shares can exceed 100% or be negative if some assets lose money."""
    rets = asset_returns.reindex(positions.index)[positions.columns]
    contrib = (positions.shift(1) * rets).sum()
    share = contrib / contrib.sum() if contrib.sum() != 0 else contrib * np.nan
    return pd.DataFrame({"P&L": contrib, "Share": share}).sort_values("P&L", ascending=False)


def print_asset_contribution(df, header="Per-asset gross P&L contribution"):
    print(f"=== {header} ===")
    print(f"  {'Asset':<8}{'P&L':>12}{'Share':>10}")
    for asset, row in df.iterrows():
        print(f"  {asset:<8}{row['P&L']:>12.4f}{row['Share']:>10.1%}")
    print()


# ============================================================
# RANDOM-WALK SANITY CHECK  (runs the REAL backtest, not a copy)
# ============================================================

def random_walk_test(n_seeds=20, n_days=5000, n_assets=8, cost_bps=0, seed=0):
    """Run tsmom_monthly.backtest() on zero-drift random walks. With no look-ahead the
    mean Sharpe over many seeds should be about 0 (use cost_bps=0; costs make it negative)."""
    from tsmom_monthly import backtest, HORIZONS, SKIP_MONTHS, TARGET_VOL, VOL_WINDOW, MAX_LEVERAGE

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-03", periods=n_days, freq="B")
    sharpes = []
    for _ in range(n_seeds):
        ret = rng.normal(0, 0.01, size=(n_days, n_assets))
        prices = pd.DataFrame(100 * np.exp(np.cumsum(ret, axis=0)), index=idx,
                              columns=[f"A{i}" for i in range(n_assets)])
        port, _, _ = backtest(prices, HORIZONS, SKIP_MONTHS, TARGET_VOL, VOL_WINDOW,
                              MAX_LEVERAGE, cost_bps)
        sharpes.append(port.mean() / port.std() * np.sqrt(252))
    sharpes = np.array(sharpes)
    return {"Mean Sharpe": sharpes.mean(), "Std Err": sharpes.std(ddof=1) / np.sqrt(len(sharpes)),
            "Min": sharpes.min(), "Max": sharpes.max(), "N seeds": len(sharpes)}


# ============================================================
# PLOTS  (saved to figures/ and shown)
# ============================================================

def plot_equity_curve(strategy_returns, benchmark_returns, split_date=None,
                      name="equity_curve.png", title="Cumulative return"):
    fig, ax = plt.subplots(figsize=(11, 6))
    cum_s = (1 + strategy_returns.fillna(0)).cumprod()
    cum_b = (1 + benchmark_returns.reindex(strategy_returns.index).fillna(0)).cumprod()
    ax.plot(cum_s.index, cum_s, label="TSMOM (net)", color="steelblue", lw=1.5)
    ax.plot(cum_b.index, cum_b, label="SPY buy-and-hold", color="gray", alpha=0.7, lw=1.5)
    if split_date is not None:
        ax.axvline(pd.Timestamp(split_date), color="black", ls="--", alpha=0.5,
                   label=f"split {split_date}")
    ax.set_yscale("log")
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_show(fig, name)


def plot_drawdowns(strategy_returns, benchmark_returns, name="drawdowns.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for r, label, color in ((strategy_returns, "TSMOM", "steelblue"),
                            (benchmark_returns, "SPY", "gray")):
        cum = (1 + r.reindex(strategy_returns.index).fillna(0)).cumprod()
        dd = cum / cum.cummax() - 1
        ax.fill_between(dd.index, dd, 0, alpha=0.4, color=color, label=label)
    ax.set_ylabel("Drawdown")
    ax.set_title("Drawdowns: TSMOM vs SPY")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_show(fig, name)


def plot_rolling_sharpe(strategy_returns, window=252, name="rolling_sharpe.png"):
    r = strategy_returns.dropna()
    roll = r.rolling(window).mean() / r.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(roll.index, roll, color="steelblue", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"Rolling {window}-day Sharpe")
    ax.set_title("Rolling Sharpe")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_show(fig, name)


def plot_rolling_correlation(strategy_returns, benchmark_returns, window=252,
                             name="rolling_corr.png"):
    df = pd.concat([strategy_returns.rename("s"), benchmark_returns.rename("b")],
                   axis=1).dropna()
    roll = df["s"].rolling(window).corr(df["b"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(roll.index, roll, color="steelblue", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"Rolling {window}-day correlation")
    ax.set_title("Rolling correlation: TSMOM vs SPY")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_show(fig, name)
    