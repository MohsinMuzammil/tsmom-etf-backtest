"""
Time-series momentum (TSMOM) on a diversified ETF basket.

Method
  1. Signal  : average of sign(past return) over HORIZONS (months), computed from
               month-end closes. +1 = all horizons up, -1 = all down.
  2. Sizing  : each asset targeted at TARGET_VOL using trailing realised vol
               (lagged one day), capped at MAX_LEVERAGE.
  3. Timing  : decided at the month-end close, held fixed until the next
               month-end, earning returns from the following day.
  4. Costs   : COST_BPS per dollar traded.
  5. Portfolio: simple average across assets (each asset has roughly equal risk).

Conventions / limitations
  - Sharpe = mean/std * sqrt(252) on daily returns, no cash subtracted.
  - No financing cost on leverage and no borrow fee on shorts.
  - ETF basket chosen with hindsight; adjusted closes include dividends.
  - The backtest starts on the first day every asset has a signal and a vol estimate.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
# TICKERS = ["SPY", "IWM", "EFA", "EEM", "VNQ", "TLT", "IEF", "LQD", "GLD", "SLV", "DBC"]
TICKERS = ["SPY", "EFA", "EEM", "TLT", "IEF", "DBC", "GLD", "VNQ"]
HORIZONS = [6, 12]          # months; signal = average of sign(return) over these
SKIP_MONTHS = 0             # skip most recent month (0 = include it)
START = "2006-01-01"        # all tickers above exist by Apr 2006
END   = "2026-01-01"

COST_BPS = 5                # per-trade transaction cost (basis points)

TARGET_VOL = 0.20           # 20% annualized target vol per asset
VOL_WINDOW = 60             # trailing days for realized vol
MAX_LEVERAGE = 2.0          # cap on position size per asset

PLOT = True
# ----------------------------


def load_prices(tickers=TICKERS, start=START, end=END, path="data/prices.csv"):
    """Daily adjusted closes, cached to disk so reruns use identical data.
    Delete data/prices.csv to force a fresh download."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        if set(tickers) <= set(px.columns):
            return px[tickers].loc[start:end]
    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False)["Close"][tickers]
    px.to_csv(path)
    return px


def month_end_days(idx):
    """Last TRADING day of each month (not the calendar month-end label)."""
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def compute_multi_horizon_signal(monthly_prices, horizons_months, skip=0):
    """Average of sign(return) over several horizons, from MONTH-END prices.
    The value at month-end M uses prices up to M only."""
    signals = [np.sign(monthly_prices.shift(skip) / monthly_prices.shift(h + skip) - 1)
               for h in horizons_months]
    return sum(signals) / len(signals)          # range -1..+1


def realized_vol(daily_returns, window):
    """Annualized trailing realized vol, shifted by 1 day to avoid look-ahead."""
    return (daily_returns.rolling(window).std() * np.sqrt(252)).shift(1)


def backtest(daily_prices, horizons, skip, target_vol, vol_window,
             max_lev, cost_bps):
    """Monthly rebalancing: signals and sizes are decided at each month-end close
    and held fixed until the next month-end.
    Returns (port, position, costs) trimmed to the common start date.
        port     : daily NET portfolio return (average across assets)
        position : daily per-asset exposure held at the close (units of capital)
        costs    : daily per-asset transaction cost
    Gross portfolio return = port + costs.mean(axis=1).
    """
    idx = daily_prices.index
    daily_ret = daily_prices.pct_change(fill_method=None)
    me = month_end_days(idx)

    # Signal: computed from month-end prices only, then held until the next month-end.
    monthly = daily_prices.loc[me]
    direction = compute_multi_horizon_signal(monthly, horizons, skip).reindex(idx).ffill()

    # Size: inverse realized vol, capped
    vol = realized_vol(daily_ret, vol_window)
    size = (target_vol / vol).clip(upper=max_lev)

    target = direction * size
    # keep only the month-end decision and hold it fixed until the next month-end
    target = target.loc[me].fillna(0).reindex(idx).ffill()
    position = target.fillna(0)

    # Common start: the first day EVERY asset has a valid signal and vol estimate.
    ready = (direction.notna() & size.notna()).all(axis=1)
    if not ready.any():
        raise ValueError("No date where all assets have a signal; check TICKERS/START.")
    start = ready.idxmax()
    position.loc[position.index < start] = 0.0   # begin flat, so the first entry is charged

    # Position set at the close of t-1 earns day t's return
    gross = position.shift(1) * daily_ret
    costs = position.diff().abs() * (cost_bps / 10000.0)
    net = gross - costs

    # Each asset is vol-targeted, so a simple average gives roughly equal risk per asset
    port = net.mean(axis=1)
    return port.loc[start:], position.loc[start:], costs.loc[start:]


def portfolio_turnover(position):
    """Daily portfolio turnover = average over assets of |change in exposure|.
    The first row counts the initial entry from flat."""
    return position.diff().abs().fillna(position.abs()).mean(axis=1)


def summarize(port):
    port = port.dropna()
    cum = (1 + port).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, cum.values])[1:]     # starting equity is a peak
    return {
        "Ann Return": cum.iloc[-1] ** (252 / len(port)) - 1,
        "Ann Vol": port.std() * np.sqrt(252),
        "Sharpe": port.mean() / port.std() * np.sqrt(252),
        "Max DD": (cum.values / peak - 1).min(),
        "Total Ret": cum.iloc[-1] - 1,
    }


def run(daily, label, target_vol=TARGET_VOL, max_lev=MAX_LEVERAGE):
    port, position, costs = backtest(daily, HORIZONS, SKIP_MONTHS, target_vol, VOL_WINDOW, max_lev, COST_BPS)
    s = summarize(port)
    years = len(port) / 252
    s["Turnover/yr"] = portfolio_turnover(position).sum() / years
    s["Cost drag"] = costs.mean(axis=1).sum()
    s["Avg gross exp"] = position.abs().mean(axis=1).mean()
    return label, port, s


def print_table(rows):
    print(f"  {'':34s}{'AnnRet':>8}{'AnnVol':>8}{'Sharpe':>8}{'MaxDD':>8}"
          f"{'Turn/yr':>9}{'CostDrag':>10}{'GrossExp':>9}")
    for label, _, s in rows:
        print(f"  {label:34s}{s['Ann Return']:>8.2%}{s['Ann Vol']:>8.2%}{s['Sharpe']:>8.2f}"
              f"{s['Max DD']:>8.1%}{s['Turnover/yr']:>9.1f}{s['Cost drag']:>10.2%}"
              f"{s['Avg gross exp']:>9.2f}")


def main():
    daily = load_prices()

    rows = [
        run(daily, "Vol-targeted", target_vol=TARGET_VOL, max_lev=MAX_LEVERAGE),
        run(daily, "Equal-weight +/-1", target_vol=1.0, max_lev=1.0),
    ]

    first, last = rows[0][1].index[0].date(), rows[0][1].index[-1].date()
    print(f"Period: {first} to {last}  |  {len(TICKERS)} ETFs  |  cost {COST_BPS} bps")
    print("Portfolio stats (net of costs; Sharpe = mean/std*sqrt(252), no cash subtracted):")
    print_table(rows)

    port_main = rows[0][1]
    spy_ret = daily["SPY"].pct_change(fill_method=None).reindex(port_main.index).fillna(0)
    s = summarize(spy_ret)
    print(f"\n  SPY buy-and-hold, same period:     AnnRet {s['Ann Return']:.2%}  "
          f"AnnVol {s['Ann Vol']:.2%}  Sharpe {s['Sharpe']:.2f}  MaxDD {s['Max DD']:.1%}")
    print(f"  Beta of vol-targeted strategy to SPY: {np.polyfit(spy_ret, port_main, 1)[0]:.2f}")

    if PLOT:
        plt.figure(figsize=(11, 6))
        for label, port, _ in rows:
            plt.plot((1 + port).cumprod(), label=label)
        plt.plot((1 + spy_ret).cumprod(), label="SPY buy-and-hold", alpha=0.7, color="gray")
        plt.yscale("log")
        plt.title("TSMOM Portfolio: Cumulative Return")
        plt.ylabel("Growth of $1 (log)")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
