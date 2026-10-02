"""Build the published ETF tables from what etf.collect gathers.

etf_daily.csv, one row per fund and trading day, all on trade-date basis:
  btc_held          bitcoin the fund holds
  btc_source        how that figure was obtained (below)
  shares_outstanding, btc_per_share, nav
  net_assets_usd    shares x NAV where both are known, else btc_held x price_usd
  price_usd, price_basis   the 4pm bitcoin price used for the fund and where it comes from
  flow_btc          coins created minus coins redeemed that day
  flow_usd          the same flow in dollars (shares change x NAV, else flow_btc x price_usd)
  other_change_btc  any other change in holdings: sponsor fees paid in bitcoin and, in rows
                    fitted to SEC quarter ends, the fitting correction
  market_share_pct  share of all US spot ETF bitcoin that day

etf_totals_daily.csv, one row per trading day across all funds: total_btc, total_flow_btc,
total_flow_usd, cumulative_flow_btc, cumulative_flow_usd, total_net_assets_usd,
flow_weighted_entry_price (cumulative dollars in / cumulative coins in, the standard ETF
entry-price line), funds.

etf_quarterly.csv, one row per fund and SEC quarter end, plus an "ALL" row: btc, fair value,
reported cost, cost per coin, unrealized gain, and our daily figure for the same trade date.

btc_source:
  exact             the fund's published holdings on a collected day, or its filed quarter-end count
  issuer history    the fund's daily shares x bitcoin per share (a centred 9-day median of NAV /
                    its 4pm index, or for Grayscale the fee schedule), scaled to pass through its exact anchors
                    (snapshots, and SEC quarter ends where the fund reports its coins)
  rebuilt           fitted to each SEC quarter end before daily collection began
  interpolated      straight line between anchors (launch, SEC quarter ends, collected days)
  carried forward   after the fund's latest data, while its reader is failing

Funds without a published daily history keep the rows the previous release published, which
carry their history from before daily collection began; each run adds only the new days.
"""
import numpy as np
import pandas as pd

from . import funds

START = pd.Timestamp("2024-01-10")  # the funds' seed NAV day; trading began 2024-01-11
PER_SHARE_WINDOW = 9
# Trading days are the dates at least this many fund histories publish; a few issuers also
# publish on market holidays.
TRADING_DAY_QUORUM = 4
# Grayscale's published NAV is unreliable day to day (some days look misdated, off by up to 8%
# against the 4pm index), so its bitcoin per share is the published current value stepped back
# by the sponsor fee, accrued daily (annual rates from each trust's 10-Q).
SPONSOR_FEE = {"GBTC": 0.015, "BTC": 0.0015}
# GBTC moved 10% of its bitcoin to the Mini Trust; its first NAV without those coins.
GBTC_SPINOFF = pd.Timestamp("2024-07-30")


def _interpolate(dates: pd.DatetimeIndex, anchor_dates, anchor_values) -> np.ndarray:
    """Straight-line interpolation in time, constant beyond the first and last anchor."""
    return np.interp(dates.asi8, pd.DatetimeIndex(anchor_dates).asi8, np.asarray(anchor_values, dtype=float))


def _trade_date_snapshots(snapshots: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Snapshots on trade-date basis: a settlement-dated fund's figure belongs to the previous trading day."""
    frame = snapshots.copy()
    frame["date"] = pd.to_datetime(frame["as_of"])
    shift = frame["fund"].isin(funds.SETTLEMENT_DATED)
    position = calendar.searchsorted(frame.loc[shift, "date"]) - 1
    frame.loc[shift, "date"] = calendar[np.clip(position, 0, len(calendar) - 1)]
    return frame


def _split_factors(shares: pd.Series, nav: pd.Series) -> pd.Series:
    """Share splits: shares jump by a whole multiple while net assets barely move (HODL split
    4-for-1 on 2024-01-31). 1.0 on every other day."""
    jump = shares / shares.shift()
    split = (jump > 1.5) & ((jump * nav / nav.shift() - 1).abs() < 0.1)
    return jump.round().where(split, 1.0).fillna(1.0)


def _fee_schedule_per_share(fund, h, snapshots):
    """Bitcoin per share from the exact current value and the daily fee accrual.

    Before GBTC's spin-off the level is set by the median of NAV / 4pm index against the curve,
    measured relative to the same median after the spin-off, so bad NAV days do not move it.
    """
    exact = snapshots[snapshots.fund == fund].set_index("date")["btc_per_share"].dropna()
    if exact.empty:
        return None
    anchor = exact.index[-1]
    curve = pd.Series(exact.iloc[-1] * (1 - SPONSOR_FEE[fund] / 365) ** (h.index - anchor).days.to_numpy(), index=h.index)
    if fund == "GBTC":
        implied = h["nav"] / h["reference_price"] / curve
        before = h.index < GBTC_SPINOFF
        curve[before] *= implied[before].median() / implied[~before].median()
    return curve


def _issuer_fund(fund, history, snapshots, sec, calendar):
    # Trading days only, so "the next day" below is the next trading day, not a weekend row.
    h = history[(history.fund == fund) & history.date.isin(calendar)].sort_values("date").set_index("date")
    shares, nav = h["shares_outstanding"], h["nav"]
    split = _split_factors(shares, nav)
    if fund in funds.SETTLEMENT_DATED:
        # The next day's share count is this trade date's; the last day has no next day yet.
        # Shift in pre-split units so a split stays on the day NAV splits.
        units = shares / split.cumprod()
        shares = units.shift(-1).fillna(units) * split.cumprod()
    schedule = _fee_schedule_per_share(fund, h, snapshots) if fund in SPONSOR_FEE else None
    if schedule is not None:
        base = shares * schedule
    else:
        # NAV is net of liabilities that move day to day, and a few published days are wrong,
        # while bitcoin per share falls smoothly by the fee: take a centred 9-day median.
        per_share = (nav / h["reference_price"]).groupby((split > 1).cumsum()).transform(
            lambda part: part.rolling(PER_SHARE_WINDOW, center=True, min_periods=1).median())
        base = shares * per_share
    base = base.reindex(calendar).ffill()
    shares = shares.reindex(calendar).ffill()
    nav = nav.reindex(calendar).ffill()

    anchors = snapshots[snapshots.fund == fund].set_index("date")["btc_held"]
    reported = sec[(sec.fund == fund) & sec.reported_btc.notna()].set_index("trade_date")["reported_btc"]
    anchors = pd.concat([reported, anchors]).groupby(level=0).last().reindex(calendar).dropna()
    anchors = anchors[base.reindex(anchors.index) > 0]
    ratio = _interpolate(calendar, anchors.index, anchors / base[anchors.index]) if len(anchors) else 1.0
    btc = base * ratio
    source = pd.Series("issuer history", index=calendar).where(~calendar.isin(anchors.index), "exact")
    split = split.reindex(calendar).fillna(1.0)
    return pd.DataFrame({"btc_held": btc, "btc_source": source, "shares_outstanding": shares, "nav": nav,
                         "split_factor": split})


def _interpolated_fund(fund, nav_only, snapshots, sec, calendar, start=None):
    """Straight lines between the fund's exact anchors: launch (no coins), or `start` (date,
    coins), then each SEC quarter end and each collected snapshot."""
    nav = nav_only.get(fund, pd.Series(dtype=float))
    launch = nav.index.min() if len(nav) else START
    begin = pd.Series({start[0]: start[1]}) if start else pd.Series({launch: 0.0})
    quarter = sec[(sec.fund == fund) & (sec.btc > 0)].set_index("trade_date")["btc"]
    exact = snapshots[snapshots.fund == fund].set_index("date")["btc_held"]
    later = pd.concat([quarter, exact]).groupby(level=0).last()
    anchors = pd.concat([begin, later[later.index > begin.index[0]]])
    btc = pd.Series(_interpolate(calendar, anchors.index, anchors.to_numpy()), index=calendar)
    btc[calendar < launch] = np.nan
    filed = sec[(sec.fund == fund) & (sec.coin_source != "fair value / price")]["trade_date"]
    source = pd.Series("interpolated", index=calendar).where(~calendar.isin(exact.index.union(filed)), "exact")
    return pd.DataFrame({"btc_held": btc, "btc_source": source, "nav": nav.reindex(calendar)})


def _recover_estimated_coins(sec: pd.DataFrame, issuer: dict) -> pd.DataFrame:
    """Coins for quarter ends whose filing gives no count, from a better accounting price.

    The fallback is fair value / a price; here the price is the median of fair value / our
    daily coins across the funds with issuer history, which pins the price the filings used
    (including weekend quarter ends, where the 4pm index of that calendar day does not).
    """
    sec = sec.copy()
    implied = []
    for _, row in sec[sec.fund.isin(issuer) & (sec.fair_value_usd > 0)].iterrows():
        coins = issuer[row.fund]["btc_held"].get(row.trade_date)
        if coins and coins > 0:
            implied.append((row.quarter_end, row.fair_value_usd / coins))
    price = pd.DataFrame(implied, columns=["quarter_end", "price"]).groupby("quarter_end")["price"].median()
    estimated = (sec.coin_source == "fair value / price") & sec.quarter_end.isin(price.index)
    sec.loc[estimated, "price_usd"] = sec.loc[estimated, "quarter_end"].map(price)
    sec.loc[estimated, "price_source"] = "fair value / issuer-history coins"
    sec.loc[estimated, "btc"] = sec.loc[estimated, "fair_value_usd"] / sec.loc[estimated, "price_usd"]
    sec["cost_per_btc"] = sec["cost_usd"] / sec["btc"]
    return sec


def _calendar(histories: pd.DataFrame, snapshots: pd.DataFrame, through: pd.Timestamp) -> pd.DatetimeIndex:
    """Trading days: dates enough fund histories publish, then collected weekdays after them."""
    counts = histories.groupby("date")["fund"].nunique()
    days = counts.index[counts >= TRADING_DAY_QUORUM]
    later = pd.to_datetime(snapshots["as_of"])
    later = later[(later > days.max()) & (later.dt.dayofweek < 5)] if len(days) else later
    calendar = pd.DatetimeIndex(sorted(set(days) | set(later)))
    return calendar[(calendar >= START) & (calendar <= through)]


def _reference_prices(histories: pd.DataFrame) -> dict[str, pd.Series]:
    """The published 4pm index levels: MarketVector (HODL's file) and CME CF BRRNY (ARKB's)."""
    prices = {}
    for fund, index in (("HODL", funds.MARKETVECTOR), ("ARKB", funds.BRRNY)):
        rows = histories[(histories.fund == fund) & histories["index_price"].notna()]
        if len(rows):
            prices[index] = rows.set_index("date")["index_price"].sort_index()
    if not prices:
        raise RuntimeError("neither published 4pm index history is available")
    return prices


def build_tables(inputs: dict[str, pd.DataFrame], through: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """The three published tables from etf.collect's inputs: histories, snapshots, sec,
    nav_history and previous_daily (the last release's etf_daily.csv, or None)."""
    histories, sec, nav_history = inputs["histories"], inputs["sec"].copy(), inputs["nav_history"]
    previous = inputs.get("previous_daily")
    previous = None if previous is None or previous.empty else previous.assign(date=pd.to_datetime(previous["date"]))
    snapshots = inputs["snapshots"]
    through = pd.Timestamp(through)

    calendar = _calendar(histories, snapshots, through)
    if previous is not None:
        published = pd.DatetimeIndex(previous["date"].unique())
        calendar = calendar.union(published[published <= through])
    # A settlement-dated fund's file for the day after the cutoff holds the cutoff's trades.
    snapshots = _trade_date_snapshots(snapshots, calendar.union(pd.DatetimeIndex([through + pd.offsets.BDay(1)])))
    snapshots = snapshots[snapshots["date"] <= through]
    # A quarter end that falls on a weekend or holiday belongs to the last trading day before it.
    sec = sec[sec["quarter_end"] <= through]
    sec["trade_date"] = calendar[np.clip(calendar.searchsorted(sec["quarter_end"], side="right") - 1, 0, None)]
    nav_only = {fund: group.set_index("date")["nav"].sort_index() for fund, group in nav_history.groupby("fund")}

    prices = _reference_prices(histories)
    fallback = prices.get(funds.MARKETVECTOR, prices.get(funds.BRRNY)).reindex(calendar).ffill()
    index_prices = {index: series.reindex(calendar) for index, series in prices.items()}
    history = histories.copy()
    own = pd.Series([index_prices.get(funds.FUND_INDEX[f], pd.Series(dtype=float)).get(d, np.nan)
                     for f, d in zip(history.fund, history.date)], index=history.index)
    history["reference_price"] = history.get("index_price", own).fillna(own).fillna(history["date"].map(fallback))

    with_history = set(history["fund"])
    # Funds without a history this run keep the previous release's rows and continue from them.
    kept = {fund: rows.set_index("date").sort_index()
            for fund, rows in (previous.groupby("fund") if previous is not None else [])
            if fund not in with_history}
    issuer = {fund: _issuer_fund(fund, history, snapshots, sec, calendar)
              for fund in funds.CURRENT if fund in with_history}
    sec = _recover_estimated_coins(sec, issuer)

    frames = []
    for fund in funds.CURRENT:
        if fund in issuer:
            frame = issuer[fund]
        elif fund in kept:
            last = kept[fund].index.max()
            frame = _interpolated_fund(fund, nav_only, snapshots, sec, calendar,
                                       start=(last, kept[fund].at[last, "btc_held"]))
        else:
            frame = _interpolated_fund(fund, nav_only, snapshots, sec, calendar)
        current_only = fund not in issuer
        # After the fund's latest data, while its reader fails, the last value is carried forward.
        latest = [snapshots.loc[snapshots.fund == fund, "date"].max(),
                  sec.loc[(sec.fund == fund) & (sec.btc > 0), "trade_date"].max(),
                  history.loc[history.fund == fund, "date"].max()]
        if fund in kept:
            latest.append(kept[fund].index.max())
        latest = max((d for d in latest if pd.notna(d)), default=None)
        if latest is not None:
            frame.loc[frame.index > latest, "btc_source"] = "carried forward"
        own = index_prices.get(funds.FUND_INDEX[fund], pd.Series(np.nan, index=calendar))
        frame["price_usd"] = own.fillna(fallback)
        frame["price_basis"] = np.where(own.notna(), funds.FUND_INDEX[fund], "VanEck 4pm index (fallback)")
        frame = frame[frame["btc_held"].notna()].copy()
        exact = snapshots[snapshots.fund == fund].set_index("date")
        if "shares_outstanding" not in frame:
            frame["shares_outstanding"] = exact["shares_outstanding"].reindex(frame.index)
        frame["btc_per_share"] = frame["btc_held"] / frame["shares_outstanding"]
        if funds.FUND_INDEX[fund] == funds.BRRNY and current_only:
            # NAV / the fund's own index is its bitcoin per share, so shares follow from holdings
            # on days without a published count; only a NAV published for that day is used.
            same_day_nav = nav_only.get(fund, pd.Series(dtype=float)).reindex(frame.index)
            derived = frame["btc_held"] / (same_day_nav / frame["price_usd"])
            frame["shares_outstanding"] = frame["shares_outstanding"].fillna(derived)
            frame["btc_per_share"] = frame["btc_held"] / frame["shares_outstanding"]
        change = frame["btc_held"].diff()
        split = frame.pop("split_factor") if "split_factor" in frame else pd.Series(1.0, index=frame.index)
        created = frame["shares_outstanding"] - frame["shares_outstanding"].shift() * split
        # Creations and redemptions move shares; fees paid in bitcoin lower bitcoin per share.
        frame["flow_btc"] = created * frame["btc_per_share"]
        frame["flow_usd"] = created * frame["nav"]
        if current_only:
            # Exact where two consecutive days' share counts are known; otherwise the line's change.
            frame["flow_btc"] = frame["flow_btc"].fillna(change)
            frame["flow_usd"] = frame["flow_usd"].fillna(frame["flow_btc"] * frame["price_usd"])
        frame["other_change_btc"] = change - frame["flow_btc"]
        frame["net_assets_usd"] = (frame["shares_outstanding"] * frame["nav"]).fillna(
            frame["btc_held"] * frame["price_usd"])
        frame.insert(0, "fund", fund)
        frame = frame.rename_axis("date").reset_index()
        if fund in kept:
            old = kept[fund].reset_index()
            frame = pd.concat([old, frame[frame["date"] > old["date"].max()]], ignore_index=True)
        frames.append(frame)

    daily = pd.concat(frames, ignore_index=True)
    daily = daily[daily["date"] > START].copy()  # flows start with trading on 2024-01-11
    total = daily.groupby("date")["btc_held"].transform("sum")
    daily["market_share_pct"] = daily["btc_held"] / total * 100
    daily = daily[["fund", "date", "btc_held", "btc_source", "shares_outstanding", "btc_per_share", "nav",
                   "net_assets_usd", "price_usd", "price_basis", "flow_btc", "flow_usd", "other_change_btc",
                   "market_share_pct"]].sort_values(["date", "fund"]).reset_index(drop=True)

    totals = daily.groupby("date").agg(
        total_btc=("btc_held", "sum"), total_flow_btc=("flow_btc", "sum"), total_flow_usd=("flow_usd", "sum"),
        total_net_assets_usd=("net_assets_usd", "sum"), funds=("fund", "nunique"))
    totals["cumulative_flow_btc"] = totals["total_flow_btc"].cumsum()
    totals["cumulative_flow_usd"] = totals["total_flow_usd"].cumsum()
    totals["flow_weighted_entry_price"] = totals["cumulative_flow_usd"] / totals["cumulative_flow_btc"]
    totals = totals.reset_index()

    quarterly = sec[sec["btc"] > 0].copy()
    quarterly["unrealized_gain_usd"] = quarterly["fair_value_usd"] - quarterly["cost_usd"]
    ours = daily.set_index(["fund", "date"])["btc_held"]
    quarterly["btc_daily_table"] = [ours.get((f, d), np.nan) for f, d in zip(quarterly.fund, quarterly.trade_date)]
    quarterly["daily_vs_sec_pct"] = (quarterly["btc_daily_table"] / quarterly["btc"] - 1) * 100
    quarterly = quarterly[["fund", "quarter_end", "trade_date", "btc", "reported_btc", "coin_source", "price_usd",
                           "price_source", "fair_value_usd", "cost_usd", "cost_per_btc", "unrealized_gain_usd",
                           "btc_daily_table", "daily_vs_sec_pct"]]
    costed = quarterly.dropna(subset=["cost_usd"])
    combined = costed.groupby("quarter_end").agg(
        trade_date=("trade_date", "first"), btc=("btc", "sum"), fair_value_usd=("fair_value_usd", "sum"),
        cost_usd=("cost_usd", "sum"), unrealized_gain_usd=("unrealized_gain_usd", "sum"),
        funds_with_cost=("fund", "nunique")).reset_index()
    combined["cost_per_btc"] = combined["cost_usd"] / combined["btc"]
    combined["fund"] = "ALL"
    quarterly = pd.concat([quarterly, combined], ignore_index=True).sort_values(["quarter_end", "fund"])

    return {"etf_daily.csv": daily, "etf_totals_daily.csv": totals, "etf_quarterly.csv": quarterly}
