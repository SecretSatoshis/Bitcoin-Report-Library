"""US spot bitcoin ETF readers, table build and release step (etf/)."""
import importlib
import io
import unittest
import zipfile
from unittest.mock import Mock, patch

import pandas as pd

import etf
from etf import build, funds, sec as etf_sec
etf_collect = importlib.import_module("etf.collect")
from etf.common import (
    Snapshot, excel_date, iso_date, json_after, number,
    previous_us_trading_day, spreadsheet2003_rows, xlsx_rows,
)


def workbook(sheet_xml: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                         'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                         '<sheet name="Daily" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                         '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/sharedStrings.xml", '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                         '<si><t>Date</t></si><si><t>Shares Outstanding</t></si></sst>')
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return buffer.getvalue()


class EtfHelperTests(unittest.TestCase):
    def test_numbers_and_dates_in_published_formats(self):
        self.assertEqual(number("1,414,960,000.00"), 1414960000.0)
        self.assertEqual(number(" 10,436 "), 10436.0)
        self.assertEqual(number("$48.36"), 48.36)
        with self.assertRaises(ValueError):
            number("--")
        self.assertEqual(iso_date("Sep 30, 2026", "%b %d, %Y"), "2026-09-30")
        self.assertEqual(iso_date("30-Sep-26", "%d-%b-%y"), "2026-09-30")
        self.assertEqual(excel_date(46295.0), "2026-09-30")

    def test_embedded_page_data_is_decoded_after_its_key(self):
        page = 'x{"holdings":{"basket":[{"shares":1.5}],"asOfDate":"2026-09-30"},"sharesOutstanding":69890000}'
        self.assertEqual(json_after(page, "holdings")["asOfDate"], "2026-09-30")
        self.assertEqual(json_after(page, "sharesOutstanding"), 69890000)
        with self.assertRaises(ValueError):
            json_after(page, "missing")

    def test_xlsx_rows_reads_shared_strings_and_numbers(self):
        sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                 '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                 '<row r="2"><c r="A2"><v>46295</v></c><c r="C2"><v>164350100</v></c></row>'
                 '</sheetData></worksheet>')
        rows = xlsx_rows(workbook(sheet), "Daily")
        self.assertEqual(rows[0], ["Date", "Shares Outstanding"])
        self.assertEqual(rows[1], [46295.0, None, 164350100.0])

    def test_snapshot_requires_a_holding_and_derives_bitcoin_per_share(self):
        snapshot = Snapshot("TEST", "2026-09-30", 500.0, "fixture", shares_outstanding=1_000_000)
        self.assertAlmostEqual(snapshot.btc_per_share, 0.0005)
        with self.assertRaises(ValueError):
            Snapshot("TEST", "2026-09-30", 0, "fixture")

    def test_xlsx_rows_handles_optional_cell_ref_and_blocks_xxe(self):
        sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                 '<row r="1"><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>'
                 '<row r="2"><c><v>46295</v></c><c r="C2"><v>10</v></c></row>'
                 '</sheetData></worksheet>')
        rows = xlsx_rows(workbook(sheet), "Daily")
        self.assertEqual(rows[0], ["Date", "Shares Outstanding"])
        self.assertEqual(rows[1], [46295.0, None, 10.0])

        xxe_sheet = ('<!DOCTYPE foo [<!ENTITY xxe "bad">]>'
                     '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData/></worksheet>')
        with self.assertRaisesRegex(ValueError, "DOCTYPE|ENTITY"):
            xlsx_rows(workbook(xxe_sheet), "Daily")

    def test_spreadsheet2003_rows_honors_ss_index_and_blocks_xxe(self):
        xml = (
            '<?xml version="1.0"?>'
            '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
            'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
            '<Worksheet ss:Name="Historical"><Table>'
            '<Row><Cell><Data ss:Type="String">Date</Data></Cell>'
            '<Cell ss:Index="4"><Data ss:Type="Number">100</Data></Cell></Row>'
            '</Table></Worksheet></Workbook>'
        ).encode("utf-8")
        rows = spreadsheet2003_rows(xml, "Historical")
        self.assertEqual(rows, [["Date", "", "", "100"]])

        xxe = b'<!DOCTYPE a [<!ENTITY b "c">]><Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"/>'
        with self.assertRaisesRegex(ValueError, "DOCTYPE|ENTITY"):
            spreadsheet2003_rows(xxe, "Historical")

    def test_previous_us_trading_day_skips_weekends_and_nyse_holidays(self):
        # 2026-01-19 is MLK Day (Monday), so previous trading day before Tue 2026-01-20 is Fri 2026-01-16.
        self.assertEqual(previous_us_trading_day("2026-01-20"), pd.Timestamp("2026-01-16"))
        # 2026-07-03 is observed Independence Day (Friday), so previous before Mon 2026-07-06 is Thu 2026-07-02.
        self.assertEqual(previous_us_trading_day("2026-07-06"), pd.Timestamp("2026-07-02"))
        # 2026-04-03 is Good Friday, so previous before Mon 2026-04-06 is Thu 2026-04-02.
        self.assertEqual(previous_us_trading_day("2026-04-06"), pd.Timestamp("2026-04-02"))

    def test_fbtc_and_msbt_raise_explicit_value_errors_on_malformed_payloads(self):
        with patch.object(funds, "get", return_value=Mock(content=b"%PDF-1.4")), patch.object(
            funds.subprocess, "run", return_value=Mock(returncode=0, stdout="unexpected layout", stderr="")
        ):
            with self.assertRaisesRegex(ValueError, "FBTC PDF"):
                funds.fbtc_current()

        with patch.object(funds, "get", return_value=Mock(**{"json.return_value": {"en": {"holdings": []}}})):
            with self.assertRaisesRegex(ValueError, "MSBT"):
                funds.msbt_current()

    def test_sec_text_counts_are_scoped_to_filing_report_date(self):
        counts = {"2026-06-30": [50_000.0], "2026-09-30": [50_200.0]}
        self.assertEqual(etf_sec._candidate_text_counts(counts, "2026-09-30"), [50_200.0])
        self.assertEqual(etf_sec._candidate_text_counts(counts, "2026-06-30"), [50_000.0])



class HistoryReaderTests(unittest.TestCase):
    def test_arkb_history_keeps_the_fund_index_and_drops_incomplete_days(self):
        payload = {"data": [
            {"valuation_date": "2026-09-30", "nav_per_share": 27.75, "total_units_outstanding": 104310000,
             "total_nav": 2895043018.52, "index": 83725.93},
            {"valuation_date": "2024-01-10", "nav_per_share": 15.283, "total_units_outstanding": 675000,
             "total_nav": 10316848.5, "index": None},
        ]}
        with patch.object(funds, "get", return_value=Mock(**{"json.return_value": payload})):
            frame = funds.arkb_history()
        self.assertEqual(frame["date"].tolist(), ["2026-09-30"])
        self.assertAlmostEqual(frame["net_assets"].iloc[0] / frame["index_price"].iloc[0], 34577.6, places=1)

    def test_btcw_history_is_nav_times_shares(self):
        page = {"navHistory": [
            {"dt": "2024-01-09T00:00:00.000Z", "nav": 46.5, "sharesOutstanding": 100000},
            {"dt": "2024-01-08T00:00:00.000Z", "nav": 49.888, "sharesOutstanding": 50000},
            {"dt": "2024-01-05T00:00:00.000Z", "nav": None, "sharesOutstanding": None},
        ]}
        with patch.object(funds, "_btcw_page", return_value=page):
            frame = funds.btcw_history()
        self.assertEqual(frame["date"].tolist(), ["2024-01-08", "2024-01-09"])
        self.assertEqual(frame["net_assets"].tolist(), [2494400.0, 4650000.0])



class BuildTests(unittest.TestCase):
    def test_share_splits_are_found(self):
        index = pd.date_range("2024-01-29", periods=4, freq="B")
        shares = pd.Series([100.0, 100.0, 400.0, 410.0], index=index)
        nav = pd.Series([48.0, 49.0, 12.3, 12.2], index=index)
        self.assertEqual(build._split_factors(shares, nav).tolist(), [1.0, 1.0, 4.0, 1.0])

    def test_interpolation_meets_anchors_and_holds_beyond_them(self):
        index = pd.date_range("2024-01-01", periods=5, freq="D")
        values = build._interpolate(index, [index[1], index[3]], [1.0, 3.0])
        self.assertEqual(values.tolist(), [1.0, 1.0, 2.0, 3.0, 3.0])

    def test_running_average_cost_basis_never_rises_on_redemptions_below_cost(self):
        # Buy 100 BTC @ $100k -> avg $100k; buy 100 BTC @ $80k -> avg $90k;
        # redeem 100 BTC @ $60k (below cost) -> avg stays $90k (old cumulative ratio rose to $120k!).
        flow_btc = pd.Series([100.0, 100.0, -100.0, -100.0])
        flow_usd = pd.Series([10_000_000.0, 8_000_000.0, -6_000_000.0, -6_000_000.0])
        entry = build._running_average_cost_basis(flow_btc, flow_usd)
        self.assertAlmostEqual(entry.iloc[0], 100_000.0)
        self.assertAlmostEqual(entry.iloc[1], 90_000.0)
        self.assertAlmostEqual(entry.iloc[2], 90_000.0)
        self.assertTrue(pd.isna(entry.iloc[3]))

    def test_ibit_trade_date_holdings_align_with_t_plus_1_shares_and_sec_takes_precedence(self):
        days = pd.bdate_range("2026-09-29", "2026-10-02")
        # IBIT shares in history are T+1: 100 on 9/29, 100 on 9/30 (trade 9/29),
        # 120 on 10/01 (trade 9/30 -> +20 shares created on 9/30), 110 on 10/02 (trade 10/01 -> -10 redeemed).
        histories = pd.DataFrame([
            {"fund": "IBIT", "date": days[0], "shares_outstanding": 100.0, "nav": 50.0, "net_assets": 5000.0, "index_price": float("nan")},
            {"fund": "IBIT", "date": days[1], "shares_outstanding": 100.0, "nav": 50.0, "net_assets": 5000.0, "index_price": float("nan")},
            {"fund": "IBIT", "date": days[2], "shares_outstanding": 120.0, "nav": 50.0, "net_assets": 6000.0, "index_price": float("nan")},
            {"fund": "IBIT", "date": days[3], "shares_outstanding": 110.0, "nav": 50.0, "net_assets": 5500.0, "index_price": float("nan")},
            *[
                {"fund": f, "date": d, "shares_outstanding": 100.0, "nav": 50.0, "net_assets": 5000.0,
                 "index_price": 100_000.0 if f in ("HODL", "ARKB") else float("nan")}
                for f in ("GBTC", "HODL", "ARKB") for d in days
            ],
        ])
        # IBIT snapshots are already trade-dated (0.0005 BTC/share -> 0.05, 0.06, 0.055, 0.052).
        # On 2026-09-30 scraped snapshot says 0.061, while audited SEC says 0.060 (SEC must win!).
        snapshots = pd.DataFrame([
            {"fund": "IBIT", "as_of": "2026-09-29", "btc_held": 0.050, "shares_outstanding": 100.0, "btc_per_share": 0.0005},
            {"fund": "IBIT", "as_of": "2026-09-30", "btc_held": 0.061, "shares_outstanding": 100.0, "btc_per_share": 0.0005},
            {"fund": "IBIT", "as_of": "2026-10-01", "btc_held": 0.055, "shares_outstanding": 120.0, "btc_per_share": 0.0005},
            {"fund": "IBIT", "as_of": "2026-10-02", "btc_held": 0.052, "shares_outstanding": 110.0, "btc_per_share": 0.0005},
        ])
        sec = pd.DataFrame([{
            "fund": "IBIT", "quarter_end": pd.Timestamp("2026-09-30"), "fair_value_usd": 6000.0,
            "cost_usd": 5000.0, "reported_btc": 0.060, "price_usd": 100_000.0,
            "price_source": "test", "coin_source": "tagged", "btc": 0.060, "cost_per_btc": 83333.33,
        }])
        tables = build.build_tables({"histories": histories, "snapshots": snapshots, "sec": sec,
                                     "nav_history": pd.DataFrame(columns=["fund", "date", "nav"]),
                                     "previous_daily": None}, days[-1])
        ibit = tables["etf_daily.csv"].query("fund == 'IBIT'").set_index("date")
        # Audited SEC 10-Q wins on 2026-09-30
        self.assertAlmostEqual(ibit.loc[pd.Timestamp("2026-09-30"), "btc_held"], 0.060)
        # Positive creation on 2026-09-30 aligns with +0.010 BTC holding increase (zero phantom swing!)
        self.assertAlmostEqual(ibit.loc[pd.Timestamp("2026-09-30"), "flow_btc"], 0.010, places=5)
        self.assertAlmostEqual(ibit.loc[pd.Timestamp("2026-09-30"), "other_change_btc"], 0.0, places=5)
        # Negative redemption on 2026-10-01 aligns with -0.005 BTC holding decrease
        self.assertAlmostEqual(ibit.loc[pd.Timestamp("2026-10-01"), "flow_btc"], -0.005, places=5)
        self.assertAlmostEqual(ibit.loc[pd.Timestamp("2026-10-01"), "other_change_btc"], 0.0, places=5)
        # Trailing day 2026-10-02 has exact snapshot (0.052) but T+1 shares aren't known yet:
        # flow_btc derives from btc_held.diff() (-0.003) and net_assets_usd = btc_held * price_usd (5200)
        self.assertAlmostEqual(ibit.loc[pd.Timestamp("2026-10-02"), "flow_btc"], -0.003, places=5)
        self.assertAlmostEqual(ibit.loc[pd.Timestamp("2026-10-02"), "net_assets_usd"], 5200.0, places=2)

    def test_missing_history_day_does_not_create_phantom_swing_from_stale_shares(self):
        days = pd.bdate_range("2026-11-10", "2026-11-13")
        # ARKB omits 2026-11-12 from history, while snapshots publish exact holdings every day.
        histories = pd.DataFrame([
            {"fund": "ARKB", "date": days[0], "shares_outstanding": 100.0, "nav": 50.0, "net_assets": 5000.0, "index_price": 100_000.0},
            {"fund": "ARKB", "date": days[1], "shares_outstanding": 100.0, "nav": 50.0, "net_assets": 5000.0, "index_price": 100_000.0},
            # days[2] missing in ARKB history!
            {"fund": "ARKB", "date": days[3], "shares_outstanding": 90.0, "nav": 45.0, "net_assets": 4050.0, "index_price": 90_000.0},
            *[
                {"fund": f, "date": d, "shares_outstanding": 100.0, "nav": 50.0, "net_assets": 5000.0,
                 "index_price": 100_000.0 if f == "HODL" else float("nan")}
                for f in ("IBIT", "GBTC", "HODL", "EZBC") for d in days
            ],
        ])
        snapshots = pd.DataFrame([
            {"fund": "ARKB", "as_of": "2026-11-11", "btc_held": 50.0, "shares_outstanding": float("nan"), "btc_per_share": float("nan")},
            {"fund": "ARKB", "as_of": "2026-11-12", "btc_held": 45.0, "shares_outstanding": float("nan"), "btc_per_share": float("nan")},
            {"fund": "ARKB", "as_of": "2026-11-13", "btc_held": 45.0, "shares_outstanding": float("nan"), "btc_per_share": float("nan")},
        ])
        sec = pd.DataFrame(columns=["fund", "quarter_end", "fair_value_usd", "cost_usd", "reported_btc",
                                    "price_usd", "price_source", "coin_source", "btc", "cost_per_btc"])
        sec["quarter_end"] = pd.to_datetime(sec["quarter_end"])
        tables = build.build_tables({"histories": histories, "snapshots": snapshots, "sec": sec,
                                     "nav_history": pd.DataFrame(columns=["fund", "date", "nav"]),
                                     "previous_daily": None}, days[-1])
        arkb = tables["etf_daily.csv"].query("fund == 'ARKB'").set_index("date")
        self.assertAlmostEqual(arkb.loc[days[2], "flow_btc"], -5.0, places=5)
        self.assertAlmostEqual(arkb.loc[days[2], "other_change_btc"], 0.0, places=5)
        self.assertAlmostEqual(arkb.loc[days[3], "flow_btc"], 0.0, places=5)
        self.assertAlmostEqual(arkb.loc[days[3], "other_change_btc"], 0.0, places=5)

    def test_etf_health_warns_when_top5_fund_stale_over_5_trading_days(self):
        days = pd.bdate_range("2026-10-01", periods=8)
        rows = []
        for fund in funds.CURRENT:
            for i, d in enumerate(days):
                src = "exact" if (fund != "FBTC" or i == 0) else "daily flows"
                rows.append({"fund": fund, "date": d, "btc_held": 100.0, "btc_source": src})
        daily = pd.DataFrame(rows)
        snaps = pd.DataFrame([{"fund": "FBTC", "as_of": "2026-10-01", "btc_held": 100.0}])
        with self.assertWarnsRegex(RuntimeWarning, r"top-5 fund FBTC has been stale for 7 trading days"):
            health = etf_collect._build_health(days[-1], daily, snaps, {"funds": {"FBTC": {"current": "ValueError: bad"}}})
        self.assertEqual(health["funds"]["FBTC"]["consecutive_stale_days"], 7)
        self.assertEqual(health["funds"]["FBTC"]["status"], "error")
        self.assertEqual(len(health["top5_stale"]), 1)


class CarriedHistoryTests(unittest.TestCase):
    def test_published_rows_of_funds_without_history_are_kept(self):
        """A run on the next day keeps the last release's rows and adds the new day."""
        days = pd.bdate_range("2026-09-28", "2026-10-01")
        histories = pd.DataFrame([
            {"fund": fund, "date": day, "shares_outstanding": 100.0, "nav": 50.0, "net_assets": 5000.0,
             "index_price": 100_000.0 if fund in ("HODL", "ARKB") else float("nan")}
            for fund in ("IBIT", "GBTC", "HODL", "ARKB") for day in days])
        previous = pd.DataFrame({"fund": "FBTC", "date": days[:3].strftime("%Y-%m-%d"), "btc_held": [10.0, 11.0, 12.0],
                                 "btc_source": "rebuilt", "flow_btc": [0.0, 1.0, 1.0]})
        snapshots = pd.DataFrame({"fund": ["FBTC"], "as_of": ["2026-10-01"], "btc_held": [13.0],
                                  "shares_outstanding": [1000.0], "btc_per_share": [0.013]})
        sec = pd.DataFrame(columns=["fund", "quarter_end", "fair_value_usd", "cost_usd", "reported_btc",
                                    "price_usd", "price_source", "coin_source", "btc", "cost_per_btc"])
        sec["quarter_end"] = pd.to_datetime(sec["quarter_end"])
        tables = build.build_tables({"histories": histories, "snapshots": snapshots, "sec": sec,
                                     "nav_history": pd.DataFrame(columns=["fund", "date", "nav"]),
                                     "previous_daily": previous}, pd.Timestamp("2026-10-01"))
        fbtc = tables["etf_daily.csv"].query("fund == 'FBTC'").set_index("date")["btc_held"]
        self.assertEqual(fbtc.tolist(), [10.0, 11.0, 12.0, 13.0])


class ProvisionalFlowTests(unittest.TestCase):
    DAYS = pd.bdate_range("2026-09-28", "2026-10-01")

    def tables(self, provisional, previous=None, snapshots=None, ibit_shares=(100.0, 100.0, 100.0, 100.0)):
        histories = pd.DataFrame([
            {"fund": fund, "date": day, "nav": 50.0,
             "shares_outstanding": ibit_shares[i] if fund == "IBIT" else 100.0,
             "index_price": 100_000.0 if fund in ("HODL", "ARKB") else float("nan")}
            for fund in ("IBIT", "GBTC", "HODL", "ARKB") for i, day in enumerate(self.DAYS)])
        histories["net_assets"] = histories["nav"] * histories["shares_outstanding"]
        sec = pd.DataFrame(columns=["fund", "quarter_end", "fair_value_usd", "cost_usd", "reported_btc",
                                    "price_usd", "price_source", "coin_source", "btc", "cost_per_btc"])
        sec["quarter_end"] = pd.to_datetime(sec["quarter_end"])
        if snapshots is None:
            snapshots = pd.DataFrame({"fund": pd.Series(dtype=str), "as_of": pd.Series(dtype=str),
                                      **{c: pd.Series(dtype=float)
                                         for c in ("btc_held", "shares_outstanding", "btc_per_share")}})
        return build.build_tables({"histories": histories, "snapshots": snapshots, "sec": sec,
                                   "nav_history": pd.DataFrame(columns=["fund", "date", "nav"]),
                                   "previous_daily": previous, "provisional": provisional},
                                  self.DAYS[-1])["etf_daily.csv"]

    def test_settlement_dated_last_day_takes_the_provisional_flow(self):
        """IBIT's last share count is the previous trade day's, so its last day's flow is provisional."""
        provisional = pd.DataFrame({"fund": ["IBIT"], "date": [self.DAYS[-1]], "flow_usd": [1_000.0]})
        daily = self.tables(provisional).set_index(["fund", "date"])
        last, before = ("IBIT", self.DAYS[-1]), ("IBIT", self.DAYS[-2])
        self.assertEqual(daily.at[last, "btc_source"], "provisional")
        self.assertEqual(daily.at[last, "flow_usd"], 1_000.0)
        self.assertAlmostEqual(daily.at[last, "btc_held"], daily.at[before, "btc_held"] + 0.01)
        # Without a provisional figure the day is carried forward, not passed off as issuer data.
        self.assertEqual(daily.at[("GBTC", self.DAYS[-1]), "btc_source"], "carried forward")
        self.assertEqual(daily.at[("ARKB", self.DAYS[-1]), "btc_source"], "issuer history")

    def test_unpublished_days_are_rebuilt_from_later_snapshots_then_provisional(self):
        previous = pd.DataFrame({"fund": "FBTC", "date": self.DAYS[:3].strftime("%Y-%m-%d"),
                                 "btc_held": [10.0, 10.0, 10.0],
                                 "btc_source": ["rebuilt", "carried forward", "carried forward"],
                                 "flow_btc": 0.0, "price_usd": 100_000.0})
        snapshots = pd.DataFrame({"fund": ["FBTC"], "as_of": [self.DAYS[2].strftime("%Y-%m-%d")], "btc_held": [12.0],
                                  "shares_outstanding": [1000.0], "btc_per_share": [0.012]})
        provisional = pd.DataFrame({"fund": ["FBTC"], "date": [self.DAYS[-1]], "flow_usd": [100_000.0]})
        fbtc = self.tables(provisional, previous, snapshots).query("fund == 'FBTC'").set_index("date")
        self.assertEqual(fbtc["btc_held"].tolist(), [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(fbtc["btc_source"].tolist(), ["rebuilt", "interpolated", "exact", "provisional"])

    def test_a_failing_readers_days_are_filled_as_daily_flows_and_kept(self):
        """Past STALE_AFTER_DAYS without its own data, a fund's filled days are its data, not provisional."""
        previous = pd.DataFrame({"fund": "FBTC", "date": self.DAYS[:1].strftime("%Y-%m-%d"), "btc_held": [10.0],
                                 "btc_source": ["rebuilt"], "flow_btc": 0.0, "price_usd": 100_000.0})
        provisional = pd.DataFrame({"fund": "FBTC", "date": self.DAYS[1:], "flow_usd": 100_000.0})
        with self.assertWarnsRegex(RuntimeWarning, r"ETF FBTC: no issuer data since 2026-09-28 \(3 trading days\)"):
            daily = self.tables(provisional, previous)
        fbtc = daily.query("fund == 'FBTC'").set_index("date")
        self.assertEqual(fbtc["btc_held"].tolist(), [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(fbtc["btc_source"].tolist(), ["rebuilt"] + [build.FILLED] * 3)

        # The next run keeps those days even when the provisional dataset no longer lists them.
        with self.assertWarns(RuntimeWarning):
            again = self.tables(provisional.iloc[:0], daily).query("fund == 'FBTC'").set_index("date")
        self.assertEqual(again["btc_held"].tolist(), [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(again["btc_source"].tolist(), ["rebuilt"] + [build.FILLED] * 3)

    def test_issuer_flow_the_provisional_data_contradicts_is_reported(self):
        provisional = pd.DataFrame({"fund": ["IBIT"], "date": [self.DAYS[2]], "flow_usd": [5_000_000.0]})
        with self.assertWarnsRegex(RuntimeWarning, r"ETF IBIT 2026-09-30: issuer flow \+0\.0M but provisional \+5\.0M"):
            self.tables(provisional)


class ReleaseStepTests(unittest.TestCase):
    def test_a_failed_collection_never_stops_the_release(self):
        previous = {name: pd.DataFrame({"x": [1]}) for name in etf.ETF_FILES}
        contents = {name: frame.to_csv(index=False).encode() for name, frame in previous.items()}
        with patch.object(etf, "collect", side_effect=RuntimeError("down")):
            with patch.object(etf, "previous_release_files", return_value=contents):
                self.assertEqual(set(etf.get_etf_files("2026-09-30")), set(etf.ETF_FILES))
            with patch.object(etf, "previous_release_files", return_value={}):
                self.assertEqual(etf.get_etf_files("2026-09-30"), {})

    def test_bitb_parses_nextjs_rsc_escaped_payload(self):
        rsc_html = (
            '<script>self.__next_f.push([1,"'
            '\\"holdings\\":{\\"asOfDate\\":\\"2026-11-23\\",\\"basket\\":[{\\"name\\":\\"BITCOIN\\",'
            '\\"marketValue\\":2188366584.14,\\"assetInTrust\\":35786.8557,\\"assetPerShare\\":0.0005424716644}]},'
            '\\"holdings\\":\\"$8f:props:holdings\\",'
            '\\"sharesOutstanding\\":66570000,'
            '\\"navMarketPrice\\":{\\"asOfDate\\":\\"2026-11-20\\",\\"nav\\":{\\"value\\":31.68},'
            '\\"chart\\":{\\"nav\\":[{\\"t\\":1704844800000,\\"v\\":25.0},{\\"t\\":1795132800000,\\"v\\":31.68}]}},'
            '\\"walletBalances\\":[{\\"active\\":true,\\"address\\":\\"bc1qtest\\",\\"balance\\":700.5}]'
            '"])</script>'
        )
        with patch.object(funds, "_bitb_page", return_value=rsc_html):
            snap, wallets = funds.bitb_current()
            nav = funds.bitb_nav_history()
        self.assertEqual(snap.as_of, "2026-11-23")
        self.assertAlmostEqual(snap.btc_held, 35786.8557, places=4)
        self.assertEqual(snap.shares_outstanding, 65970000.0)
        self.assertEqual(len(wallets), 1)
        self.assertEqual(list(nav["date"]), ["2024-01-10", "2026-11-20"])
        self.assertEqual(list(nav["nav"]), [25.0, 31.68])


if __name__ == "__main__":
    unittest.main()
