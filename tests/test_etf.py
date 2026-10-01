"""US spot bitcoin ETF readers, table build and release step (etf/)."""
import io
import unittest
import zipfile
from unittest.mock import Mock, patch

import pandas as pd

import etf
from etf import build, funds
from etf.common import Snapshot, excel_date, iso_date, json_after, number, xlsx_rows


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


class ReleaseStepTests(unittest.TestCase):
    def test_a_failed_collection_never_stops_the_release(self):
        previous = {name: pd.DataFrame({"x": [1]}) for name in etf.ETF_FILES}
        contents = {name: frame.to_csv(index=False).encode() for name, frame in previous.items()}
        with patch.object(etf, "collect", side_effect=RuntimeError("down")):
            with patch.object(etf, "previous_release_files", return_value=contents):
                self.assertEqual(set(etf.get_etf_files("2026-09-30")), set(etf.ETF_FILES))
            with patch.object(etf, "previous_release_files", return_value={}):
                self.assertEqual(etf.get_etf_files("2026-09-30"), {})


if __name__ == "__main__":
    unittest.main()
