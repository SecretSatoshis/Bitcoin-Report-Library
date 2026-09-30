"""Published number precision (publication.py)."""

import unittest

import numpy as np
import pandas as pd

from publication import format_float, round_for_publication


class PublicationPrecisionTests(unittest.TestCase):
    def test_floats_are_written_to_ten_significant_digits(self):
        self.assertEqual(format_float(764.2000122070312), "764.2000122")
        self.assertEqual(format_float(160742.40000000002), "160742.4")
        self.assertEqual(format_float(0.06), "0.06")

    def test_large_values_keep_whole_units_without_scientific_notation(self):
        self.assertEqual(format_float(1678620519784.41), "1678620519784")
        self.assertEqual(format_float(-25496171191406.25), "-25496171191406")

    def test_values_beyond_the_whole_number_range_stay_numeric(self):
        text = format_float(1062501776542745755648.0)
        self.assertEqual(text, "1.062501777e+21")
        self.assertEqual(float(text), 1.062501777e21)

    def test_non_finite_values_are_blank(self):
        for value in (np.nan, np.inf, -np.inf):
            self.assertEqual(format_float(value), "")

    def test_rounded_frame_matches_the_published_text(self):
        frame = pd.DataFrame({"price": [83550.41, 1196.8822175737976, np.nan],
                              "flag": [True, False, True], "count": [1, 2, 3]})
        rounded = round_for_publication(frame)
        self.assertEqual(rounded.loc[1, "price"], 1196.882218)
        self.assertTrue(np.isnan(rounded.loc[2, "price"]))
        self.assertEqual(list(rounded["flag"]), [True, False, True])
        self.assertEqual(list(rounded["count"]), [1, 2, 3])
        for value in rounded["price"].dropna():
            self.assertEqual(float(format_float(value)), value)


if __name__ == "__main__":
    unittest.main()
