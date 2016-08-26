from unittest import TestCase

from gislib import vectors
import numpy as np


class TestBbox(TestCase):

    def setUp(self):
        self.pnts = np.array(
            [[1, 14], [5, 13], [0, 11], [3.5, 10]]
        )

    def test_get_bbox(self):
        expected = [[0, 10], [5, 14]]
        res = vectors.get_bbox(self.pnts)
        np.testing.assert_equal(res, expected)

    def test_get_extent(self):
        expected = [[0, 10], [7, 16]]
        second_set = np.array(
            [[7, 15], [3, 16], [3, 13], [4, 11]]
        )
        res = vectors.get_extent((self.pnts, second_set))
        np.testing.assert_equal(res, expected)
