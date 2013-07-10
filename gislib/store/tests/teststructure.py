# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import unittest

import numpy as np

from gislib.store import structures


class TestStructure(unittest.TestCase):
    """ Integration tests. """
    def setUp(self):
        self.frame = structures.Frame(
            dimensions=[
                structures.SpatialDimension(projection=28992, size=256),
                structures.TimeDimension(size=1,
                                         calendar='minutes since 200130401'),
            ],
            dtype='f4',
            nodatavalue=np.finfo('f4').min,
        )

    def test_extent(self):
        location = structures.Location(frame=self.frame, sublocations=(
            structures.Sublocation(level=1, indices=(1, 1)),
            structures.Sublocation(level=1, indices=(1,)),
        ))
        expected_extent = (
            ((512, 512), (1024, 1024)),
            ((2,), (4,)),
        )
        computed_extent = location.get_extent()
        self.assertEqual(computed_extent, expected_extent)
        computed_locations = list(self.frame.get_locations(
            computed_extent, (128, 0.5)
        ))
        self.assertEqual(len(computed_locations), 1)
        self.assertEqual(computed_locations[0], location)
