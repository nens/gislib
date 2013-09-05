# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import unittest

import numpy as np

from gislib.store import core


class TestStructure(unittest.TestCase):
    """ Integration tests. """
    def setUp(self):
        spatial_scale = core.SpatialScale(projection=28992, size=(256, 256))
        time_scale = core.TimeScale(size=(1,),
                                    calendar='minutes since 20130401')
        scales = [core.FrameScale(spatial_scale), core.FrameScale(time_scale)]
        metric = core.FrameMetric(scales=scales)
        self.frame = core.Frame(metric=metric,
                                dtype='f4',
                                nodatavalue=np.finfo('f4').min)

    def test_extent(self):
        location = core.Location(parts=(
            core.Sublocation(level=1, indices=(1, 1)),
            core.Sublocation(level=1, indices=(1,)),
        ))
        expected_extent = (
            ((512, 512), (1024, 1024)),
            ((2,), (4,)),
        )
        computed_extent = self.frame.metric.get_extent(location)
        self.assertEqual(computed_extent, expected_extent)
        computed_locations = list(self.frame.get_locations(
            computed_extent, size=((256, 256), (1,))
        ))
        self.assertEqual(len(computed_locations), 1)
        self.assertEqual(computed_locations[0], location)
