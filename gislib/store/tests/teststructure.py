#!/usr/bin/python
# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import unittest

import numpy as np

from gislib.datastore import dimensions
from gislib.datastore import structures



class TestStructure(unittest.TestCase):
    """ Integration tests. """
    def setUp(self):
        self.structure = structures.Structure(
            dimensions=[
                dimensions.SpatialDimension(projection=28992, size=256),
                dimensions.TimeDimension(calendar='minutes since 200130401', size=1),
            ],
            dtype='f4',
            nodatavalue=np.finfo('f4').min,
        )

    def test_extent(self):
        location = (
            dimensions.Location(level=1, indices=(1, 1)),
            dimensions.Location(level=1, indices=(1,)),
        )
        expected_extent = (
            ((512, 512), (1024, 1024)),
            ((2,), (4,)),
        )
        computed_extent = self.structure.get_extent(location)
        self.assertEqual(computed_extent, expected_extent)
        computed_locations = list(self.structure.get_locations(
            computed_extent, (128, 0.5)
        ))
        self.assertEqual(len(computed_locations), 1)
        self.assertEqual(computed_locations[0], location)


        #chunk1 = chunks.Chunk(storage=self.storage, location='onzin')
        #chunk1['data'] = 'de data'
        #chunk2 = chunks.Chunk(storage=self.storage, location='aaaaaa')
        #chunk2['data'] = 'de data2'
        #chunk = chunks.Chunk.first(storage=self.storage)
        #print(chunk['data'])
        #chunk1['data'] = None
        #chunk2['data'] = None
