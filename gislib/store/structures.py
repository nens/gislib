# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import numpy as np

from gislib.store import dimensions

"""
A structure defines the combination of dimensions and datacontent for
a datastore.

The dimensions and data are interweaved, since NED dimensions need a
dimension parameter in the basic datastructure.
"""

class Structure(object):
    def __init__(self, dimensions, dtype, nodatavalue):
        """
        For ned dimensions, need to add a dtype per ned dimension (or
        even multiple, consider the case of ned 3d space). Then the
        specified datatype must be prepended by the parameters into the
        ned interval.
        """
        self.nodatavalue = nodatavalue  # Also used to identify NED removals.
        self.dimensions = dimensions
        self.dtype = dtype  # Any pickleable numpy dtype object will do.
        self.loc_size = 8 * sum(1 + d.M for d in self.dimensions)

    def __len__(self):
        """ Return the length of the location data. """
        return 

    def dumps(self, location):
        """ Return a serialized location string. """
        return np.int64([j for l, i in location for j in (l,) + i]).tostring()

    def loads(self, location):
        """ Load location from blobstring. """
        v = iter(np.fromstring(location[:self.locsize], dtype=np.int64))
        return tuple(dimensions.Location(level=v.next(),                                   
                                         indices=tuple(v.next() 
                                                       for n in range(d.M)))
                     for d in self.dimensions)                           

    def get_locations(self, extent, resolution):
        """
        Return a generator of location tuples.
        """
        # Coerce resolutions to levels
        level = tuple(d.get_level(r)
                      for d, r in zip(self.dimensions, resolution))

        # Get the funcs that return the per-dimension location generators
        funcs = (d.get_locations(e, l)
                 for d, e, l in zip(self.dimensions, extent, level))

        # Nest via reduce function.
        def reducer(f1, f2):
            return lambda: (tuple([i]) + tuple([j]) 
                            for j in f2() for i in f1())

        return reduce(reducer, funcs)()


    def get_extent(self, location):
        """ Return the extent of a chunk at location. """
        return tuple(dimension.get_extent(level)
                     for dimension, level in zip(self.dimensions, location))

    def get_parent_location(self, location, dimension, levels):
        """ Return a location. """
        level = location[dimension].level + levels
        extent = self.get_extent(location)[dimension]
        insert = self.dimensions[dimension].get_locations(
            extent=extent, level=level,
        )().next(),
        return location[:dimension] + insert + location[dimension + 1:]


    def get_child_locations(self, location, dimension):
        """ Return a location generator. """
        level = location[dimension].level - 1
        extent = self.get_extent(location)[dimension]
        inserts = self.dimensions[dimension].get_locations(
            extent=extent, level=level,
        )()
        for insert in inserts:
            yield location[:dimension] + (insert, ) + location[dimension + 1:]
