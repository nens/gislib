# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

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

    def get_locations(self, extent, resolution):
        """
        Return a generator of location tuples.
        """
        def reducer(f1, f2):
            return lambda: (tuple([i]) + tuple([j]) 
                            for j in f2() for i in f1())
        
        funcs = (d.get_locations(e, r)
                 for d, e, r in zip(self.dimensions, extent, resolution))

        return reduce(reducer, funcs)()


    def get_extent(self, location):
        """
        Return the extent of a chunk at location
        """
        return tuple(dimension.get_extent(level)
                     for dimension, level in zip(self.dimensions, location))
