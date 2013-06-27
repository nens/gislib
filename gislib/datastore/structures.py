# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

"""
A structure defines the combination of dimensions and datacontent for
a datastore.

The dimensions and data are interweaved, since NED dimensions need a
dimension parameter in the basic datastructure.
"""

class Structure(object):
    def __init__(self, dimensions, chunkshape, dtype, nodatavalue):
        """
        For ned dimensions, need to add a dtype per ned dimension (or
        even multiple, consider the case of ned 3d space). Then the
        specified datatype must be prepended by the parameters into the
        ned interval.
        """
        self.nodatavalue = nodatavalue  # Also used to identify NED removals.
        self.dimensions = dimensions
        self.chunkshape = chunkshape
        self.dtype = dtype  # Any pickleable numpy dtype object will do.
