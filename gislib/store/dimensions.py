# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import math

Location = collections.namedtuple('Location', ('level', 'indices'))


class BaseDimension(object):
    
    M = 1  # Dimension multiplicity
    
    def __init__(self, size, **kwargs):
        """
        Handle dimension mapping to files and blocks

        Non-equidistant dimensions must have at least one aggregator.
        """
        self.size = size
        self.equidistant = kwargs.get('equidistant', True)
        self.aggregators = kwargs.get('aggregators', [])
        self.offset = kwargs.get('offset', 0)

    def get_level(self, resolution):
        """ Return chunk level. """
        return math.floor(math.log(self.size / resolution, 2))

    def get_extent(self, location):
        """ 
        Return extent tuple for a location.

        Location must have a resolution level and an iterable of indices.
        A format for a 2D-extent: ((x1, y1), (x2, y2))
        """
        return tuple(tuple(2 ** location.level * self.size * (i + j) 
                           for i in location.indices)
                     for j in (0, 1))

    def get_locations(self, extent, level):
        """ 
        Return a function that, when called, returns a generator of locations.

        Why a function? So reduce() can be used to iterate over all the
        locations of all the dimensions of the enclosing structure.

        If resolution does not exactly match, locations for the next
        matching higher resolutions are returned.
        """

        # Determine the chunk index ranges for this dimension
        levelsize = 2 ** level * self.size
        irange = map(
            xrange,
            (int(math.floor(e / levelsize)) for e in extent[0]),
            (int(math.ceil(e / levelsize)) for e in extent[1]),
        )
        
        # Prepare for reduce by creating a list of functions
        funcs = [lambda: ((i, ) for i in r) for r in irange]

        # Reduce by nesting the generators, combine with level and return again a function.
        def reducer(f1, f2):
            return lambda: (i + j for j in f2() for i in f1())

        # Combine the (sub)dimensions and return a function
        return lambda: (Location(level=level, indices=indices) 
                        for indices in reduce(reducer, funcs)())


class UnitDimension(BaseDimension):
    """
    Base for all dimension classes
    """
    def __init__(self, unit, *args, **kwargs):
        self.unit = unit
        super(UnitDimension, self).__init__(*args, **kwargs)
        

class TimeDimension(BaseDimension):
    """ 
    Dimension with coards calender units.
    """
    def __init__(self, calendar, *args, **kwargs):
        self.calendar = calendar
        super(TimeDimension, self).__init__(*args, **kwargs)


class SpatialDimension(BaseDimension):
    """
    Dimension with projection units, suitable for gdal reprojection.
    """
    M = 2  # Dimension multiplicity

    def __init__(self, projection, *args, **kwargs):
        self.projection = projection
        super(SpatialDimension, self).__init__(*args, **kwargs)
