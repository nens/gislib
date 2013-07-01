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
    def __init__(self, size, **kwargs):
        """
        Handle dimension mapping to files and blocks

        Non-equidistant dimensions must have at least one aggregator.
        """
        self.size = size
        self.equidistant = kwargs.get('equidistant', True)
        self.aggregators = kwargs.get('aggregators', [])
        self.offset = kwargs.get('offset', 0)

    def get_extent(self, location):
        """ 
        Return extent tuple for a location.

        Location must have a resolution level and an iterable of indices.
        A format for a 2D-extent: ((x1, y1), (x2, y2))
        """
        return tuple(tuple(2 ** location.level * self.size * (i + j) 
                           for i in location.indices)
                     for j in (0, 1))

    def get_locations(self, extent, resolution):
        """ 
        Return a function that, when called, returns a generator of locations.

        Why a function? So reduce() can be used to iterate over all the
        locations of all the dimensions of the enclosing structure.

        If resolution does not exactly match, locations for the next
        matching higher resolutions are returned.
        """
        # Determine level for resolution using size
        level = math.floor(math.log(self.size / resolution, 2))
        
        # Determine range of indices 
        def _index(tile):
            """ Return the second xrange argument. """
            ceil = math.ceil(tile)
            return int(ceil)
            return int(ceil) if ceil == tile else int(ceil)

        levelsize = 2 ** level * self.size
        irange = map(
            xrange,
            (int(math.floor(e / levelsize)) for e in extent[0]),
            (_index(e / levelsize) for e in extent[1]),
        )
        
        # Make a list of functions that return generators of indices
        funcs = [lambda: ((i, ) for i in r) for r in irange]

        # Reduce by nesting the generators, combine with level and return again a function.
        def reducer(f1, f2):
            return lambda: (i + j for j in f2() for i in f1())
        return lambda: (Location(level=level, indices=indices) 
                        for indices in reduce(reducer, funcs)())


class UnitDimension(BaseDimension):
    """
    Base for all dimension classes
    """
    DIMENSIONS = 1

    def __init__(self, unit, *args, **kwargs):
        self.unit = unit
        super(UnitDimension, self).__init__(*args, **kwargs)
        

class TimeDimension(BaseDimension):
    """ 
    Dimension with coards calender units.
    """
    DIMENSIONS = 1

    def __init__(self, calendar, *args, **kwargs):
        self.calendar = calendar
        super(TimeDimension, self).__init__(*args, **kwargs)


class SpatialDimension(BaseDimension):
    """
    Dimension with projection units, suitable for gdal reprojection.
    """
    DIMENSIONS = 2

    def __init__(self, projection, *args, **kwargs):
        self.projection = projection
        super(SpatialDimension, self).__init__(*args, **kwargs)
