# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import math

import numpy as np

"""
The splitting of data in blocks and levels is determined in the frame object.
a frame consists of a number of dimensions and a definition of the
basic datablock.
"""

# =============================================================================
# Dimensions
# -----------------------------------------------------------------------------
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
        self.scale = kwargs.get('scale', 1)

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

    def get_sublocations(self, extent, level):
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
        return lambda: (Sublocation(level=level, indices=indices) 
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


# =============================================================================
# Frames
# -----------------------------------------------------------------------------
class Frame(object):
    def __init__(self, dimensions, dtype, nodatavalue):
        """
        For ned dimensions, need to add a dtype per ned dimension (or
        even multiple, consider the case of ned 3d space). Then the
        specified datatype must be prepended by the parameters into the
        ned interval.
        """
        self.dimensions = dimensions
        self.dtype = dtype  # Any pickleable numpy dtype object will do.
        self.nodatavalue = nodatavalue  # Also used to identify NED removals.

        # Slices to acces binary data
        location_nedims = 8 * sum(1 + d.M for d in self.dimensions)
        nedims_data = 0
        self.location = slice(location_nedims)
        self.nedims = slice(location_nedims, nedims_data)
        self.data = slice(nedims_data, None)

    @property
    def size(self):
        """ Return bytesize. """
        raise NotImplementedError

    @property
    def shape(self):
        """ Return shape based on frame. """
        raise NotImplementedError


    def make_dataset_string(self, dataset):
        """ Return serialized dataset string. Includes the NED dims. """
        return sum([n.tostring() 
                    for n in dataset.nedims] + [dataset.array.tostring()])

    def extract_location(self, data):
        """ Load location from blobstring. """
        v = iter(np.fromstring(location[self.location], dtype=np.int64))
        return tuple(dimensions.Location(level=v.next(),                                   
                                         indices=tuple(v.next() 
                                                       for n in range(d.M)))
                     for d in self.dimensions)

    def extract_scales(self, data):
        """ Get the scales in the data. """
        return 'scales'

    def extract_array(self, data):
        """ Get the array in the data. """
        return np.ma.array(np.ma.masked_equal(
            fromstring(data[self.data], dtype=self.dtype),
            self.nodatavalue,
        )

    def to_dataset(self, data):
        """ Return a dataset corresponding to data. """
        return Dataset(
            extent=self.get_extent(self.extract_location(data)),
            scales=self.extract_scales(data),
            array=self.extract_array(data),
        )

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


# =============================================================================
# Locations
# -----------------------------------------------------------------------------
Sublocation = collections.namedtuple('Sublocation', ('level', 'indices'))

class Location(object):
    def __init__(self, frame, sublocations):
        self.sublocations = sublocations
        self.frame = frame
        self.key = hashlib.md5(self.tostring()).hexdigest()
    
    def get_extent(self):
        """ Return the extent covered by a chunk."""
        return tuple(dimension.get_extent(sublocation)
                     for sublocation, dimension in zip(self.sublocations,
                                                       self.frame.dimensions))
    
    def get_parent(self, dimension=0, levels=1):
        """ Return a location. """
        level = self.sublocations[dimension].level + levels
        extent = self.get_extent()[dimension]
        insert = self.frame.dimensions[dimension].get_sublocations(
            extent=extent, level=level,
        )().next(),
        sublocations = (self.sublocations[:dimension] +
                        insert + self.sublocations[dimension + 1:])
        return Location(frame=self, sublocations=sublocations)

    def get_children(self, dimension=0):
        """ Return a location generator. """
        level = self.sublocations[dimension].level - 1
        extent = self.get_extent()[dimension]
        inserts = self.frame.dimensions[dimension].get_sublocations(
            extent=extent, level=level,
        )()
        for insert in inserts:
            sublocations = (self.sublocations[:dimension] +
                            (insert,) + self.sublocations[dimension + 1:])
            yield Location(frame=self, sublocations=sublocations)

    def get_root(self, func):
        """ 
        Return the root chunk. It only works if the data is fully
        aggregated in all dimensions to a single chunk at the highest
        level.

        Func must be a function that returns true when there is data at
        location, and false otherwise.
        """
        self[self.DATA]  # Crash if we don't even have data ourselves.
        # Get parents in all dimensions until nodata.
        root = self
        for i in range(len(self.store.structure.dimensions)):
            begin = 0
            end = 1
            # First begin extending end until non-existing chunk found.
            while True:
                try:
                    root.get_parent(dimension=i, levels=end)[self.DATA]
                except KeyError:
                    break
                end = end * 2
            while end - begin != 1:
                # Now begin testing the middle of end until end - begin == 1 again.
                middle = (begin + end) // 2
                try:
                    root.get_parent(dimension=i, levels=middle)[self.DATA]
                except KeyError:
                    end = middle
                    continue
                begin = middle
            if begin:
                root = root.get_parent(dimension=i, levels=begin)
        return root

    def tostring(self):
        return self.toarray().tostring()

    def toarray(self):
        return np.int64([j for l, i in self.sublocations for j in (l,) + i])

    def __str__(self):
        return '<Location: {}>'.format(self.toarray())


# =============================================================================
# Datasets
# -----------------------------------------------------------------------------
class Dataset(object):
    """
    """

    def __init__(self, data, structure):
        """ 
        Init.        
        """
        self.structure = structure
        self.location = structure.get_location(data)
        self.nedims = structure.get_nedims(data)
        self.data = structure.get_array(data)

    def tostring(self):
        return structure.tostring(
