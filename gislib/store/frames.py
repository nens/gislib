# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import hashlib
import math

import numpy as np

from gislib.store import datasets

Sublocation = collections.namedtuple('Sublocation', ('level', 'indices'))


class Location(object):
    """ Defines a location in a frame. """
    def __init__(self, parts):
        self.parts = parts
        self.key = hashlib.md5(self.tostring()).hexdigest()

    def tostring(self):
        return self.toarray().tostring()

    def toarray(self):
        return np.int64([j for l, i in self.parts for j in (l,) + i])

    def __str__(self):
        return '<Location: {}>'.format(self.toarray())

    def __repr__(self):
        return '<Location: {}>'.format(self.toarray())

    def __eq__(self, location):
        return self.tostring() == location.tostring()


class Domain(object):
    """ A domain combined with offset, domain and base. """
    def __init__(self, domain, size, offset=None, factor=None, base=2):
        self.base = base  # resolution ratio between levels
        self.domain = domain
        self.size = size
        multiplicity = len(size)

        # Domain factor with respect to the base unit
        if factor is None:
            self.factor = (1,) * multiplicity
        else:
            self.factor = factor

        # Tile origin offset
        if offset is None:
            self.offset = (0,) * multiplicity
        else:
            self.offset = offset

    def __iter__(self):
        """ Convenience for the deserialization of locations. """
        return (None for s in self.size)

    def get_level(self, extent, size):
        """
        Return integer.

        Level 0 is defined as the level where the smallest side
        of a cell spans exactly one unit multiplied by factor.
        """
        return int(math.floor(math.log(
            min((e2 - e1) / s / f
                for s, f, e1, e2 in zip(size, self.factor, *extent)),
            self.base,
        )))

    def get_extent(self, sublocation):
        """
        Return extent tuple for a sublocation

        Location must have a resolution level and an iterable of indices.
        A format for a 2D-extent: ((x1, y1), (x2, y2))
        """
        return tuple(tuple((self.base ** sublocation.level *
                            self.size[i] *
                            (sublocation.indices[i] + j) + self.offset[i])
                           for i in range(len(sublocation.indices)))
                     for j in (0, 1))

    def get_sublocations(self, extent, level):
        """
        Return a function that, when called, returns a generator of locations.

        Why a function? So reduce() can be used to iterate over all the
        locations of all the domains of the enclosing structure.

        If resolution does not exactly match, locations for the next
        matching higher resolutions are returned.
        """
        # Determine the block span at requested level
        span = tuple(self.base ** level * s * f
                     for s, f in zip(self.size, self.factor))
        # Determine the ranges for the indices
        irange = map(
            xrange,
            (int(math.floor((e - o) / s))
             for e, o, s in zip(extent[0], self.offset, span)),
            (int(math.ceil((e - o) / s))
             for e, o, s in zip(extent[1], self.offset, span)),
        )

        # Prepare for reduce by creating a list of functions
        funcs = [lambda: ((i, ) for i in r) for r in irange]

        # Reduce by nesting the generators, combine with level and return
        # again a function.
        def reducer(f1, f2):
            return lambda: (i + j for j in f2() for i in f1())

        # Combine the (sub)domains and return a function
        return lambda: (Sublocation(level=level, indices=indices)
                        for indices in reduce(reducer, funcs)())


class Config(object):
    """ Collection of frame domains. """

    def __init__(self, domains):
        self.domains = domains

    @property
    def size(self):
        return tuple(d.size for d in self.domains)

    @property
    def shape(self):
        return tuple(j for i in self.size for j in i)

    def get_extent(self, location):
        """ Return extent tuple. """
        return tuple(d.get_extent(s)
                     for s, d in zip(location.parts, self.domains))

    def get_config(self, location):
        """ Return dataset config. """
        extent = self.get_extent(location)
        domains = [datasets.Domain(domain=d.domain, extent=e, size=s)
                   for d, s, e in zip(self.domains, self.size, extent)]
        return datasets.Config(domains=domains)

    # =========================================================================
    # Location navigation
    # -------------------------------------------------------------------------
    def get_parent(self, location, axis=0, levels=1):
        """
        Return a location.

        location: location for which to return the parent
        axis: Index into self.domains
        levels: Amount of level to traverse
        """
        level = location.parts[axis].level + levels
        extent = self.get_extent(location)[axis]
        insert = self.domains[axis].get_sublocations(
            extent=extent, level=level,
        )().next(),

        return Location(
            parts=location.parts[:axis] + insert + location.parts[axis + 1:],
        )

    def get_children(self, location, axis=0):
        """ Return a location generator. """
        level = location.parts[axis].level - 1
        extent = self.get_extent()[axis]
        inserts = self.frame.axiss[axis].get_sublocations(
            extent=extent, level=level,
        )()
        for insert in inserts:
            sublocations = (location.parts[:axis] +
                            (insert,) + location.parts[axis + 1:])
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
                # Now begin testing the middle of end until
                # end - begin == 1 again.
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


# =============================================================================
# Frames
# -----------------------------------------------------------------------------
class Frame(object):
    def __init__(self, config, dtype, fill_value):
        """
        For ned dimensions, need to add a dtype per ned dimension (or
        even multiple, consider the case of ned 3d space). Then the
        specified datatype must be prepended by the parameters into the
        ned interval.
        """
        self.config = config
        self.dtype = dtype  # Any pickleable numpy dtype object will do.
        self.fill_value = fill_value  # Also used to identify NED removals.

        # Slices to acces binary data
        axis = 8 * sum(1 + len(f.size) for f in self.config.domains)
        data = axis  # No (ned) axis currently.
        self.location = slice(axis)
        self.axes = slice(axis, data)
        self.data = slice(data, None)

    @property
    def size(self):
        """ Return bytesize. """
        return self.dtype.itemsize * np.prod(self.shape)

    @property
    def shape(self):
        """ Return numpy shape. """
        return self.config.shape

    @property
    def empty(self):
        """ Return empty masked array. """
        data = np.ma.masked_all(self.shape, self.dtype)
        data.fill_value = self.fill_value
        return data

    def get_config(self, location):
        """ Convenience method. """
        return self.config.get_config(location)

    def get_location(self, string):
        """ Load location from string. """
        v = iter(np.fromstring(string[self.location], dtype=np.int64))
        return Location(
            parts=tuple(Sublocation(level=v.next(),
                                    indices=tuple(v.next() for size in domain))
                        for domain in self.config.domains),
        )

    def get_axes(self, string):
        """ Get the domains in the data. """
        return np.fromstring(string[self.axes])

    def get_data(self, string):
        """ Get the array in the data. """
        result = np.ma.array(np.ma.masked_equal(
            np.fromstring(string[self.data],
                          dtype=self.dtype),
            self.fill_value,
            copy=False,
        )).reshape(*self.shape)
        result.fill_value = self.fill_value
        return result

    def get_saved_dataset(self, string):
        """ Create a dataset from a string. """
        if string == string[self.location]:
            raise ValueError('No data found beyond location')
        location = self.get_location(string)
        dataset_kwargs = dict(
            location=location,
            config=self.get_config(location),
            axes=self.get_axes(string),
            data=self.get_data(string),
        )
        return datasets.SerializableDataset(**dataset_kwargs)

    def get_empty_dataset(self, location):
        """ Return empty dataset for location. """
        dataset_kwargs = dict(
            location=location,
            config=self.get_config(location),
            axes=tuple(),
            data=self.empty,
        )
        return datasets.SerializableDataset(**dataset_kwargs)

    def get_locations(self, config):
        """
        Return a generator of location tuples.
        """
        extent = config.extent
        size = config.size
        # Coerce resolutions to levels
        level = tuple(d.get_level(e, s)
                      for d, e, s in zip(self.config.domains, extent, size))

        # Get the funcs that return the per-dimension sublocation generators
        funcs = (s.get_sublocations(e, l)
                 for s, e, l in zip(self.config.domains, extent, level))

        # Nest via reduce function
        def reducer(f1, f2):
            return lambda: ((i,) + (j,)
                            for j in f2() for i in f1())

        return (Location(parts=parts)
                for parts in reduce(reducer, funcs)())
