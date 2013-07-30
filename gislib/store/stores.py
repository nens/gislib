# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import hashlib
import math
import pickle

import numpy as np

from gislib.store import datasets
from gislib.store import kinds

GuideLocus = collections.namedtuple('Locus', ('level', 'indices'))


class GridLocus(object):
    """ Defines a location in a frame. """
    def __init__(self, loci):
        self.loci = loci
        self.key = hashlib.md5(self.tostring()).hexdigest()

    def tostring(self):
        return self.toarray().tostring()

    def toarray(self):
        return np.int64([j for l, i in self.loci for j in (l,) + i])

    def __str__(self):
        return '<Locus: {}>'.format(self.toarray())

    def __repr__(self):
        return '<Locus: {}>'.format(self.toarray())

    def __eq__(self, locus):
        return self.tostring() == locus.tostring()


class Guide(object):
    """ Base class for guides. """
    def __init__(self, kind, size, offset=None, factor=None, base=2):
        self.kind = kind
        self.size = size  # Size of chunk for this domain
        self.base = base  # resolution ratio between levels
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

    def get_domain(self, extent, size):
        """ Return domain for use in dataset config. """
        return datasets.Domain(kind=self.kind, extent=extent, size=size)

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

    def get_extent(self, locus):
        """
        Return extent tuple for a sublocation

        Location must have a resolution level and an iterable of indices.
        Example format for 2D-extent: ((x1, y1), (x2, y2))
        """
        return tuple(tuple((self.base ** locus.level *
                            self.size[i] *
                            (locus.indices[i] + j) + self.offset[i])
                           for i in range(len(locus.indices)))
                     for j in (0, 1))

    def get_loci(self, extent, level):
        """
        Return a function that, when called, returns a generator of loci.

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
        get_func = lambda r: lambda: ((i,) for i in r)
        funcs = [get_func(r) for r in irange]

        # Reduce by nesting the generators, combine with level and return
        # again a function.
        def reducer(f1, f2):
            return lambda: (i + j for j in f2() for i in f1())

        # Combine the (sub)domains and return a function
        return lambda: (GuideLocus(level=level, indices=indices)
                        for indices in reduce(reducer, funcs)())

    def create_axis(self):
        """ Return axes for discrete kinds, None otherwise. """
        if isinstance(self.kind, kinds.DiscreteKind):
            return np.ones(self.size) * -1


class Store(object):
    """
    Raster object with optional arguments including.
    """
    CONFIG_KEY = 'config'

    def __init__(self, storage, **kwargs):
        """
        Separate schemas in the storage are placed as attributes on
        the store.
        """
        # Init schemas
        for schema in ('databox', 'metabox', 'config', 'metadata'):
            split = False if schema == 'config' else True
            setattr(self, schema, storage.get_schema(schema, split=split))

        # Add a schema for each aggregator when they are added.
        # TODO

        if kwargs:
            # New store
            self._check_if_new()
            self.config[self.CONFIG_KEY] = pickle.dumps(kwargs)
            init_kwargs = kwargs
        else:
            # Existing store
            init_kwargs = pickle.loads(self.config[self.CONFIG_KEY])

        self.guides = init_kwargs['guides']
        self.dtype = init_kwargs['dtype']
        self.fill = init_kwargs['fill']

        self.locus_size = 8 * sum([1 + len(g.size) for g in self.guides])
        self.axes_size = 0

    def _check_if_new(self):
        """ If the store already has a config, raise an exception. """
        try:
            self.config[self.CONFIG_KEY]
        except KeyError:
            return  # That's expected.
        raise IOError('Store already has a structure!')

    @property
    def size(self):
        return tuple(d.size for d in self.guides)

    @property
    def shape(self):
        return tuple(j for i in self.size for j in i)

    # =========================================================================
    # Location handling
    # -------------------------------------------------------------------------
    def _get_extent(self, locus):
        """ Return extent tuple. """
        return tuple(d.get_extent(s)
                     for s, d in zip(locus.loci, self.guides))

    def _get_trans_geom(self, dataset):
        """
        Return dictionary with size and extent.

        Size and extent are with respect to store guides.
        """
        make_domains = lambda d, g: d.transform(kind=g.kind)['domain']
        domains = map(make_domains, dataset.domains, self.guides)
        size, extent = zip(*((d.size, d.extent) for d in domains))
        return dict(size=size, extent=extent)

    def _get_loci(self, dataset):
        """
        Return a generator of loci.
        """
        # Need to determine size and extent in stores kinds.
        geom = self._get_trans_geom(dataset)
        size = geom['size']
        extent = geom['extent']

        # Determine the appropriate levels
        level = tuple(g.get_level(e, s)
                      for g, s, e in zip(self.guides, size, extent))

        # Get the funcs that return the per-dimension locus generators
        funcs = (g.get_loci(e, l)
                 for g, e, l in zip(self.guides, extent, level))

        # Nest via reduce function
        def reducer(f1, f2):
            return lambda: ((i,) + (j,)
                            for j in f2() for i in f1())

        return (GridLocus(loci=loci)
                for loci in reduce(reducer, funcs)())

    def _get_parent(self, locus, i=0, levels=1):
        """
        Return a location.

        locus: location for which to return the parent
        levels: Amount of level to traverse
        i: is the index to the guides.
        """
        level = locus.parts[i].level + levels
        extent = self.get_extent(locus)[i]
        insert = self.guides[i].get_loci(
            extent=extent, level=level,
        )().next(),

        return GridLocus(
            loci=locus.loci[:i] + insert + locus.loci[i + 1:],
        )

    def _get_children(self, locus, i=0):
        """
        Return a locus generator. 

        i is the index to the guides.
        """
        level = locus.loci[i].level - 1
        extent = self.get_extent()[i]
        inserts = self.guide[i]._get_loci(extent=extent, level=level)()
        for insert in inserts:
            loci = (locus.loci[:i] + (insert,) + locus.loci[i + 1:])
            yield GridLocus(loci=loci)

    def _get_root(self, func):
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

    # =========================================================================
    # Data handling
    # -------------------------------------------------------------------------

    def _read_locus(self, string):
        """ Load location from string. """
        v = iter(np.fromstring(string[self.location], dtype=np.int64))
        return Location(
            parts=tuple(GuideLocus(level=v.next(),
                                   indices=tuple(v.next() for size in domain))
                        for domain in self.config.domains),
        )

    def _read_axes(self, string):
        """ Get the domains in the data. """
        return np.fromstring(string[self.axes])

    def _read_data(self, string):
        """ Get the array in the data. """
        data = np.frombuffer(string[self.data], dtype=self.dtype)
        data.shape = self.config.shape
        data.flags.writeable = True  # This may be a bug.
        return data

    def _read_dataset(self, string):
        """ Create a dataset from a string. """
        if string == string[self.location]:
            raise ValueError('No data found beyond location')
        location = self.get_location(string)
        dataset_kwargs = dict(
            location=location,
            config=self.config.get_dataset_config_for_location(location),
            axes=self.get_axes(string),
            data=self.get_data(string),
        )
        return datasets.SerializableDataset(**dataset_kwargs)

    def _create_axes(self):
        """
        Return empty axes for self.
        """
        return tuple(g.create_axis() for g in self.guides)

    def _create_dataset(self, locus):
        """ 
        Return empty dataset for location.
        """
        extent = self._get_extent(locus)
        kwargs = dict(
            domains=[g.get_domain(size=s, extent=e)
                     for g, s, e in zip(self.guides, self.size, extent)],
            fill=self.fill,
            axes=self._create_axes(),
            data = np.ones(self.shape, self.dtype) * self.fill
        )
        return datasets.SerializableDataset(locus, **kwargs)

    def __getitem__(self, location):
        """ Return dataset or location. """
        try:
            string = self.databox[location.key]
            return self._read_dataset(string)
        except ValueError:
            return location  # Or would the empty string suffice?
        except KeyError:
            return self._create_dataset(location)

    def __setitem__(self, location, dataset):
        """ Write dataset to location. """
        self.databox[location.key] = dataset.tostring()

    def __delitem__(self, location):
        """ Remove data at location. """
        del self.databox[location.key]

    def add_from(self, adapter):
        """
        Put data in the store from an iterable of datasets.

        Returns a generator of updated loci.
        """
        for source in adapter:
            for locus in self._get_loci(source):
                target = self[locus]
                try:
                    datasets.reproject(source, target)
                    yield locus
                except datasets.DoesNotFitError as error:
                    dimension = error.dimension
                    for sublocus in locus.get_children(dimension):
                        target = grid.get_dataset(sublocus)
                        reproject(source, target)  # should fit now
                        grid.remove_dataset(locus) # data has moved
                        yield sublocus
            exit()
    

    def fill_into(self, dataset):
        """
        Fill dataset with data from the store.
        """
        for location in self.frame.config.get_locations(dataset.config):
            #string = self.databox[location.key]
            #t = self.get_dataset(location)
            #datasets.reproject(t, dataset)
            datasets.reproject(self.get_dataset(location), dataset)
