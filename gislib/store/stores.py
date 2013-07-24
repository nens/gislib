# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import pickle

from gislib.store import datasets

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


Space = collections.namedtuple('Space', ('proj',))
Time = collections.namedtuple('Time', ('unit',))


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
        A format for a 2D-extent: ((x1, y1), (x2, y2))
        """
        return tuple(tuple((self.base ** sublocation.level *
                            self.size[i] *
                            (sublocation.indices[i] + j) + self.offset[i])
                           for i in range(len(sublocation.indices)))
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
        get_func = lambda r: lambda:((i,) for i in r)
        funcs = [get_func(r) for r in irange]

        # Reduce by nesting the generators, combine with level and return
        # again a function.
        def reducer(f1, f2):
            return lambda: (i + j for j in f2() for i in f1())


        # Combine the (sub)domains and return a function
        return lambda: (Sublocation(level=level, indices=indices)
                        for indices in reduce(reducer, funcs)())



class Grid(object):
    """ Defines a store. """
    def __init__(self):
        """ Define indexes into data based. """
        self.locus_size = 8 * sum([1 + len(g.size) for g in self.guides])
        self.axes_size = 0
        
        
    @property
    def size(self):
        return tuple(d.size for d in self.guides)

    @property
    def shape(self):
        return tuple(j for i in self.size for j in i)

    def get_extent(self, locus):
        """ Return extent tuple. """
        return tuple(d.get_extent(s)
                     for s, d in zip(locus.loci, self.guides))
    
    def get_config(self, size, extent):
        """ 
        Return dataset config.
        
        Convenient if a dataset must be created that matches this grid.
        """
        domains = [g.get_domain(size=s, extent=e)
                   for g, s, e in zip(self.guides, size, extent)]
        dtype = self.dtype
        fill = self.fill
        return datasets.Config(domains=domains, dtype=dtype, fill=fill)
    
    def get_config_for_location(self, location):
        """ 
        Return dataset config.

        Used by store to create a dataset for the data at some location
        """
        size = self.size
        extent = self.get_extent(location)
        return self.get_config(size=size, extent=extent)

    # =========================================================================
    # Location navigation
    # -------------------------------------------------------------------------
    def get_loci(self, dataset):
        """
        Return a generator of loci.
        """
        extent = config.extent
        size = config.size
        # Coerce resolutions to levels
        level = tuple(d.get_level(e, s)
                      for d, e, s in zip(self.domains, extent, size))

        # Get the funcs that return the per-dimension sublocation generators
        funcs = (s.get_sublocations(e, l)
                 for s, e, l in zip(self.domains, extent, level))

        # Nest via reduce function
        def reducer(f1, f2):
            return lambda: ((i,) + (j,)
                            for j in f2() for i in f1())

        return (Location(parts=parts)
                for parts in reduce(reducer, funcs)())

    def get_parent(self, locus, guide=0, levels=1):
        """
        Return a location.

        location: location for which to return the parent
        axis: Index into self.domains
        levels: Amount of level to traverse
        """
        level = location.parts[axis].level + levels
        extent = self.get_extent(locus)[guide]
        insert = self.guides[guide].get_loci(
            extent=extent, level=level,
        )().next(),

        return GridLocus(
            loci=locus.loci[:axis] + insert + locus.loci[axis + 1:],
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

    # =========================================================================
    #  Methods that convert stored data to objects
    # -------------------------------------------------------------------------

    def get_locus(self, string):
        """ Load location from string. """
        v = iter(np.fromstring(string[self.location], dtype=np.int64))
        return Location(
            parts=tuple(GuideLocus(level=v.next(),
                                   indices=tuple(v.next() for size in domain))
                        for domain in self.config.domains),
        )

    def get_axes(self, string):
        """ Get the domains in the data. """
        return np.fromstring(string[self.axes])

    def get_data(self, string):
        """ Get the array in the data. """
        data = np.frombuffer(string[self.data], dtype=self.dtype)
        data.shape = self.config.shape
        data.flags.writeable = True  # This may be a bug.
        return data

    def get_saved_dataset(self, string):
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

    def get_empty_dataset_for_config(self, config, location=None):
        """ Return empty dataset. """
        dataset_kwargs = dict(
            config=config,
            axes=tuple(),
            data=np.ones(config.shape, self.dtype) * self.config.fill,
        )
        if location is None:
            return datasets.Dataset(**dataset_kwargs)
        dataset_kwargs.update(location=location)
        return datasets.SerializableDataset(**dataset_kwargs)

    def get_empty_dataset(self, extent, size):
        """ Return empty dataset. """
        config = self.config.get_dataset_config(extent=extent, size=size)
        return self.get_empty_dataset_for_config(config=config)
    
    def get_empty_dataset_for_location(self, location):
        """ Return empty dataset. """
        config=self.config.get_dataset_config_for_location(location)
        return self.get_empty_dataset_for_config(config=config,
                                                 location=location)

    def get_locations(self, dataset, update=True):
        """ 
        This is the highest level location finder.
        - Reproject units and projections
        - 


            
        """

class Store(object):
    """
    Raster object with optional arguments including.
    """
    GRID = 'grid'

    def __init__(self, storage, grid=None):
        """
        Separate schemas in the storage are placed as attributes on
        the store.
        """
        # Init schemas
        for schema in ('databox', 'metabox', 'config', 'metadata'):
            split = False if schema == 'config' else True
            setattr(self, schema, storage.get_schema(schema, split=split))

        # Add a schema for each aggregator when they are added.
        if grid is None:
            self.grid = pickle.loads(self.config[self.GRID])
            return

        # Write config
        self.verify_not_initialized()
        self.config[self.GRID] = pickle.dumps(grid)
        self.grid = grid

    def verify_not_initialized(self):
        """ If the store already has a structure, raise an exception. """
        try:
            self.config[self.GRID]
        except KeyError:
            return  # That's expected.
        raise IOError('Store already has a structure!')


    def get_dataset(self, location):
        """
        Return dataset or location.
        """
        try:
            string = self.databox[location.key]
            return self.frame.get_saved_dataset(string=string)
        except ValueError:
            return self.frame.get_location(string=string)
        except KeyError:
            return self.frame.get_empty_dataset_for_location(location=location)

    def put_dataset(self, location, dataset):
        """
        Write dataset to location.
        """
        self.databox[location.key] = dataset.tostring()

    def add_from(self, adapter):
        """
        Put data in the store from an iterable of datasets.

        Returns a generator of updated loci.
        """
        for source in adapter:
            for locus in self.grid.get_loci_for(dataset):
                target = grid.get(locus)
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
    

    def fill_into(self, dataset):
        """
        Fill dataset with data from the store.
        """
        for location in self.frame.config.get_locations(dataset.config):
            #string = self.databox[location.key]
            #t = self.get_dataset(location)
            #datasets.reproject(t, dataset)
            datasets.reproject(self.get_dataset(location), dataset)
