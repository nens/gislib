# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from gislib.datastore import chunks
from gislib.datastore import files
from gislib.datastore import utils
from gislib.datastore import aggregators
from gislib.datastore import dimensions

""" 
General code about the module here.

Some design considerations: - A non-equidistant dataset stores a range
in the dimension of a chunk, and a single precision parameter that
markes the position of the data in the extent. For example, we have a
NED  time chunk from 2 to 3 seconds, and the parameter specifies 0.1,
meaning that the time of the event is 2 + 0.1 * (3 -2).

- Clearing data form NED dimensions requires some tolerance specified, to
determine if a location is a new one or not. Let's not implement that now.

- A dimension need also to specify an offset. So that we can have for
radar data a unit like 'days since 1-1-2012' offsetted with 8 hours.

- Updated data always aggregates using all available aggregators. During
the process, the datastore can be read, but the aggregations may not
show the latest results.

- NED dimensions can only add data to the chunks with the highest
resolution. To be consistent, ED chunks also accept only data at
their lowest resolution. So we can guarantee consistency and prevent
dataloss. That means the user has to explicitly clear a datastore
if he wants to add lowres stuff, by filling with nodata at the lower
resolution and running a clean operation on the whole store. Expensive,
but it isn't logical behaviour for a typical anyway.

A datastore does not deal with optimizations in the form of blocksize
tweaking. Simply create another datastore and update this datastore with
it whenever possible. But a datastore does try to update with very high
performance, using multiprocessing and in-memory merged chunks whenever
possible.

A datastore has a method to 
- find first arbitrary chunk

- find the toplevel extent from chunk, by grabbing an arbitrary chunk
and walking to the top of the aggregation pyramid. So, aggregation must
be compulsory then?

- Let's say we don't do multidimensional aggregations. How then to find the extent of a non-aggregating dimension? No, we have to aggregate them, or keep track of the extent via the storage; but that would imply some index. No. Let's say, we don't do multidimensional aggregations, but always do aggregation in any dimension. Or do we do single block dimensions? Makes stuf complex.

- Return an iterable of all basechunks (highest resolution chunks)find
the total extent by walking down from top level chunk to all highest
resolution chunks.

So choices are: 
    - disallow non-aggregating dimensions?
        - What about performance?
        - What about flat blockdimensions? They can't be aggregated! Or can they?
        - What about restricting aggregation in orthogonal directions?

    - non-aggregating, single-block dimension?
    - non-aggregating, keep-track of extent in config? No. No state in configuration.
    - non-aggregating, walking chunks? No, possibly unlimited chunks in a directions.


Create converter: gdal2chunks: structure

"""



class Datastore(object):
    """
    Raster object with optional arguments including
    time. Choices to be made:

    chunksize, filesize
    dimensions: ['time', 'spatial', 'spatial']
    calendars:
    equidistant time or not?
    equidistant x y or not?
    z or not?
    coupled x y
    """
    def __init__(self, storage, structure=None):
        """ initialize. """
        self.storage = storage
        self.structure = structure
        if storage is None:
            self.storage = storage.load()
        else
            self.structure = structure
        if self.storage is None:
            raise ValueError('No struc')
        storage.load()
        try:
        if structure is None:
            structure is storage.load()
        else:
            

        pass

    def update(data, extent):
        """
        Update extent with data in numpy masked array
        """
        pass

    def update(iterable):
        pass


