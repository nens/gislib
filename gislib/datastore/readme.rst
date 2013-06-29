Datastore for gridded data
==========================

Need to keep track of some stuf during updating:
- Aggregated chunks must not be updated directly, only via aggregations.

- Chunks at higher zoom must first copy lower zoom ancestor data before
becoming the bottom data.

- Need to check if current toplevel chunk is updated, or else aggregation
must continue until a new toplevel chunk is found.

- When aggregating: First aggregate the chunks with highest zoom. Done.

How fast is it to find the top chunk?
The optimal chunks for reading a certain extent / zoom?
The realdata chunks?

Must any request start with determining the toplevel chunk? How fast is
that? But otherwise we won't notice the differences. And that would be
bad, too.

Chunk updating system
---------------------

Chunks have a unique location that maps one-to-one with an extent. However
data sources and targets have extents and datastructures, but they don't
(necessarily) map onto the chunk extents. Therefore, algorithms are
needed to translate source and target datasets to and from chunks.

We also need adapters to convert things like gdal datasets to the
datastore structure. It would be nice to predict the chunks that will
be updated from a complete set of datasources, in order to divide the
jobs between processes.

Aggregation system
------------------

Chunks that have been written must be automatically added to a
set of updated chunks. Their parents have to be updated too, use
multiprocessing. The aggregations use predefined algorithms according
to the datastore structure. Only when we aggregate aggregated
dimensions(Highly complex stuff?) There can be a master chunk that
contains the full datastore extent. Otherwise, it has to be constructed
from the last aggregation from each aggregation.

Requirements
------------

Datastore (?) can translate chunk location to extent Datastore (?) can
translate extent to chunk(location)generator Note that when updating, we
need to keep track of the orginal extent, because the last aggregations
must include the original data.

Masterchunkfinder based on datastore.first()
Meaningful chunkfinder
Chunks parents and children finders.


Structure
---------
A structure knows the dimensions of the dataset. It can convert locations
to extents and vice versa.

Aggregators
-----------
Take care of the aggregation of data in a dimension from chunks with
more detail to chunks with less detail.
Resampling, simplification, etc.

Retrievers and ned dimensions
-----------------------------
For a ned dimension, just grab anything in the extent and clip.
For a ed dimension, grab anything in the extent and resample according to call.

NED dimensions have some special properties:

- A default zoomlevel for the first point (when there is one point, it
is not clear at which zoom to put it. As soon there is a second point,
the zoom is settled and the original zoom can be cleared. Or should
the default just be zoom '0'? What about the precision of the parameter
then? However, usually, more than one value will soon be there.

Some design considerations: - A non-equidistant dataset stores a range
in the dimension of a chunk, and a single precision parameter that
markes the position of the data in the extent. For example, we have a
NED  time chunk from 2 to 3 seconds, and the parameter specifies 0.1,
meaning that the time of the event is 2 + 0.1 * (3 -2).

- Clearing data form NED dimensions requires some tolerance specified, to
determine if a location is a new one or not. Let's not implement that now.

- Updated data always aggregates using all available aggregators. During
the process, the datastore can be read, but the aggregations may not
show the latest results.

- NED dimensions can only add data to the chunks with the highest
resolution. To be consistent, ED chunks also accept only data at
their lowest resolution, otherwise raise an exception 'Trying to put data at an aggregated level for dimension ....'
So we can guarantee consistency and prevent
dataloss. That means the user has to explicitly clear a datastore
if he wants to add lowres stuff, by filling with nodata at the lower
resolution and running a clean operation on the whole store. Expensive,
but it isn't logical behaviour for typical use case anyway.

A datastore does not deal with optimizations in the form of blocksize
tweaking. Simply create another datastore and update this datastore with
it whenever possible. But a datastore does try to update with very high
performance, using multiprocessing and in-memory merged chunks whenever
possible.

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


This would be nice:
    datastore.iterchunks(extent=???), what aggregation level?
    datastore.itermeaningfulchunks
    datastore.add_data
    datastore.iterchunks(extent)

    chunk.parent(dimension)
    chunk.children(dimension)


Arbitrary source datasets => generator for datasets (we need an
object!) in the storage structure => Generator for chunklevel data.

When a chunk is created at a lower level, one always need to put the
higher level data into it. This holds for both ned and ed.
