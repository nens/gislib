Datastore for gridded data
==========================

What's next?
- gdal dataset to extent filters, requiring values for new dimensions.
- Extent => locations => filters
- Warpers per 
- Build filters for adding data:
    - filter: extent for a dataset
    - datastore: chunks for extent
    - datastore: adapter for dataset to chunk
    - extent to chunks
    - chunk to extents
    - smart adapter that buffers arbitrary amount of data in memory until writing to arbitrary amount of chunks.
    - gdal dataset in, chunkiterator out.
    - build auto aggregation
    - build chunk iterators:
        - all chunks
        - extent based chunks

    - build masterchunkfinder, meaningfulchunkiterator b
    - 

- Adding some equidistant data from multiple sources
    - Use multiprocessing
    - Do simple aggregation (no warping pleaz)
    - How to determine the extents of the data?
- Extracting some equidistant data
- Dealing with the requests of 

Filters
-------
Convert source data to chunkdata and vice versa. The datastore delivers the correct chunks based on the souces, and the filter takes care of filling the chunks, or reading the chunks.

Chunks
------
A chunk contains data, metadata and a location in the multidimensional space of a datastore. Using file storage, data, metadata and location are separately stored.

Datastore
---------
A datastore is a collection of chunks in a multidimensional space.
The Structure defines the datastores properties. With a structure, we can predict what chunks can be found where, if they exist; even different aggregations.
- Get an arbitrary chunk from the storage
- Determine it's location
- Find the masterchunk
- Guide an add operation. Before the add, we have the master chunk. After add, there may be a new master which must contain the masters of the new and the old data.

Structure
---------
A structure knows the dimensions of the dataset. It can convert locations to extents and vice versa.

Aggregators
-----------
Take care of the aggregation of data in a dimension from chunks with more detail to chunks with less detail.
Resampling, simplification, etc.

Retrievers
----------
For a ned dimension, just grab anything in the extent and clip.
For a ed dimension, grab anything in the extent and resample according to call.

Storage of structure and location
---------------------------------
Location gets pickled by chunk when stored, but unpickled by who? The datastore?

NED dimensions have some special properties:

- A default zoomlevel for the first point (when there is one point, it is not clear at which zoom to put it. As soon there is a second point, the zoom is settled and the original zoom can be cleared. Or should the default just be zoom '0'? What about the precision of the parameter then? However, usually, more than one value will soon be there.

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
their lowest resolution. So we can guarantee consistency and prevent
dataloss. That means the user has to explicitly clear a datastore
if he wants to add lowres stuff, by filling with nodata at the lower
resolution and running a clean operation on the whole store. Expensive,
but it isn't logical behaviour for typical use case anyway.

A datastore does not deal with optimizations in the form of blocksize
tweaking. Simply create another datastore and update this datastore with
it whenever possible. But a datastore does try to update with very high
performance, using multiprocessing and in-memory merged chunks whenever
possible.

A datastore has a method to 
- find first arbitrary chunk

- find the toplevel extent from chunk, by grabbing an arbitrary chunk
and walking to the top of the aggregation pyramid. So, aggregation must
be compulsory then? At least, from a chunk in the datastore we must be able to infer it's chunk extent. Therefore we must save the chunk extent with the chunk data. Let's do that using (c)Pickle.

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

    chunk.data.get()
    chunk.data.put()
    chunk.meta.get()
    chunk.meta.put()
    chunk.location.put()
    chunk.location.
    chunk.parent(dimension)
    chunk.children(dimension)
    chunk.

    chunk.data.put()


- A chunk must now its siblings(?), parent and children.
- 

    

But what about the aggregations?



Create converter: gdal2chunks: structure


Attach storage to chunk directly after unpickling. Remove storage from chunk object before pickling.Easy.




Initialization:
    Existing storage: no structure allowed!
    New storage: structure must be supplied.




storage must be able to produce a single chunk? Na, then we need to import the chunks in the storages. Don't want to.

When creating a chunk from the storage, we need to get the meta and the data for an arbitrary chunk from the storage. single chunk from the storage.

Hey, what about separate storage for base chunks and aggregated chunks? Base chunks can easily be iterated over for copying, then. Maybe later.

Store must be able to produce one or more tiles.

Store produces locations; with a location a chunk can be instantiated. It must receive a storage as well. 

