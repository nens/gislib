Datastore for gridded data
==========================

Todo
----
Pitch:
- Just show data adding and retrieving, and converting to another store
  optimized for timeseries.
- Simple retrieving should work as long as you know where the data is.
  Do something with AHN2 to show the speed, or radar too.
- Configuring a store is tedious. Make json templates:
    - AHN2
    - Radar

Testing:
- Lower level storage tests
- Lower level tests for scales and metrics.

Naming:
- SpatialDimension => domains.Raster
- TimeDimension => domains.Time
- UnitDimension => domains.Quantity

- FrameDimension => frames.Domain
- DatasetDimension => datasets.Domain
- FrameMetric => frames.Config
- DatasetMetric => datasets.Config

Conversion
- converters.Raster
- converters.Time
- converters.Global?
- converters.DouglasPuecker
- converters.NearestNeighbour

Aggregators:
- Take care of lining up the right datasets and using the right converters.

Adapters:
- Put various formats into our structure format (for example gdal)
- Get various formats from our structure format


Roadmap
-------
- Conversion gdal => dataset => image.
- Make demo page to demonstrate fast image retrieval
- 'fill method' to fill datasets with same dimensions, but different calendars, projections, etc.
- Multiprocessed addition of datasets.
- Aggregation (denken)
- Metadata
- Non-equidistant

- Do we need to always query for the top chunk? No!
    - Only when aggregating
    - Investigate for get_root vs just trying all datasets at a level.

Datatructure
============
Store
    (Aggregators)
    Storage
    Frame
        FrameConfig
            FrameScale(dimension, offset, scale, factor)
            FrameScale...
            ...

Dataset
    Config
        Domain(dimension, extent)
        Domain...
        ...
    Axes
    Data

Location becomes just a container. Ask your frame config for the extent
of a location, or the root, or the parents or children.

Aggregation system
------------------
- Aggregation:
    - Aggregate up to the level where there is only one pixel or datavalue left in the block.
    - A base aggregator stores no data, but stores the location.
    - Separate store for aggregated data, multiple aggregators possible
    - Define aggregators in the store per dimension.
    - Adding an aggregator:
        - Selected Aggregators stored in common storage.
        - Assign storage schema
        - keys will be created from aggregator names. Aggregators are lists at each level
        - aggregated data: key will be hash of aggregators a1b2 for example.
        - Link all data in databox schema if it is the first dimension
        - Link all data from all aggregators of the previous dimension
        - No metadata for aggregators
    - Removing aggregator deletes aggregate data
    - Enabling aggregates all data again...


Thoughts on non-equidistant datasets
------------------------------------
ned scale aggregation: For a given level increase, gather all ned values in other dimensions and aggregate them, then aggrate the scale under hand.

ned scale updating: If there is only one point in a dimension, it will be placed at level 0. If another point is added, the level can be determined, but one has to check level 0 for the existence of a single point.


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
