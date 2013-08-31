New plan
========
- Keep it simple!
- Use key value store already developed
- Use embedded location system
- Use ogr for topdown location system for the data

- Store has separate multiband aggregating pyramid store and a simple time store

- Time configuration:
    datatype
    units
    blocksize (65535)

- Data configuration:
    projection
    datatype
    blocksize spatial (256, 256, 1), hybrid (16, 16, 256), time (1, 1, 65535)

datalocations: tile_y, tile_x, tile_level, flag_original, data
timelocations: block, data

Rasters are stored like it is now, but the time dimension becomes simpler. Additionaly, a flag as stored to indicate original data.
It is an exception when one tries to store basedata in a tile flagged as aggregated data.



Any store now has simple methods similary to the pyramid way:
    add(dataset, time=slice(0, 1))
    warpinto(dataset, time=slice(0, 1))

Additional, the stores times can be set using standard slice notation:
    store.time[0] = 5
    time = store.time[0]
but in the background this gets stored in the multichunked time dataset.

Also, the store has a method that returns a generator of (dataset, time) tuples, for a . This can be used to copy the dataset into a similar store with different blocksizes.

Now we know how to present arbitrary numpy array as gdal dataset, warping is easy.
Raster 

def dataset2polygon(dataset):
    """ Return polygon formed by pixel edges. """
    pass

class Store(object):
    def __init__(self, storage):
    """ Try to read time & data configuration from storage. """


    def create_time(self, unit, size=(65536,))
        self.time = Time(schema=schema, units=units, size=size)

    def create_data(self, projection, size=(16, 16, 256))
        self.data = Data(schema=schema, projection=projection, size=size)
    
    def warpinto(self, dataset, times):
        """ """
        pass

    def __getitem__(self, times):
        """ Return a generator of datasets for times.
            They come in the blocks defined by data.


class Data(object):
    def __init__(self, schema, units, datatype, shape):
        """ Writes config, or reads it from storage. """

    def __setitem__(self, times, dataset):
        """ Previously the pyramids add method """

    def __getitem__(self, times):
        """ Return a generator of datasets for times.
            They come in the blocks defined by data.
        



class Time(object):
    """ Just a large array! """
    def __init__(self, schema, units=None, datatype=None, shape=None):
        """ Writes config, or reads it from storage. """

    def __setitem__(self, times, data)

    def __getitem__(self, times)


Bottlenecks
===========
Wishes:
    - have get_locations return locations sorted per parent location in all directions.

Reproject: 
    Immediately if projections and calendars and extents are matching
    Space
        Always gdal nearest neighbour; one should configure the frame to match source data.
    Time
        Coards + affine
    NED time: Coards & clip is ok for retrieving, but not for updating.
    NED arbitrary: clip / projection and clip, whatever.

Aggregate:
    - converters.Raster
    - converters.Time
    - converters.Generic
    - converters.DouglasPuecker
    - converters.NearestNeighbour
    - converters.CompleteDomainConverter?
- Add aggregation framework
- Add metadata framework

Testing:
- Lower level storage tests
- Lower level tests for scales and metrics.

Aggregators:
- Take care of lining up the right datasets and using the right converters.

Adapters:
- Put various formats into our structure format (for example gdal)
- Get various formats from our structure format

- Do we need to always query for the top chunk? No!
    - Only when aggregating
    - Investigate for get_root vs just trying all datasets at a level.


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
