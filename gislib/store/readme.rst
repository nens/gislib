Datastore for gridded data
==========================

Roadmap
-------
Update workflow::
    Determine locations (and remember)
    try:
        reproject
        store
        yield_location
    except DoesNotFitError(dimension) as e: 
        # We have ned data that does not fit
        for child_location in location.get_children(dimension):
            reproject
            store
            yield child_location
        remove location

Reproject::
    if configs equal:
        assign and return
    gdaldimension: reproject
    timedimension: recalendar and affine
    ned time time dimension: modify axes and data




We have a lowlevel storage facility. It must have some config:
    - Addition to other store: get_adapter() method for stores;
    - get_locations_for_config(self, config):
        like get_locations, but takes into account projections. method of FrameConfig.

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

