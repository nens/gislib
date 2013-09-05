Todo
====
Raster:
- rename data objects to raster objects for clarity
- make warpinto method accepting a dataset and filling it with store data
- make the time indices actually work:
    when adding: init an array that contains dataset and integer amount
       of chunks as a dataset. reproject and then write the blocks one
       by one.
    when reading: init an array, fill from chunks one by one and then
       create a dataset from it, reproject.

- bypass reproject if it is unnecessary - very fast reading of tiled layers

Time:
- add init_time(self, units, datatype) method similar to 
- make __setitem__ and __getitem__ work for time.

Interface:
    store.raster[indices] = dataset  # dataset bandcount must correspond to indices
    store.raster[indices] returns generator of original datablocks.
    store.raster.projection  # epsg, proj4, wkt?
    store.time[indices] = array
    store.time[indices] returns an array
    store.time.units  # coards?
    store.raster.guess_extent()  # maybe add amount of tries?
    store.raster.read_extent()  # intensive, reads all blocks
    store.raster.warpinto(dataset)
implement slicing instead of reprojection if transformation indicates roughly same.
