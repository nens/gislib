Todo
====
    
Tests:
    - Almost all lowlevel stuff in vectors, rasters and projections

Documentation:
    - How to use the Pyramid

Make blockwise updating the responsibility of some special reproject function

Grid redesign:
    - Standalone grid, autodetecting properties from path
    - Support for arbitrary offsets
    - Investigate using h5 for storage performance inprovement

Pyramid redesign:
    - Pyramid is just a loose collection of Grid objects
    - There are classes Grid and Pyramid that lock, _Grid and _Pyramid
      that do not lock, but essentially do the same things. The difference
      is there for two reasons:
        - Locking / not locking (not all applications need locking)
        - Caching of _Grid and _Pyramid instances that refresh every now and then.
    - Maybe get rid of the concept of levels. Just order the grids by their cellsize.
    - Top level will always have block and tile size 256 and consist of
      only one dataset.
    - A retile method will retile intermediate grids
      that have 256 tile size to the basegrid tilesize.
    - Initialization:
      Init all the grids and build numpy interpolation data for grid
      assignment

Time incorporation:
    - Settle on some standard directory structure.
    - Use an hdf5 index using np.datetime64 for time, int64 for intra and interblock indices.
