Todo
====

Tests:
    - Almost all lowlevel stuff in vectors, rasters and projections

Documentation:
    - How to use the Pyramid

Make blockwise updating the responsibility of some special reproject function

Merge enhanced pyramid and start new names for grids, pyramids, timestores.

Grid redesign:
    - Standalone grid
    - Store as hdf5
    - Store properties in json:
        - geotransform - that is for the superraster
        - file shape
        - block shape(s) - store twice for fast interband access
        - path levels
        - current extent (update on add)
        - current bands (update on add)
    - Use a sharedmem transport
    - Has paths, geotransform, pixels method for addition
    - Has datasets method for the warpinto method of the container
    - grid.add() assumes filling bands from 0 and up
    - grid[i] is effectively a grid instance with band offset i
    - grid[5:7].add() fills bands 5 and 6 and crashes if not 2 bands dataset?

Need object for paths:
    - Configurable depth and extension
    - Accepts tuples of ints (use numpy.tostring)
    - Coords = collections.namedtuple('Coords', ('x', 'y', 'z'))
    - coords = Coords(1, 2, 3)
    - key = hashlib.md5(np.int64(coords)).tostring()).hexdigest()
    - path = os.path.join(textwrap.wrap(key[:4], 2), key[:]

Need object to convert geotransform and fileshape:
    - fileindices
    - pixelindices and geotransforms per file
    - generator of above for given extent
    - counter of affected files for given dataset or extent
    - could use geometries for touch detection instead of using dataset envelope

Pyramid
    - Top level will always have block and tile size 256 and consist of
      only one dataset
    - On add keep pyramid conformed:
        reinit of toplevel grid:
        Needs current bands and extent to find datasets in old grid
    - Initialization:
        Init all the grids and build numpy interpolation
        data for grid assignment

TimeStores:
    - Separate 1D store for timedata
    - get_data must return a namedtuple with .datetime .value and optionally .distance

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
