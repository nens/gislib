Todo
====
    
Tests:
    - Almost all lowlevel stuff in vectors, rasters and projections

Documentation:
    - How to use the Pyramid

Grid redesign:
    - Standalone grid, autodetecting properties from path
    - Support for arbitrary offsets
    - Investigate using h5 for storage performance inprovement

Pyramid redesign:
    - Investigate the performance improvement by using the transport
      instead of a basic sharedmemorydataset
    - Pyramid is just a loose collection of Grid objects
    - Maybe get rid of the factor 2 between levels
    - Top level will always have block and tile size 256 and consist of only one dataset
    - That is: a topshape routine will regenerate new tops until it satisfies above requirement.
    - After that it will conform the lower level grids to baselevel
    - Initialization: Init all the grids and build numpy interpolation data for grid assignment
