Todo
====
    
Tests:
    - Almost all lowlevel stuff in vectors, rasters and projections

Documentation:
    - How to use the Pyramid

Pyramid redesign:
    - class per level with info and warpinto
    - Reproject everything from the source - much faster than from highres subtiles
    - taper levels towards the top, no more peak dataset.
    - Generic function for multiprocessed reprojection of one soure to many targets
        - Make a joblist on which tiles to update
        - Keep source in shared memory
    - If new top:
        - Create temporary new levels (since tilesize changes)
        - Remove old ones on completion
        - After source reprojection, perform old top reprojection to higher levels
    - Create the datasets that don't exist yet beforehand
