TODO:
    aggregate up to the point where less then one pixel contains the data,
    so when top is found, go up some amount based on base and tilesize.

    quicker non-segfaulting way to get at memory datasets (look at writeraster)
    for reading: Use the gdal_array method

    rounding problem determining the top level when in same projection etc.

    empty array generation somehow generates float64 data.
