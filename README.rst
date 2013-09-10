gislib
==========================================

Warp algorithms test for gdal
-----------------------------
Gdalwarp on the commandline uses an error tolerance of 0.125, but the python bindings default it to 0! Don't forget to set it to something else than 0, because this makes the warping really slow.

Some tests with downsampling of landuse imagery:
near:     0.390s
bilinear: 0.735s <= This one gave a better antialias effect than cubic
cubic:	  1.153s
lanczos: 43.388s

Usage, etc.

Creating pyramids for the geoserver pyramid plugin
--------------------------------------------------
Create a bunch of symlinks in <myDirWithSymlinks>::
    
    find <mySourceDir> | grep tif$ | xargs -I {} ln -s {}

Create the pyramid using gdal::

    gdal_retile.py -v -co COMPRESS=DEFLATE -co TILED=YES -of gtiff -ps 2000 2500 -s_srs epsg:28992 -levels 10 -r near -targetDir <myTargetDir> <myDirWithSymLinks>

Working with the time-pyramid raster store
------------------------------------------
store = stores.Pyramid(path_to_store)
store.add(gdal.Open(path_to_rasterfile)
store.warpinto(my_writable_gdal_dataset)
