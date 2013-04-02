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

Post-nensskel setup TODO
------------------------

Here are some instructions on what to do after you've created the project with
nensskel.

- Fill in a short description on https://github.com/lizardsystem/gislib or
  https://github.com/nens/gislib if you haven't done so already.

- Use the same description in the ``setup.py``'s "description" field.

- Fill in your username and email address in the ``setup.py``, see the
  ``TODO`` fields.

- Check https://github.com/nens/gislib/settings/collaboration if the team
  "Nelen & Schuurmans" has access.

- Add a new jenkins job at
  http://buildbot.lizardsystem.nl/jenkins/view/djangoapps/newJob or
  http://buildbot.lizardsystem.nl/jenkins/view/libraries/newJob . Job name
  should be "gislib", make the project a copy of the existing "lizard-wms"
  project (for django apps) or "nensskel" (for libraries). On the next page,
  change the "github project" to ``https://github.com/nens/gislib/`` and
  "repository url" fields to ``git@github.com:nens/gislib.git`` (you might
  need to replace "nens" with "lizardsystem"). The rest of the settings should
  be OK.
