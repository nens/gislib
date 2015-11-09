Changelog of gislib
===================================================


0.7 (2015-11-09)
================

- Added ``calculate_great_circle_distance`` function to the vectors
  module (including unittests).

- Update to newest bootstrap


0.6 (2014-12-24)
================

- No topping in get_counts.


0.5 (2014-11-06)
================

- Nothing changed yet.


0.4 (2014-03-04)
================

- Implement nodatavalue in NumpyContainer


0.3 (2014-02-20)
----------------

- Added NumpyContainer, used in threedi-wms

0.2.14 (2014-01-29)
-------------------

- Update landuse colors.


0.2.13 (2013-12-18)
-------------------

- Added counts for landuse raster.

- Implement get_value for pyramids.

- Added colors to nudge an extra module ;).

0.2.12 (2013-10-17)
-------------------

- Fix get_data datatype handling.

- Remove progress indicator in favor of debug line on the pyramid script.


0.2.11 (2013-10-14)
-------------------

- Add statistics module with ground curve function.


0.2.10 (2013-10-08)
-------------------

- Change debug messaging in pyramid and pyramid script.


0.2.9 (2013-10-07)
------------------

- Fix pyramid add (the new pyramid).


0.2.8 (2013-10-07)
------------------

- Fixed get_transformed_extent, added Geometry.fromextent.

- Implement generic store data retrieval for geometries


0.2.7 (2013-09-26)
------------------

- Add a script to round gdal datasets.

- Log the tilepath on exceptions during pyramid add

- Log the sourcepath on exceptions during pyramid add

- Do not raise, but always add to pyramid baselevel, regardless of source resolution


0.2.6 (2013-09-25)
------------------

- Fix error in extent2polygon.


0.2.5 (2013-09-25)
------------------

- Move extent2polygon to utils

- Rewrite get_transformed_extent


0.2.4 (2013-09-24)
------------------

- Get_profile now accepts a wkt.


0.2.3 (2013-09-24)
------------------

- Fix for the get_profile.


0.2.2 (2013-09-23)
------------------

- Make get_array() and get_profile() directly accessible from pyramid objects.


0.2.1 (2013-09-23)
------------------

- Implement get_profile for pyramids wrapper.

- Make it possible for pixelize to use non-square pixels.


0.2 (2013-09-18)
----------------

- Add convenience method for authorities and start using SetFromUserInput.
  Integers are no longer accepted as argument for get_spatial_reference.


0.1.4 (2013-09-18)
------------------

- Add juggle script to multiprocessed calculate hillshades and others.

- More accurate extent for pyramids.


0.1.3 (2013-09-17)
------------------

- Fix abs bug. Boundaries are now correctly calculated.


0.1.2 (2013-09-16)
------------------

- New pyramid class added that supports repeated additions and more.


0.1.1 (2013-08-01)
------------------

- Use signed integers for pyramid indices.


0.1 (2013-05-18)
----------------

- Initial project structure created with nensskel 1.33.dev0.

- Add modules for vector and raster data.

- Add a pyramid object to store vast sizes of raster data.
