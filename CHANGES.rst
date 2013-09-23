Changelog of gislib
===================================================


0.3 (unreleased)
----------------

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
