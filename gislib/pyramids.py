# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import datetime
import logging
import os

from osgeo import gdal
import numpy as np

from gislib import projections
from gislib import rasters


# Enable gdal exceptions
gdal.UseExceptions()


class LockError(Exception):
    pass


class AbstractGeoContainer(object):
    """ Abstract class with locking mechanism. """

    def _lock(self):
        """ Create a lockfile. """
        # Create directory if it does not exist in a threadsafe way
        try:
            os.makedirs(os.path.dirname(self._lockpath))
        except:
            pass
        # Make a lockfile. Raise LockException if not possible.
        try:
            fd = os.open(self._lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            self._raise_locked_exception()

        # Write current date in the lockfile.
        with os.fdopen(fd, 'w') as lockfile:
            lockfile.write(str(datetime.datetime.now()))

    def _unlock(self):
        """ Remove a lockfile """
        os.remove(self._lockpath)

    def _raise_locked_exception(self):
        """ Raise locking specific OSError. """
        raise LockError('Object is locked.')

    def is_locked(self):
        """ Return if the container is locked for updating. """
        return os.path.exists(self._lockpath)

    def verify_not_locked(self):
        """ Return None or raise exception. """
        if self.is_locked():
            self._raise_locked_exception()


def _reproject_from_paths(tilepath, sourcepaths, **kwargs):
    """
    Reproject the datasets at sourcepaths into the dataset at tilepath.

    Kwargs will contain the algorithm. Uses reproject function from this module

    THis van
    """
    pass


def _create_tile(targetpath, templatepath, geotransform):
    """
    Create an empty tile at path, with the properties of the tile at
    templatepath and the indicated geotransform
    """
    pass


class Pyramid(AbstractGeoContainer):
    """
    New style pyramid object. The warpinto method behaves the same
    as the old pyramid object.  However, the add method now accepts
    a list of paths to be used wih gdal.Open, so /vsizip/, /vsicurl/,
    are allowed as well.

    features:
        - Multiprocessing is used for add operations, therefore
        - Much faster adding of both single and multiple file datasets
        - Configuration is not stored as json, but taken from toplevel tile
        - Changes are recorded in metadata, via commit-style messages
        - Arbitrary amount of zoomlevels
    """
    Tile = collections.namedtuple('Tile', ['x', 'y', 'z'])

    def __init__(self, path):
        """
        If path exists, configure from toplevel tile in pyramid if path exists

        """
        self.path = path
        if os.path.exists(self.path):
            self._read_config()

    def _read_config(self):
        """ Get pyramid parameters from configuration tile. """
        toptile = self._get_toptile()
        band = toptile.GetRasterBand(1)

        geometry = rasters.DatasetGeometry.from_dataset(toptile)
        self.extent = geometry.extent
        self.tilesize = geometry.size

        self.projection = projections.get_wkt(toptile.GetProjection())
        self.rastercount = toptile.GetRasterCount()

        # Note the assumption that bands have the same blocksize and datatype
        self.blocksize = band.GetBlockSize()
        self.datatype = band.GetDataType()

    def _get_tiledict(self, sourcepaths):
        """
        Return a dictionary of tilepath: sourcepaths.

        This dictionary can than be mapped to a multiprocessing pool to
        update the tiles.
        """
        # Per source: determine zoomlevel, extent, tile
        # Per tile: Determine sources (defaultdict)
        pass

    def _get_tilepath(self, tile):
        """ Convert a tile namedtuple to a path using self.path. """
        return os.path.join(self.path, tile.z, tile.x, tile.y + '.tif')

    def _get_tile(self, tilepath):
        """ Convert a tilepath to a tile namedtuple. """
        relpath = os.path.relpath(tilepath, self.path)
        root, ext = os.path.splitext(relpath)
        z, x, y = root.split(os.path.sep)
        return self.Tile(x=x, y=y, z=z)

    def add(self, sourcepaths,
            projection='epsg:3857',
            algorithm='near', compress='deflate',
            tilesize=(2048, 2048), blocksize=(256, 256),
            nodatavalue=np.finfo('f4').min, datatype=gdal.GDT_Float32):
        """
        If first add, config with defaults or from args.
        If not first add, raise if args are given.
        Determine which sources affect which tiles
        Project sources into tiles
        Update all tiles above the affected tiles
        """
        pass

    def warpinto(self):
        """
        """
        pass
