# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import datetime
import glob
import logging
import math
import os
import time

from osgeo import gdal
from osgeo import ogr
from osgeo import osr

import numpy as np

from gislib import projections
from gislib import rasters
from gislib import stores
from gislib import vectors

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)

GDAL_DRIVER_GTIFF = gdal.GetDriverByName(b'gtiff')
TIMEOUT = 60  # seconds


def initialize(*args):
    """ Set some pool data globally. """
    global initargs
    initargs = args


def warp(source, targetpath):
    target = gdal.Open(targetpath, gdal.GA_Update)
    source.warpinto(target)


def crop(dataset):
    """
    Return cropped memorycopy of dataset.

    ReadAsArray()
    Find nonzero edges
    adjust geotransform accordingly
    array2dataset
    projection, geotransform, nodatavalues, etc.
    Some useful code:
        peak = self.info['peak']
        dataset = peak.get_dataset(peak.info['top_tile'])

        # Return geotransform as matrices
        p, a, b, q, c, d = np.array(dataset.GetGeoTransform())

        # Determine the pixels that are not nodata
        array = dataset.GetRasterBand(1).GetMaskBand().ReadAsArray()
        pixels = np.array(tuple((i.min(), i.max() + 1)
                          for i in array.nonzero()))[::-1]

        # Return as extent
        (x1, x2), (y2, y1) = np.dot([[a, b], [c, d]], pixels) + p, q
        return x1, y1, x2, y2
    """
    logger.warn('Cropper nog implemented yet!')
    return dataset


def dataset2outline(dataset, projection=None):
    """
    Return a polygon formed by dataset edges, wrapped in a Geometry.

    TODO Make use of np.dot, ideal for geotransform stuff.
    """
    nx, ny = dataset.RasterXSize, dataset.RasterYSize
    if projection is None:
        # will not be transformed, corners are enough
        p, a, b, q, c, d = dataset.GetGeoTransform()
        points = np.dot([[a, b], [c, d]], [[0, nx, nx, 0, 0],
                                           [0, 0, ny, ny, 0]]) + [[p], [q]]
        polygon = vectors.points2polygon(points.transpose())
    else:
        # take all pixelcrossings along the edge into account
        npoints = 2 * (nx + ny) + 1
        array = np.empty((npoints, 2))

        # Construct pixel indices arrays
        px = np.empty(npoints, dtype=np.uint64)
        py = np.empty(npoints, dtype=np.uint64)
        # top
        a, b = 0, nx
        px[a:b] = np.arange(nx)
        py[a:b] = 0
        # right
        a, b = nx, nx + ny
        px[a:b] = nx
        py[a:b] = np.arange(ny)
        # bottom
        a, b = nx + ny, 2 * nx + ny
        px[a:b] = np.arange(nx, 0, -1)
        py[a:b] = ny
        # left
        a, b = 2 * nx + ny, 2 * (nx + ny)
        px[a:b] = 0
        py[a:b] = np.arange(ny, 0, -1)
        # repeat first
        px[-1], py[-1] = 0, 0

        # Use geotransform to convert the points to coordinates
        xul, dxx, dxy, yul, dyx, dyy = dataset.GetGeoTransform()
        array[:, 0] = xul + dxx * px + dxy * py
        array[:, 1] = yul + dyx * px + dyy * py

        polygon = vectors.array2polygon(array)

    geometry = vectors.Geometry(polygon)
    if projection is None:
        return geometry
    geometry.transform(source=dataset.GetProjection(), target=projection)
    return geometry


def dataset2pixel(dataset, projection=None):
    """ Return a polygon formed by pixel edges, wrapped in a Geometry. """
    p, a, b, q, c, d = dataset.GetGeoTransform()
    points = np.dot([[a, b], [c, d]], [[0, 1, 1, 0, 0],
                                       [0, 0, 1, 1, 0]]) + [[p], [q]]
    polygon = vectors.points2polygon(points.transpose())
    geometry = vectors.Geometry(polygon)
    if projection is None:
        return geometry
    geometry.transform(source=dataset.GetProjection(), target=projection)
    return geometry


def get_options(blocksize):
    """
    Return blocksize dependent gtiff creation options.
    """
    return ['TILED=TRUE',
            'SPARSE_OK=TRUE',
            'COMPRESS=DEFLATE',
            'BLOCKXSIZE={}'.format(blocksize[0]),
            'BLOCKYSIZE={}'.format(blocksize[1])]


def get_config(dataset):
    """ Return dictionary. """
    band = dataset.GetRasterBand(1)
    return dict(datatype=band.DataType,
                bands=dataset.RasterCount,
                blocksize=band.GetBlockSize(),
                nodatavalue=band.GetNoDataValue(),
                projection=dataset.GetProjection(),
                rastersize=(dataset.RasterYSize, dataset.RasterXSize))


def get_tiles(spacing, extent):
    """
    Return generator of index tuples.

    Indices identify the tiles that intersect given extent on a grid
    with given spacing.
    """
    # Determine the ranges for the tiles
    x_range, y_range = map(
        xrange,
        (int(math.floor(e / s))
         for e, s in zip(extent[:2], spacing)),
        (int(math.ceil(e / s))
         for e, s in zip(extent[2:], spacing)),
    )

    # Iterate over the ranges
    for y in y_range:
        for x in x_range:
            yield x, y


class LockError(Exception):
    pass


class Transport(object):
    """
    Simple in-shared-memory pyramid designed to fill the bigger pyramid
    storage.
    """
    def __init__(self, dataset, stop=256):
        """
        Add overviews of the dataset until the overview is smaller
        than given size.
        """
        self.levels = [rasters.SharedMemoryDataset(dataset)]
        while max(self.levels[-1].array.shape[1:]) > stop:
            self.levels.append(rasters.SharedMemoryDataset(
                dataset=self.levels[-1].dataset, shrink=2,
            ))

        self.pixelsizes = [dataset2pixel(l.dataset).size for l in self.levels]
        self.projection = self.levels[0].dataset.GetProjection()

    def get_level(self, dataset):
        """ Return the appropriate level for reprojection into dataset. """
        pixelsize = dataset2pixel(dataset=dataset,
                                  projection=self.projection).size

        for p, l in reversed(zip(self.pixelsizes, self.levels)):
            if max(p) < min(pixelsize):
                return l
            return self.levels[0]

    def warpinto(self, dataset):
        """
        Reproject appropriate level from self into dataset.
        """
        rasters.reproject(self.get_level(dataset).dataset, dataset)


class Grid(object):
    """
    Represent a grid of rastertiles.
    """
    def __init__(self, config):
        # from config dictionary
        self.__dict__.update(config)
        # derived
        self.spacing = tuple(c * t
                             for c, t in zip(self.tilesize,
                                             self.cellsize))

    def tile2geotransform(self, tile):
        """ Return geotransform for a tile. """
        return (
            self.spacing[0] * tile[0],
            self.cellsize[0],
            0,
            self.spacing[1] * (tile[1] + 1),
            0,
            -self.cellsize[1],
        )

    def tile2path(self, tile):
        """ Return the path for the tile file. """
        return os.path.join(self.path, '{}', '{}.tif').format(*tile)

    def create(self, tile):
        """ Create a new dataset. """
        # prepare
        path = self.tile2path(tile)
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # directory already exists
        # create
        dataset = GDAL_DRIVER_GTIFF.Create(path,
                                           self.tilesize[0],
                                           self.tilesize[1],
                                           self.bands,
                                           self.datatype,
                                           get_options(self.blocksize))
        # config
        dataset.SetProjection(projections.get_wkt(self.projection))
        dataset.SetGeoTransform(self.tile2geotransform(tile))
        for i in range(self.bands):
            dataset.GetRasterBand(i + 1).SetNoDataValue(self.nodatavalue)

    def get_tiles(self, dataset):
        """
        Return generator of tile indices.
        """
        outline = dataset2outline(dataset=dataset, projection=self.projection)
        return get_tiles(spacing=self.spacing, extent=outline.extent)

    def get_paths(self, dataset):
        """
        Return path generator.

        It is guaranteed that a dataset exists at every path in the
        generator.
        """
        for tile in self.get_tiles(dataset):
            path = self.tile2path(tile)
            if not os.path.exists(path):
                logger.debug('Create {}'.format(path))
                self.create(tile)
            else:
                logger.debug('Update {}'.format(path))
            yield path

    def get_datasets(self, dataset=None):
        """
        Return read-only dataset generator.

        If a dataset is supplied, yield available datasets that intersect
        with datasets extent. Otherwise, return all available datasets
        in the grid.
        """
        if dataset is None:
            # These paths always exist
            paths = glob.iglob(os.path.join(
                self.path, str(self.level), '*', '*',
            ))
        else:
            # These paths may or may not exist
            paths = (self.get_path(tile) for tile in self.get_tiles(dataset))
        for path in paths:
            try:
                yield gdal.Open(self.get_path(tile))
            except RuntimeError:
                continue

    def warpinto(self, dataset):
        """ Warp appropriate tiles into dataset. """
        for source in self.get_datasets(dataset):
            rasters.reproject(source, dataset)


class Manager(object):
    """
    Keeps the pyramid properties and delegates read and write requests
    to appropriate pyramid layers.
    """
    def __init__(self, path):
        """ Derive availability of layers from filesystem. """
        # Determine levels
        self.path = path
        try:
            names = os.listdir(path)
        except OSError:
            names = []
        levels = [int(n) for n in names if not n.startswith('.')]
        self.levels = sorted(levels)
        if not levels:
            return

        # Determine config from sample dataset
        self.config = get_config(self.top)

    def __getattr__(self, name):
        """ Return items from config for convenience. """
        if name == 'config':
            raise AttributeError(name)
        return self.config[name]

    def __getitem__(self, level):
        """ Index is the index to the level, so it is not the level itself. """
        """ Return store corresponding to level. """
        cellsize = tuple(2 * [2 ** level])
        path = os.path.join(self.path, str(level))
        config = dict(cellsize=cellsize, path=path)
        config.update(self.config)
        return Grid(config)

    @property
    def peakpath(self):
        return os.path.join(self.path, '.pyramid.tif')

    @property
    def extent(self):
        """ Return pyramid extent tuple. """
        try:
            return dataset2outline(gdal.Open(self.peakpath)).extent
        except RuntimeError:
            return None

    def bootstrap(self, dataset, overrides):
        """ Bootstrap manager for a new pyramid. """
        self.config = get_config(dataset)
        self.config.update(overrides)
        self.levels = range(self.get_level(dataset),
                            self.get_toplevel(dataset) + 1)

    def sync(self):
        """
        Create a dataset at the top of half the blocksize
            Get extent of highest managed level
        - , determine extent of data containing part via
        - Create toptile accordingly, but shrink by 2
        - warpinto from highest store.
        - Do a temp & move to keep it available
        """
        pass

    def get_level(self, dataset):
        """
        Return appropriate level for dataset.
        """

        pixelsize = min(dataset2pixel(dataset=dataset,
                                      projection=self.projection).size)
        return int(math.floor(math.log(pixelsize, 2)))

    def get_toplevel(self, dataset):
        """
        Return the level for which the size of the combined extent of
        the dataset and peak fits within the size of a single block.
        """
        outline = dataset2outline(dataset=dataset,
                                  projection=self.projection)
        extent = self.extent
        if extent is None:
            combinedsize = outline.size
        else:
            x1, y1, x2, y2 = zip(extent, outline.extent)
            combinedsize = max(x2) - min(x1), max(y2) - min(y1)
        pixelsize = max(d / b for d, b in zip(combinedsize, self.blocksize))
        return int(math.ceil(math.log(pixelsize)))

    def extend(self, newmax):
        """
        Extend levels to include level.

        Used if there is data in the pyramid, but the amount of levels
        need to be extended.
        """
        oldmax = self.levels[-1]
        levels = range(oldmax + 1, newmax + 1)
        self.levels.extend(levels)
        for source in self[oldmax].get_datasets():
            transport = Transport(source)
            paths = (p for l in levels for p in self[l].get_paths(source))
            for path in paths:
                warp(transport, path)

    def add(self, dataset, **kwargs):
        """ Add a dataset to manager. """
        # prepare for data addition
        if self.levels:
            oldlevel = self.levels[-1]
            newlevel = self.get_newlevel(dataset)
            if newlevel > oldlevel:
                self.extend(newlevel)
        else:
            self.bootstrap(dataset=dataset, overrides=kwargs)

        # do the data addition
        transport = Transport(dataset)
        paths = (p for l in self.levels for p in self[l].get_paths(dataset))
        for path in paths:
            warp(transport, path)

    def warpinto(self, dataset):
        """
        Return true if anything was warped, false otherwise.

        Warp tiles from appropriate store into dataset.
        """
        level = max(self.level[0], self.get_level(dataset))
        if level in self.levels:
            return self[level].warpinto(dataset)
        try:
            rasters.reproject(
                source=gdal.Open(self.peakpath),
                target=dataset,
            )
        except RuntimeError:
            pass

    def single(self, point):
        """ Return value from lowest level. """
        pass


class Pyramid(stores.BaseStore):
    """
    Pyramid datastore. Physically consisting of a number of grids.
    """
    def __init__(self, path):
        """
        The idea is to initialize almost nothing, so that any other
        instances will always be up-to-date.
        """
        self.path = path

    @property
    def manager(self):
        """ Return cached pyramid manager. """
        now = time.time()
        available = hasattr(self, '_manager')
        expired = now > self._manager['expires'] if available else True
        if not available or expired:
            manager = Manager(self.path)
            expires = now + TIMEOUT
            self._manager = dict(expires=expires, manager=manager)
        return self._stores['manager']

    @property
    def lockpath(self):
        return os.path.join(self.path, '.pyramid.lock')

    def lock(self):
        """ Create a lockfile. """
        # Create directory if it does not exist in a threadsafe way
        try:
            os.makedirs(os.path.dirname(self.lockpath))
        except:
            pass
        # Make a lockfile. Raise LockException if not possible.
        try:
            fd = os.open(self.lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            raise LockError('Object is locked.')

        # Write current date in the lockfile.
        with os.fdopen(fd, 'w') as lockfile:
            lockfile.write(str(datetime.datetime.now()))

    def unlock(self):
        """ Remove a lockfile """
        os.remove(self.lockpath)
        try:
            os.removedirs(os.path.dirname(self.lockpath))
        except:
            pass

    def add(self, dataset, sync=True, **kwargs):
        """
        Any kwargs are used to override dataset projection and datatype,
        nodatavalue, tilesize. Kwargs are ignored if data already exists
        in the pyramid.
        """
        self.lock()
        manager = Manager(self.path)  # do not use the cached manager
        manager.add(dataset, **kwargs)
        self.unlock()
        if sync:
            manager.sync()

    def sync(self):
        return self.manager.sync()


    #def warpinto(self, dataset):
        #""" Warp data from the pyramid into dataset. """
        #elf.manager.warpinto(dataset):
            #return
        #try:
            #rasters.reproject(
                #source=gdal.Open(self.toppath),
                #target=dataset,
            #)
        #except RuntimeError:
            #pass

    #def single(self, point):
        #""" Return value from lowest level. """
        #pass
