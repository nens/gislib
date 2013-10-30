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

GTIFF = gdal.GetDriverByName(b'gtiff')
TIMEOUT = 60  # seconds


def initializer(*args):
    """ Set some pool data globally. """
    global initargs
    initargs = args


def warper(source, targetpath):
    target = gdal.Open(targetpath, gdal.GA_Update)
    source.warpinto(target)


def dataset2outlinepolygon(dataset):
    """ Return polygon formed by pixel edges. """
    nx, ny = dataset.RasterXSize, dataset.RasterYSize
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

    return vectors.array2polygon(array)


def dataset2pixelpolygon(dataset):
    """ Return polygon corresponding to the first pixel of dataset. """
    p, a, b, q, c, d = dataset.GetGeoTransform()
    points = np.dot([[a, b], [c, d]], [[0, 1, 1, 0, 0],
                                       [0, 0, 1, 1, 0]]) + [[p], [q]]
    return vectors.points2polygon(points.transpose())


def dataset2extent(dataset, projection=None):
    """
    Get extent of a dataset.

    If projection is given, transform extent first.
    """
    if projection is None:
        # quick lookup
        p, a, b, q, c, d = dataset.GetGeoTransform()
        w, h = dataset.RasterXSize, dataset.RasterYSize
        extent = np.dot([[a, b], [c, d]], [[0, w], [0, h]]) + [[p], [q]]
        return tuple(extent.ravel())
    else:
        # full transform
        outline = dataset2outlinepolygon(dataset)
        transformation = projections.get_coordinate_transformation(
            source=dataset.GetProjection, target=projection,
        )
        outline.Transform(transformation)
        return vectors.Geometry(outline).extent


def dataset2pixelsize(dataset, projection=None):
    """
    Get pixelsize of a dataset.

    If projection is given, transform extent first.
    """
    pixel = dataset2pixelpolygon(dataset)
    if projection is not None:
        # full transform
        transformation = projections.get_coordinate_transformation(
            source=dataset.GetProjection, target=projection,
        )
        pixel.Transform(transformation)
    return vectors.Geometry(pixel).size


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

        self.pixelsizes = [dataset2pixelsize(l.dataset) for l in self.levels]
        self.projection = self.levels[0].dataset.GetProjection

    def get_level(self, dataset):
        """ Return the appropriate level for reprojection into dataset. """
        pixelsize = dataset2pixelpolygon(dataset=dataset,
                                         projection=self.projection)

        for p, l in reversed(zip(self.pixelsize, self.levels)):
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

    def get_path(self, tile):
        """ Return the path for the tile file. """
        return os.path.join(self.path, '{}', '{}.tif').format(*tile)

    def get_geotransform(self, tile):
        """ Return geotransform for a tile. """
        return (
            self.spacing[0] * tile[0],
            self.cellsize[0],
            0,
            self.spacing[1] * (tile[1] + 1),
            0,
            -self.cellsize[1],
        )

    def get_tiles(self, dataset):
        """
        Return generator of tilepaths.
        """
        extent = dataset2extent(dataset, self.projection)
        return get_tiles(spacing=self.spacing, extent=extent)

    def get_datasets(self, dataset):
        """
        Return read-only dataset generator.

        Silently skips when dataset does not exist on disk.
        """
        for tile in self.get_tiles(dataset):
            try:
                yield gdal.Open(self.get_path(tile))
            except RuntimeError:
                continue

    def get_paths(self, dataset):
        """
        Return path generator.

        It is guaranteed that a dataset exists at every path in the
        generator.
        """
        for tile in self.get_tiles(dataset):
            path = self.get_path(tile)
            if not os.path.exists(path):
                self.create(tile)
            yield path

    def create_dataset(self, tile, info):
        """ Create a new dataset. """
        # prepare
        path = self.get_path(tile)
        logger.debug('Create {}'.format(path))
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # directory already exists
        # create
        dataset = GTIFF.Create(self.get_path(tile),
                               self.tilesize[0],
                               self.tilesize[1],
                               self.bands,
                               self.datatype,
                               get_options(self.blocksize))
        # config
        dataset.SetProjection(projections.get_wkt(self.projection))
        dataset.SetGeoTransform(self.get_geotransform(tile))
        for i in range(self.bands):
            dataset.GetRasterBand(i + 1).SetNoDataValue(self.nodatavalue)

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

    def bootstrap(self, dataset, overrides):
        """ Bootstrap manager for a new pyramid. """
        self.config = get_config(dataset)
        self.config.update(overrides)
        self.levels = range(self.get_level(dataset),
                            self.get_toplevel(dataset) + 1)


    @property
    def top(self):
        """
        Return toplevel dataset directly.

        Used to determine pyramid properties and to extend the pyramid.
        """
        path = glob.glob(os.path.join(
            self.path, str(self.levels[-1]), '*', '*',
        ))[0]
        return gdal.Open(path)

    def __getattr__(self, name):
        """ Return items from config for convenience. """
        return self.config[name]

    def level2grid(self, level):
        """ Return store corresponding to level. """
        cellsize = tuple(2 * [2 ** level])
        config = dict(cellsize=cellsize)
        config.update(self.config)
        return Grid(config)

    def get_level(self, dataset):
        """ Return appropriate level for dataset."""
        pixel = dataset2pixelpolygon(dataset)
        transformation = self.projection.get_coordinate_transformation(
            source=dataset.GetProjection(), target=self.projection,
        )
        pixel.Transform(transformation)
        pixelsize = vectors.Geometry(pixel).size
        return int(math.floor(math.log(min(pixelsize), 2)))

    def get_store(self, dataset):
        """ Instantiate the appropriate store for warping into dataset. """
        level = max(self.levels[0], self.get_level(dataset))
        if level > self.levels[-1]:
            return None
        else:
            return self.level2layer(level)

    def get_stores(self, levels=None):
        """ Return store generator for levels, for writing purposes. """
        if levels is None:
            levels = self.levels

        for l in levels:
            yield self.level2layer(l)

    def get_toplevel(self, dataset):
        """
        Get the first tile for which a block completely contains geometry.
        """
        extent = dataset2extent(dataset=dataset, projection=self.projection)
        x1, y1, x2, y2 = extent
        datasetsize = x2 - x1, y2 - y1

        # find the level for which the dataset fits within the blocksize
        pixelsize = max(d / b for d, b in zip(datasetsize, self.blocksize))
        level = [int(math.ceil(math.log(pixelsize)))]

        #grid = self.level2grid(level)
        #tiles = get_tiles(spacing=grid.spacing, extent=extent)
        #import ipdb; ipdb.set_trace()
        increment = 0
        return level + increment


class Pyramid(stores.BaseStore):
    """
    Pyramid datastore. Physically consisting of a number of pyramid layers
    and a top.
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

    # =========================================================================
    # Locking
    # -------------------------------------------------------------------------

    @property
    def lockpath(self):
        return os.path.join(self.path, '.lock')

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

    # =========================================================================
    # Top
    # -------------------------------------------------------------------------

    @property
    def toppath(self):
        return os.path.join(self.path, '.top')

    def topsync(self):
        """
        Create a dataset at the top of half the blocksize
        - Read the top tile, determine extent of data containing part
        - Create toptile accordingly, but shrink by 2
        - warpinto from highest store.
        - Do a temp & move to keep it available
        """
        # Below is code from old get_extent.
        """ Return the extent of the datapart in peaks top tile. """
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

    # =========================================================================
    # Interface
    # -------------------------------------------------------------------------

    @property
    def extent(self):
        """ Return extent of top. """
        return dataset2extent(gdal.Open(self.toppath))

    def extend(self, manager, level):
        """ Extend pyramid 

    def add(self, dataset, sync=True, **kwargs):
        """
        If there is no dataset, check if locked and reload.

        Any kwargs are used to override dataset projection and datatype,
        nodatavalue, tilesize. Kwargs are ignored if data already exists
        in the pyramid.

        Sync indicates wether to build a synced peak on top of the pyramid.
        """
        self.lock()

        # get actual info from stores. otherwise use dataset and kwargs
        manager = Manager(self.path)
        if manager.levels:
            oldlevel = manager.levels[-1]
            newlevel = manager.get_toplevel(dataset)
        else:
            manager.initialize(dataset=dataset, overrides=kwargs)
            upper = manager.get_toplevel(dataset)
            if upper
        current = manager.levels[-1]

        if upper > current:
            # First extend the pyramid with existing data
            top = manager.top
            transport = Transport(top)
            levels = (i + 1 for i in range(current, upper))
            for grid in manager.get_stores(levels):
                for path in grid.get_paths(top):
                    warper(transport, path)

        # Now add the data from the new dataset
        transport = Transport(dataset)
        for grid in manager.get_stores():
            for path in grid.get_paths(dataset):
                warper(transport, path)

        self.unlock()
        if sync:
            self.sync()

    def warpinto(self, dataset):
        """ Warp data from the pyramid into dataset. """
        # if no pyramid info, pyramid is empty.
        stores = self.stores
        if not stores:
            return

        # get bounds in pyramids projection
        projections.Get
        bounds = get_bounds(dataset=dataset, projection=info['projection'])
        level = max(get_level(bounds['pixel']), self.info['baselevel'])
            
        extent = vectors.Geometry(bounds['raster']).extent
        storey = self.info['storeys'].get(level)
        if storey is None:
            sources = [gdal.Open(self.peak)]
        else:
            tiles = get_tiles(extent=extent,
                              level=self.level,
                              tilesize=self.tilesize)
            sources = storey.get_datasets(tiles)
        for source in sources:
            rasters.reproject(source, dataset)

    def single(self, point):
        """ Return value from lowest level. """
        pass
