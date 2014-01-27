# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import datetime
import glob
import logging
import math
import multiprocessing
import os
import tempfile
import time

from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
from osgeo import osr

import numpy as np

from gislib import projections
from gislib import rasters
from gislib import stores
from gislib import vectors
from gislib import utils

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)
transport = None

GDAL_DRIVER_GTIFF = gdal.GetDriverByName(b'gtiff')
TIMEOUT = 60  # seconds


def initialize(arg_transport):
    """ Set some pool data globally. """
    global transport
    transport = arg_transport


def warp(path_and_blocks):
    """ Warp global transport into specified blocks from dataset at path. """
    path, blocks = path_and_blocks
    for i, j in blocks:
        dataset = rasters.Dataset(gdal.Open(path, gdal.GA_Update))
        block = dataset.read_block((i, -j - 1))
        rasters.reproject(source=transport.dataset, target=block['dataset'])
        block['dataset'].FlushCache()  # really update underlying numpy array
        dataset.write_block((i, -j - 1), block['array'])


def point_from_dataset(dataset, point):
    x, y = point
    p, a, b, q, c, d = dataset.GetGeoTransform()
    minv = np.linalg.inv([[a, b], [c, d]])
    u, v = np.dot(minv, [x - p, y - q])
    band_list = range(1, dataset.RasterCount + 1)
    data = dataset.ReadRaster(int(u), int(v), 1, 1, band_list=band_list)
    data_type = gdal_array.flip_code(dataset.GetRasterBand(1).DataType)
    return np.fromstring(data, dtype=data_type)


def crop(dataset):
    """
    Return cropped memory copy of dataset.

    The code uses u, v, w for indices, x, y, z and p, q for points.
    """
    array = dataset.ReadAsArray()
    array.shape = dataset.RasterCount, dataset.RasterYSize, dataset.RasterXSize
    no_data_value = dataset.GetRasterBand(1).GetNoDataValue()
    indices = np.where(~np.equal(array, no_data_value))
    w1, w2, v1, v2, u1, u2 = (m for i in indices for m in (i.min(), i.max()))

    # determine new geotransform
    p, a, b, q, c, d = np.array(dataset.GetGeoTransform())
    points = (u1, u2), (v1, v2)
    (x1, x2), (y2, y1) = np.dot([[a, b], [c, d]], points) + ((p,), (q,))
    width, height = u2 - u1, v2 - v1
    geotransform = x1, (x2 - x1) / width, 0, y2, 0, (y1 - y2) / height

    # create cropped result dataset
    cropped = gdal_array.OpenArray(array[:, v1:v2, u1:u2])
    cropped.SetProjection(dataset.GetProjection())
    cropped.SetGeoTransform(geotransform)
    for i in range(cropped.RasterCount):
        cropped.GetRasterBand(i + 1).SetNoDataValue(no_data_value)

    return cropped


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
    if projection is None or projection == dataset.GetProjection():
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
    if projection is None or projection == dataset.GetProjection():
        return geometry
    geometry.transform(source=dataset.GetProjection(), target=projection)
    return geometry


def get_options(block_size):
    """
    Return block_size dependent gtiff creation options.
    """
    return ['TILED=TRUE',
            'SPARSE_OK=TRUE',
            'COMPRESS=DEFLATE',
            'BLOCKXSIZE={}'.format(block_size[0]),
            'BLOCKYSIZE={}'.format(block_size[1])]


def get_config(dataset):
    """ Return dictionary. """
    band = dataset.GetRasterBand(1)
    return dict(
        block_size=band.GetBlockSize(),
        data_type=band.DataType,
        no_data_value=band.GetNoDataValue(),
        projection=dataset.GetProjection(),
        raster_count=dataset.RasterCount,
        raster_size=(dataset.RasterYSize, dataset.RasterXSize),
    )


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


def get_tile(spacing, point):
    x, y = point
    extent = x, y, x, y
    return next(get_tiles(spacing, extent))


class LockError(Exception):
    pass


class Grid(object):
    """
    Represent a grid of rastertiles.
    """
    def __init__(self, **kwargs):
        """ Put all the kwargs on the class and derive spacing. """
        self.__dict__.update(kwargs)
        self.spacing = tuple(t * c
                             for t, c in zip(self.cell_size,
                                             self.raster_size))

    def tile2geotransform(self, tile):
        """ Return geotransform for a tile. """
        return (
            self.spacing[0] * tile[0],
            self.cell_size[0],
            0,
            self.spacing[1] * (tile[1] + 1),
            0,
            -self.cell_size[1],
        )

    def tile2extent(self, tile):
        """ Return tile extent. """
        return (
            self.spacing[0] * tile[0],
            self.spacing[1] * tile[1],
            self.spacing[0] * (tile[0] + 1),
            self.spacing[1] * (tile[1] + 1),
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
                                           self.raster_size[0],
                                           self.raster_size[1],
                                           self.raster_count,
                                           self.data_type,
                                           get_options(self.block_size))
        # config
        dataset.SetProjection(projections.get_wkt(self.projection))
        dataset.SetGeoTransform(self.tile2geotransform(tile))
        for i in range(self.raster_count):
            dataset.GetRasterBand(i + 1).SetNoDataValue(self.no_data_value)

    def get_tiles(self, dataset):
        """
        Return generator of tile indices.
        """
        outline = dataset2outline(dataset=dataset, projection=self.projection)
        return get_tiles(spacing=self.spacing, extent=outline.extent)

    def get_tiles_and_blocks(self, dataset):
        """ Return generator of tile and block indices. """
        tile_spacing = self.spacing
        dataset_extent = dataset2outline(dataset=dataset,
                                         projection=self.projection).extent

        for tile in get_tiles(spacing=tile_spacing, extent=dataset_extent):
            tile_extent = self.tile2extent(tile)
            intersection_extent = utils.get_extent_intersection(
                extent1=dataset_extent, extent2=tile_extent,
            )
            shifted_extent = (
                intersection_extent[0] - tile_extent[0],
                intersection_extent[1] - tile_extent[3],
                intersection_extent[2] - tile_extent[0],
                intersection_extent[3] - tile_extent[3],
            )
            block_spacing = tuple(c * b for c, b in zip(self.cell_size,
                                                        self.block_size))
            blocks = tuple(get_tiles(spacing=block_spacing,
                                     extent=shifted_extent))
            yield tile, blocks

    def get_paths(self, dataset):
        """
        Return path generator.

        It is guaranteed that a dataset exists at every path in the
        generator.
        """
        for tile, blocks in self.get_tiles_and_blocks(dataset):
            path = self.tile2path(tile)
            if not os.path.exists(path):
                logger.debug('Create {}'.format(path))
                self.create(tile)
            else:
                logger.debug('Update {}'.format(path))
            yield path, blocks

    def get_datasets(self, dataset):
        """
        Return read-only dataset generator.

        The generator yields only datasets whose extent intersect with
        datasets extent.
        """
        paths = (
            self.tile2path(tile) for tile in self.get_tiles(dataset))
        for path in paths:
            try:
                yield gdal.Open(path)
            except RuntimeError:
                continue

    def fetch_single_point(self, x, y):
        tile = get_tile(self.spacing, (x, y))
        path = self.tile2path(tile)
        try:
            logger.debug("Fetching single point: {}".format(path))
            dataset = gdal.Open(path)
        except RuntimeError:
            return [None]  #* self.rastercount

        values = point_from_dataset(dataset, (x, y))
        return np.ma.masked_equal(values, self.no_data_value).tolist()

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

        # Determine config first toplevel dataset.
        topleveldatasets = self.get_datasets(-1)
        self.__dict__.update(get_config(topleveldatasets.next()))

    def __getitem__(self, level):
        """ Return store corresponding to level. """
        cell_size = tuple(2 * [2 ** level])
        path = os.path.join(self.path, str(level))
        kwargs = dict(
            block_size=self.block_size,
            cell_size=cell_size,
            data_type=self.data_type,
            no_data_value=self.no_data_value,
            path=path,
            projection=self.projection,
            raster_count=self.raster_count,
            raster_size=self.raster_size,
        )
        return Grid(**kwargs)

    @property
    def peakpath(self):
        return os.path.join(self.path, b'.pyramid.tif')

    @property
    def extent(self):
        """ Return pyramid extent tuple. """
        try:
            return dataset2outline(gdal.Open(self.peakpath)).extent
        except RuntimeError:
            return None

    def bootstrap(self, dataset, overrides):
        """ Bootstrap manager for a new pyramid. """
        self.__dict__.update(get_config(dataset))
        self.__dict__.update(overrides)
        lowest_level = self.get_level(dataset)
        highest_level = max(self.get_toplevel(dataset), lowest_level)
        self.levels = range(lowest_level, highest_level + 1)

    def sync(self):
        """
        In the flooding lib branch: Only make sure that the peakpath
        exists by creating it as an empty file. The .pyramid.tif is
        needed so that the rasterserver demo page (flooding branch) can
        tell where a pyramid directory structure starts.
        """
        peakpath = self.peakpath
        if not os.path.exists(peakpath):
            open(peakpath, 'w')
        return

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
        pixelsize = max(d / b for d, b in zip(combinedsize, self.block_size))
        return int(math.ceil(math.log(pixelsize)))

    def get_bottomlevel(self):
        """Return the lowest level."""
        return self[self.levels[0]]

    def get_datasets(self, index):
        """
        Return all datasets from a path.

        Note that it assumes a two-deep directory structure for the level.
        """
        level = self.levels[index]
        paths = glob.iglob(os.path.join(
            self.path, str(level), b'*', b'*',
        ))
        for path in paths:
            yield gdal.Open(path)

    def extend(self, levels):
        """
        Extend levels to include level.

        Used if there is data in the pyramid, but the amount of levels
        need to be extended.
        """
        addlevels = [1 + self.levels[-1] + l for l in range(levels)]
        for source in self.get_datasets(-1):
            transport = rasters.SharedMemoryDataset(source)
            paths = (p for l in addlevels for p in self[l].get_paths(source))
            for path in paths:
                initialize(transport)
                map(warp, paths)
        self.levels.extend(addlevels)

    def add(self, dataset, **kwargs):
        """ Add a dataset to manager. """
        # prepare for data addition
        if self.levels:
            oldlevel = self.levels[-1]
            newlevel = self.get_toplevel(dataset)
            if newlevel > oldlevel:
                self.extend(newlevel)
        else:
            self.bootstrap(dataset=dataset, overrides=kwargs)

        # do the data addition
        transport = rasters.SharedMemoryDataset(dataset)
        paths = (p for l in self.levels for p in self[l].get_paths(dataset))

        # Single process addition
        #initialize(transport)
        #map(warp, paths)

        # Multiprocessed addition
        pool = multiprocessing.Pool(
            initializer=initialize, initargs=[transport]
        )
        pool.map(warp, paths)
        pool.close()

    def warpinto(self, dataset):
        """
        Return true if anything was warped, false otherwise.

        Warp tiles from appropriate store into dataset.
        """
        level = self.get_level(dataset)
        if level > self.levels[-1]:
            level = self.levels[-1]
        if level < self.levels[0]:
            level = self.levels[0]
        return self[level].warpinto(dataset)

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
        return self._manager['manager']

    @property
    def projection(self):
        return self.manager.projection

    def fetch_single_point(self, x, y):
        """X and Y are in the pyramid's projection."""
        grid = self.manager.get_bottomlevel()
        return grid.fetch_single_point(x, y)

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
        nodatavalue, tile_size. Kwargs are ignored if data already exists
        in the pyramid.
        """
        self.lock()
        manager = Manager(self.path)  # do not use the cached manager
        manager.add(dataset, **kwargs)
        self.unlock()

        if sync:
            manager.sync()

    def warpinto(self, dataset):
        """ See manager. """
        self.manager.warpinto(dataset)

    def sync(self):
        return self.manager.sync()

    @property
    def extent(self):
        """ Return pyramid extent tuple. """
        return self.manager.extent

    #def single(self, point):
        #""" Return value from lowest level. """
        #pass
