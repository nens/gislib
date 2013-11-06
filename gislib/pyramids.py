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

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)

GDAL_DRIVER_GTIFF = gdal.GetDriverByName(b'gtiff')
TIMEOUT = 60  # seconds


def initialize(arg_transport):
    """ Set some pool data globally. """
    global transport
    transport = arg_transport


def warp(path_and_blocks):
    """ Warp global transport into specified blocks from dataset at path. """
    path, blocks = path_and_blocks

    # Old version without the blocks
    target = gdal.Open(path, gdal.GA_Update)
    transport.warpinto(target)
    
    # After implementing read_block and write_block, to be replaced by:
    #for i, j in blocks:
        #dataset = rasters.Dataset(gdal.Open(path, gdal.GA_Update))
        #block = dataset.read_block((i, -j + 1))
        #transport.warpinto(block['dataset'])
        #block['dataset'].FlushCache()
        #dataset.write_block((i, -j), block['array'])


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
        self.projection = dataset.GetProjection()
        self.projection = dataset.GetProjection()
        self.levels = [rasters.SharedMemoryDataset(dataset)]
        while max(self.levels[-1].array.shape[1:]) > stop:
            self.levels.append(rasters.SharedMemoryDataset(
                dataset=self.levels[-1].dataset, shrink=2,
            ))

        self.pixelsizes = tuple(max(dataset2pixel(l.dataset).size) 
                                for l in self.levels)
        # Points for numpy interpolation, used for get_level method.
        self.points = zip(*tuple((s, i + o) 
                                 for i, s in enumerate(self.pixelsizes[1:]) 
                                 for o in (0, 1)))

    def get_level(self, dataset):
        """ Return the appropriate level for reprojection into dataset. """
        pixelsize = min(dataset2pixel(dataset=dataset,
                                      projection=self.projection).size)
        return self.levels[int(np.interp(pixelsize, *self.points))]

    def warpinto(self, dataset):
        """
        Reproject appropriate level from self into dataset.
        """
        rasters.reproject(self.get_level(dataset).dataset, dataset)


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
        original_extent = dataset2outline(dataset=dataset,
                                          projection=self.projection).extent
        x1, y1, x2, y2 = original_extent                                          
        for tile in get_tiles(spacing=tile_spacing, extent=original_extent):
            left, top = (s * (t + i)
                         for s, t, i in zip(tile_spacing, tile, (0,1)))
            shifted_extent = x1 - left, y1 - top, x2 - left, y2 - top
            block_spacing = tuple(c * b for c, b in zip(self.cell_size,
                                                        self.block_size))
            blocks = get_tiles(spacing=block_spacing, extent=shifted_extent)
            yield tile, tuple(blocks)


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
        self.__dict__.update(get_config(dataset))
        self.__dict__.update(overrides)
        self.levels = range(self.get_level(dataset),
                            self.get_toplevel(dataset) + 1)

    def sync(self):
        """
        Create or replace the peak with current data.
        """
        cropped = []
        for dataset in self.get_datasets(-1):
            cropped.append(crop(dataset))
        extents = [dataset2outline(c).extent for c in cropped]
        x1, y1, x2, y2 = zip(*extents)
        x1, y1, x2, y2 = min(x1), min(y1), max(x2), max(y2)
        geotransform = x1, (x2 - x1) / 256, 0, y2, 0, (y1 - y2) / 256
        
        # create
        fd, temppath = tempfile.mkstemp(dir=self.path, prefix='.pyramid.tmp.')
        dataset = GDAL_DRIVER_GTIFF.Create(temppath,
                                           256,
                                           256,
                                           self.raster_count,
                                           self.data_type,
                                           get_options(block_size=(256, 256)))
        dataset.SetProjection(projections.get_wkt(self.projection))
        dataset.SetGeoTransform(geotransform)
        for i in range(self.raster_count):
            dataset.GetRasterBand(i + 1).SetNoDataValue(self.no_data_value)
        for c in cropped:
            rasters.reproject(c, dataset)

        dataset = None
        os.close(fd)
        os.rename(temppath, self.peakpath)

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
        
    def get_datasets(self, index):
        """ 
        Return all datasets from a path.
        
        Note that it assumes a two-deep directory structure for the level.
        """
        level = self.levels[index]
        paths = glob.iglob(os.path.join(
            self.path, str(level), '*', '*',
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
            transport = Transport(source)
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
        transport = Transport(dataset)
        paths = (p for l in self.levels for p in self[l].get_paths(dataset))
        
        #initialize(transport)
        #map(warp, paths)
        pool = multiprocessing.Pool(initializer=initialize, initargs=[transport])
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

    #def single(self, point):
        #""" Return value from lowest level. """
        #pass
