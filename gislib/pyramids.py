# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import datetime
import glob
import logging
import math
import os
import time
import shutil

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
    xul, dxx, dxy, yul, dyx, dyy = dataset.GetGeoTransform()
    points = ((xul, yul),
              (xul + dxx, yul + dyx),
              (xul + dxx + dxy, yul + dyx + dyy),
              (xul + dxy, yul + dyy),
              (xul, yul))
    return vectors.points2polygon(points)


def dataset2extent(dataset, projection=None):
    """ 
    Get extent of a dataset. 

    If projection is given, transform extent first.
    """
    if projection is None:
        # quick lookup
        p, a, b, q, c, d = dataset.GetGeoTransform()
        w, h = dataset.RasterXSize, dataset.RasterYSize
        extent = np.dot([[a, b],[c, d]], [[0, w], [0, h]]) + [[p], [q]]
        return tuple(result.ravel())
    else:
        # full transform
        outline = dataset2outlinepolygon(dataset)
        transformation = projections.get_coordinate_transformation(
            source=dataset.GetProjection, target=projection,
        )
        outline.Transform(transformation)
        return vector.Geometry(outline).extent


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
    return dict(
        datatype=band.DataType,
        bands=dataset.RasterCount,
        blocksize=band.GetBlockSize(),
        nodatavalue=band.GetNoDataValue(),
        projection=dataset.GetProjection(),
        tilesize=(dataset.RasterYSize, dataset.RasterXSize),
        #cellsize=(lambda g:(x[1], -x[5]))(dataset.GetGeoTransform()),
    )




def get_tiles(spacing, extent):
    """ 
    Return tile iterator.
    """
    # Determine the ranges for the tiles
    x_range, y_range = map(
        xrange,
        (int(math.floor((e) / s))
         for e, s in zip(extent[:2], spacing)),
        (int(math.ceil((e) / s))
         for e, s in zip(extent[2:], spacing)),
    )

    # Iterate over the ranges
    for y in y_range:
        for x in x_ranges:
            yield x, y


def get_parent(tile):
    """ Return tile. """
    return get_tiles(tilesize=tile.size,
                     level=tile.level + 1,
                     extent=tile.extent).next()


def get_top_tile(geometry, tilesize, blocksize):
    """
    Get the first tile for which a block completely contains geometry.
    """
    # Determine at which level it would fit
    envelopesize = geometry2envelopesize(geometry)
    size = tuple(e / b for e, b in zip(envelopesize, blocksize))
    level = int(math.floor(math.log(min(size), 2)))

    # Find intersecting tile at that level
    point = geometry.Centroid().GetPoint(0)
    tiles = tuple(int(math.floor(p / (2 ** level * t)))
                    for t, p in zip(tilesize, point))
    tile = Tile(size=tilesize, level=level, tiles=tiles)

    # Get higher tiles until tile contains geometry
    while not tile.polygon.Contains(geometry):
        tile = get_parent(tile)

    return tile

class LockError(Exception):
    pass



class SingleDatasetPyramid(object):
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
        level2area = lambda l: dataset2pixelpolygon(l.dataset).GetArea()
        self.areas = [level2area(l) for l in self.levels]

    @property
    def projection(self):
        return self.levels[0].dataset.GetProjection()

    def get_level(self, dataset):
        """ Return the appropriate level for reprojection into dataset. """
        area = get_bounds(dataset=dataset,
                          projection=self.projection)['pixel'].GetArea()

        for a, l in reversed(zip(self.areas, self.levels)):
            if a < area:
                return l
            return self.levels[0]
            
    def warpinto(self, dataset):
        """
        Reproject appropriate level from self into dataset.
        """
        reproject(self.get_level(dataset).dataset, dataset)


class TileStore(object):
    """
    path
    bands
    cellsize
    tilesize
    blocksize
    projection
    nodatavalue

    Possible to initialize from a path, or a file, but also from a dictionary
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
    

    def get_extent(self, dataset):
        """ Return extent of a dataset in our projection. """
        transformation = projections.get_coordinate_transformation(
            dataset.projection, self.projection,
        )
        polygon = dataset2outlinepolygon(dataset)
        polygon.Transform(transformation)
        return vectors.Geometry(polygon).extent
    
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
        tiles = get_tiles(spacing=self.spacing,
                          extent=self.extent(dataset))

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
        logger.debug('Create {}'.format(path))
        path = self.get_path(tile)
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
        dataset.SetGeoTransform(self.get_geotransform(tile)
        for i in range(self.bands):
            dataset.GetRasterBand(i + 1).SetNoDataValue(self.nodatavalue)

    def warpinto(dataset):
        """ Warp appropriate tiles into dataset. """
        for source in self.get_datasets(dataset):
            reproject(source, dataset)

class PyramidInfo(object):
    """ 
    Container for pyramid info
    """
    def __init__(self, template, levels):
        """ Template is a dataset, levels a list of integers. """
        self.config = get_config(dataset=template)
        self.levels = levels
        for k, v in get_config(dataset):
            setattr(self, k, v)

    def get_level(self, dataset):
        """ Return appropriate level for dataset. """
        pixel = dataset2pixelpolygon(dataset)
        transformation = projection.get_coordinate_transformation(
            source=dataset.GetProjection(), target=self.projection,
        )
        pixel.Transform(transformation)
        span = geometry2envelopesize(pixel)
        return int(math.floor(math.log(min(span), 2)))

    def level2store(self, level):
        """ Return store corresponding to level. """
        cellsize = (2 ** level)
        config = dict(cellsize=cellsize)
        config = self.config
        return TileStore(


    def get_store(self, dataset)
        """ Instantiate the appropriate store for warping into dataset. """
        level = max(levels[0], self.get_level(dataset))
        if level > levels[-1]:
            return None
        else:
            return 
            
        

    def get_stores(levels):
        """ Return store generator for levels, for writing purposes. """
        for l in levels:
            
            

    



class Pyramid(stores.BaseStore):
    """
    Pyramid datastore. Physically consisting of a number of tilestores
    and a top.
    """

    def __init__(self, path):   
        """
        Initialize.

        The idea is to initialize almost nothing, so that any other
        instances will always be up-to-date.
        """
        self.path = path

    @property
    def actual_stores(self):
        """

        """
        # See if any contents in the pyramid
        try:
            names = os.listdir(self.path)
        except OSError:
            return []
        levels = [int(n) for n in names if not name.startswith('.')]
        if not levels:
            return []

        # Derive properties from a sample dataset in the pyramid
        samplelevel = str(levels.max())
        samplepath = glob.glob(os.path.join(path, highest, '*', '*'))[0]
        baseconfig = get_config(gdal.Open(samplepath))

        # Populate a list of tilestore objects
        stores = []
        for l in levels:
            cellsize = 2 ** l,
            config = dict(cellsize=cellsize)
            config.update(baseconfig)
            stores.append(TileStore(config))

        return stores

    @property
    def stores(self):
        """ Cached version of actual_stores. """
        now = time.time()
        if not hasattr(self, '_stores') or now - self._stores['time'] > TIMEOUT:
            self._stores = dict(time=now, stores=self.actual_stores)
        return self._stores['stores']

    @property
    def extent(self):
        """ Return extent of top. """
        return dataset2extent(gdal.Open(self.toppath))

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
        (x1, x2), (y2, y1) = np.dot([[a, b],[c, d]], pixels) + p, q
        return x1, y1, x2, y2

    # =========================================================================
    # Interface
    # -------------------------------------------------------------------------

    def add(self, dataset=None, sync=True, **kwargs):
        """
        If there is no dataset, check if locked and reload.

        Any kwargs are used to override dataset projection and datatype,
        nodatavalue, tilesize. Kwargs are ignored if data already exists
        in the pyramid.

        Sync indicates wether to build a synced peak on top of the pyramid.
        """
        self.lock()

        # get actual info from stores. otherwise use dataset and kwargs
        stores = self.actual_stores
        if config is None:
            config = get_config(dataset)
            info.update(kwargs)

        # get bounds in pyramids projection
        bounds = get_bounds(dataset=dataset, projection=info['projection'])

        # find new top tile
        top_tile = get_top_tile(geometry=bounds['raster'],
                                tilesize=info['tilesize'],
                                blocksize=info['blocksize'])



        # walk and reproject
        tiles = walk_tiles(tile=top_tile,
                           stop=min_level,
                           geometry=bounds['raster'])

        children = collections.defaultdict(list)
        previous_level = min_level
        for tile in tiles:
            # Get the dataset
            target = self.get_dataset(tile=tile, info=info)

            # To aggregate or not
            if tile.level == previous_level + 1:
                while children[previous_level]:
                    rasters.reproject(
                        children[previous_level].pop(), target
                    )
            else:
                rasters.reproject(dataset, target)
            target = None  # Writes the header

            children[tile.level].append(self.get_dataset(tile))
            previous_level = tile.level

        # Update peak
        if sync:
            self.topsync()

        self.unlock()

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
