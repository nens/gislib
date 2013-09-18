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
from gislib import vectors

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)

GTIFF = gdal.GetDriverByName(b'gtiff')
MEM = gdal.GetDriverByName(b'mem')
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


def extent2polygon(xmin, ymin, xmax, ymax):
    """ Return an extent polygon. """
    points = ((xmin, ymin),
              (xmax, ymin),
              (xmax, ymax),
              (xmin, ymax),
              (xmin, ymin))
    return vectors.points2polygon(points)


def geometry2envelopepoints(geometry):
    """ Return array. """
    return np.array(geometry.GetEnvelope()).reshape(2, 2).transpose()


def geometry2envelopeextent(geometry):
    """ Return extent. """
    return tuple(geometry2envelopepoints(geometry).ravel())


def geometry2envelopesize(geometry):
    """ Return size tuple. """
    xmin, xmax, ymin, ymax = geometry.GetEnvelope()
    return xmax - xmin, ymax - ymin


def get_info(dataset):
    """ Return dictionary. """
    band = dataset.GetRasterBand(1)
    return dict(datatype=band.DataType,
                blocksize=band.GetBlockSize(),
                nodatavalue=band.GetNoDataValue(),
                projection=dataset.GetProjection(),
                tilesize=(dataset.RasterXSize, dataset.RasterYSize))


def get_bounds(dataset, projection):
    """
    Return dictionary.

    Dictionary contains the boundaries of the dataset and its first pixel,
    transformed to projection when necessary.

    Transformation is considered unnecessary if the outline changes
    less then 1 % of the pixelsize by transformation.

    The outline is shrunk by 1% of the shortest pixelside, to prevent
    edge intersections.
    """
    transformation = osr.CoordinateTransformation(
        projections.get_spatial_reference(dataset.GetProjection()),
        projections.get_spatial_reference(projection),
    )

    # outline
    raster_org = dataset2outlinepolygon(dataset)
    raster_trf = raster_org.Clone()
    raster_trf.Transform(transformation)

    # pixel
    pixel_org = dataset2pixelpolygon(dataset)
    pixel_trf = pixel_org.Clone()
    pixel_trf.Transform(transformation)

    # verdict
    pixel_trf_size = geometry2envelopesize(pixel_trf)
    diff = np.abs(geometry2envelopepoints(raster_trf) -
                  geometry2envelopepoints(raster_org))
    transform = (100 * diff > pixel_trf_size).any()

    # return
    if transform:
        pixel = pixel_trf
        raster = raster_trf.Buffer(-0.01 * min(pixel_trf_size))
    else:
        pixel = pixel_org
        raster = raster_org.Buffer(-0.01 * min(pixel_trf_size))

    return dict(raster=raster, pixel=pixel)


def get_level(pixel):
    """
    Get the minimum level for a pixel polygon.
    """
    size = geometry2envelopesize(pixel)
    return int(math.floor(math.log(min(size), 2)))


def get_tiles(tilesize, level, extent):
    """ Return tile iterator. """
    # tilesize are pixels, tilespan are projection units
    tilespan = tuple(2 ** level * t for t in tilesize)

    # Determine the ranges for the indices
    tiles_x, tiles_y = map(
        xrange,
        (int(math.floor((e) / t))
         for e, t in zip(extent[:2], tilespan)),
        (int(math.ceil((e) / t))
         for e, t in zip(extent[2:], tilespan)),
    )

    # Iterate over the ranges
    for tile_y in tiles_y:
        for tile_x in tiles_x:
            yield Tile(size=tilesize, level=level, indices=(tile_x, tile_y))


def walk_tiles(tile, stop, geometry):
    """
    Return generator of tiles at lower levels that intersect
    with geometry, stopping at level stop.
    """
    if not geometry.Intersects(tile.polygon):
        return  # subtiles will not intersect either
    if tile.level > stop:
        # walk subtiles, too
        subtiles = get_tiles(tilesize=tile.size,
                             level=tile.level - 1,
                             extent=tile.extent)
        for subtile in subtiles:
            walktiles = walk_tiles(tile=subtile,
                                   stop=stop,
                                   geometry=geometry)
            for walktile in walktiles:
                yield walktile
    # finally, yield this tile
    yield tile


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
    indices = tuple(int(math.floor(p / (2 ** level * t)))
                    for t, p in zip(tilesize, point))
    tile = Tile(size=tilesize, level=level, indices=indices)

    # Get higher tiles until tile contains geometry
    while not tile.polygon.Contains(geometry):
        tile = get_parent(tile)

    return tile


def scale_geotransform(geotransform, factor):
    """ Return geotransform scaled by a factor. """
    return(geotransform[0],
           geotransform[1] * factor,
           geotransform[2] * factor,
           geotransform[3],
           geotransform[4] * factor,
           geotransform[5] * factor)


class LockError(Exception):
    pass


class LevelError(Exception):
    pass


class Tile(object):
    def __init__(self, size, level, indices):
        """ x, y, and t refer to block indices. """
        self.size = size
        self.level = level
        self.indices = indices
        self.path = os.path.join(str(level),
                                 str(indices[0]),
                                 str(indices[1]) + '.tif')

    def __str__(self):
        return '<Tile: size {}, level {}, indices {}>'.format(
            self.size, self.level, self.indices,
        )

    def __repr__(self):
        return str(self)

    @property
    def extent(self):
        """
        Return extent tuple.
        """
        return tuple(2 ** self.level *
                     self.size[i] *
                     (self.indices[i] + j)
                     for j in (0, 1) for i in (0, 1))

    @property
    def geotransform(self):
        """ Return geotransform tuple. """
        pixelsize = 2 ** self.level
        xmin, ymin, xmax, ymax = self.extent
        return xmin, pixelsize, 0, ymax, 0, -pixelsize

    @property
    def polygon(self):
        """ Return extent geometry. """
        return extent2polygon(*self.extent)


class Pyramid(object):
    """
    Pyramid datastore.
    """

    TILES = 'tiles'
    PEAK = 'peak'

    def __init__(self, path):
        """
        Initialize.

        The idea is to initialize almost nothing, so that any other
        instances will always be up-to-date.
        """
        self.path = path

    def get_options(self, blocksize):
        """ Return blocksize dependent gtiff creation options. """
        return [
            'BLOCKXSIZE={}'.format(blocksize[0]),
            'BLOCKYSIZE={}'.format(blocksize[1]),
            'COMPRESS=DEFLATE',
            'SPARSE_OK=TRUE',
            'TILED=TRUE',
        ]

    @property
    def info(self):
        """
        Return info dictionary or None if empty pyramid.
        """
        # See if any contents in the pyramid
        path = os.path.join(self.path, self.TILES)
        try:
            levels = os.listdir(path)
        except OSError:
            return
        if not levels:
            return

        # Derive extreme levels and top tile properties from filesystem
        top_path = glob.glob(os.path.join(path,
                                          max(levels, key=int), '*', '*'))[0]

        # Start with info from top tile dataset
        info = get_info(gdal.Open(top_path))

        # Update with info from folder structure
        min_level = int(min(levels, key=int))
        max_level, x, y = map(int, top_path[:-4].split(os.path.sep)[-3:])
        top_tile = Tile(size=info['tilesize'], level=max_level, indices=(x, y))

        info.update(
            max_level=max_level,
            min_level=min_level,
            top_tile=top_tile,
        )

        # Add a peak if this is not the peak
        if not self.path.endswith(self.PEAK):
            info.update(peak=Pyramid(os.path.join(self.path, self.PEAK)))

        return info

    @property
    def infocache(self):
        now = time.time()
        if not hasattr(self, '_info') or now - self._info['time'] > TIMEOUT:
            self._info = dict(time=now, info=self.info)
        return self._info['info']

    @property
    def extent(self):
        """ Return the extent of the datapart in peaks top tile. """
        peak = self.infocache['peak']
        dataset = peak.get_dataset(peak.infocache['top_tile'])

        # Return geotransform as matrices
        geotransform = np.array(dataset.GetGeoTransform())
        matrix = geotransform[[1, 2, 4, 5]].reshape(2, 2)
        offset = geotransform[[0, 3]].reshape(2, 1)

        # Determine the pixels that are not nodata
        array = dataset.GetRasterBand(1).GetMaskBand().ReadAsArray()
        pixels = np.array(tuple((i.min(), i.max() + 1)
                          for i in array.nonzero()))[::-1]

        # Return as extent
        (x1, x2), (y2, y1) = np.dot(matrix, pixels) + offset
        return x1, y1, x2, y2

    # =========================================================================
    # Locking
    # -------------------------------------------------------------------------

    @property
    def lockpath(self):
        return os.path.join(self.path, 'lock')

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
    # Datasets
    # -------------------------------------------------------------------------

    def new_dataset(self, tile, info):
        """ Return gdal dataset. """
        path = os.path.join(self.path, self.TILES, tile.path)

        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass  # It existed.

        create_options = self.get_options(info['blocksize'])
        dataset = GTIFF.Create(path, tile.size[0], tile.size[1],
                               1, info['datatype'], create_options)

        dataset.SetProjection(projections.get_wkt(info['projection']))
        dataset.SetGeoTransform(tile.geotransform)
        dataset.GetRasterBand(1).SetNoDataValue(info['nodatavalue'])

        return dataset

    def get_dataset(self, tile, info=None):
        """
        Return a gdal dataset corresponding to a tile.

        Adding an info dictionary implies write mode.

        If the dataset does not exist, in read mode a RuntimeError is
        raised; in write mode a new dataset is created.
        """
        path = os.path.join(self.path, self.TILES, tile.path)

        if info is None:
            return gdal.Open(path)

        try:
            dataset = gdal.Open(path, gdal.GA_Update)
            logger.debug('Update {}'.format(path))
        except RuntimeError:
            dataset = self.new_dataset(tile=tile, info=info)
            logger.debug('Create {}'.format(path))
        return dataset

    def get_datasets(self, tiles):
        """
        Return read-only dataset generator.

        Silently skips when dataset does not exist on disk.
        """
        for tile in tiles:
            try:
                yield self.get_dataset(tile=tile)
            except RuntimeError:
                continue

    def get_promoted(self, tile, info):
        """
        Return parent tile.

        Reproject tiles dataset into parent dataset.
        """
        parent = get_parent(tile)
        source = self.get_dataset(tile)
        target = self.get_dataset(tile=parent, info=info)
        rasters.reproject(source, target)
        return parent

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

        if dataset is None:
            # unlock and return
            return self.unlock()

        # use pyramid info if possible, otherwise use dataset and kwargs
        info = self.info
        if info is None:
            info = get_info(dataset)
            info.update(kwargs)
            if not self.path.endswith(self.PEAK):
                info.update(peak=Pyramid(os.path.join(self.path, self.PEAK)))

        # get bounds in pyramids projection
        bounds = get_bounds(dataset=dataset, projection=info['projection'])

        # derive baselevel
        min_level = get_level(bounds['pixel'])
        if min_level != info.get('min_level', min_level):
            raise LevelError('Incompatible resolution.')

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
            # Get the data
            target = self.get_dataset(tile=tile, info=info)

            # To aggregate or not
            if tile.level == previous_level + 1:
                while children[previous_level]:
                    rasters.reproject(children[previous_level].pop(), target)
            else:
                rasters.reproject(dataset, target)
            target = None  # Writes the header

            children[tile.level].append(self.get_dataset(tile))
            previous_level = tile.level

        # sync old and new toptiles
        lo, hi = top_tile, info.get('top_tile', top_tile)
        if hi.level < lo.level:
            hi, lo = lo, hi  # swap
        while lo.level < hi.level:
            lo = self.get_promoted(tile=lo, info=info)
        while lo.indices != hi.indices:
            lo = self.get_promoted(tile=lo, info=info)
            hi = self.get_promoted(tile=hi, info=info)

        # Update peak
        if sync:
            self.sync()

        self.unlock()

    def sync(self):
        """
        Update a peak pyramid.

        Pyramids with large tilesizes warp slow at high zoomlevels. The
        use of a peak pyramid with a small tilesize on top of the main
        pyramid solves that.
        """
        path = os.path.join(self.path, self.PEAK)
        info = self.info
        if os.path.exists(path):
            shutil.rmtree(path)
        peak = Pyramid(path)
        dataset = self.get_dataset(info['top_tile'])

        # Base of peak is one higher than tile
        base = MEM.Create('',
                          info['tilesize'][0] // 2,
                          info['tilesize'][1] // 2,
                          1,
                          info['datatype'])
        base.SetProjection(info['projection'])
        base.GetRasterBand(1).SetNoDataValue(info['nodatavalue'])
        base.SetGeoTransform(
            scale_geotransform(dataset.GetGeoTransform(), 2),
        )
        base.GetRasterBand(1).Fill(info['nodatavalue'])
        rasters.reproject(dataset, base)
        peak.add(base, sync=False, blocksize=(256, 256), tilesize=(256, 256))

    def warpinto(self, dataset):
        """ Warp data from the pyramid into dataset. """
        # if no pyramid info, pyramid is empty.
        info = self.infocache
        if info is None:
            return

        # get bounds in pyramids projection
        bounds = get_bounds(dataset=dataset, projection=info['projection'])
        level = max(info['min_level'], get_level(bounds['pixel']))

        # warp from peak if appropriate
        if level > info['max_level'] and 'peak' in info:
            return info['peak'].warpinto(dataset)

        # This is the top of the main pyramid (level == max_level) or the peak
        if level >= info['max_level']:
            tiles = info['top_tile'],
        else:
            tiles = get_tiles(tilesize=info['tilesize'],
                              level=level,
                              extent=geometry2envelopeextent(bounds['raster']))
        for source in self.get_datasets(tiles):
            rasters.reproject(source, dataset)
