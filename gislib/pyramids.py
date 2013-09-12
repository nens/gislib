# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import datetime
import glob
import hashlib
import logging
import math
import os
#import pickle

from osgeo import gdal
from osgeo import gdal_array
from osgeo import ogr
from osgeo import osr

import numpy as np

from gislib import projections
from gislib import rasters
from gislib.store import storages

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

logger = logging.getLogger(__name__)


def array2dataset(array):
    """ 
    Return gdal dataset. 
    
    Fastest way to get a gdal dataset from a numpy array,
    but keep a refernce to the array around, or a segfault will
    occur. Also, don't forget to call FlushCache() on the dataset after
    any operation that affects the array.
    """
    datapointer = array.ctypes.data
    bands, lines, pixels = array.shape
    datatypecode = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype.type)
    datatype = gdal.GetDataTypeName(datatypecode)
    bandoffset, lineoffset, pixeloffset = array.strides

    dataset_name_template = (
        'MEM:::'
        'DATAPOINTER={datapointer},'
        'PIXELS={pixels},'
        'LINES={lines},'
        'BANDS={bands},'
        'DATATYPE={datatype},'
        'PIXELOFFSET={pixeloffset},'
        'LINEOFFSET={lineoffset},'
        'BANDOFFSET={bandoffset}'
    )
    dataset_name = dataset_name_template.format(
        datapointer=datapointer,
        pixels=pixels,
        lines=lines,
        bands=bands,
        datatype=datatype,
        pixeloffset=pixeloffset,
        lineoffset=lineoffset,
        bandoffset=bandoffset,
    )
    return gdal.Open(dataset_name, gdal.GA_Update)


def array2polygon(array):
    """
    Return a polygon geometry.

    This method numpy to prepare a wkb string. Seems only faster for
    larger polygons, compared to adding points individually.
    """
    # 13 bytes for the header, 16 bytes per point
    nbytes = 13 + 16 * array.shape[0]
    data = np.empty(nbytes, dtype=np.uint8)
    # little endian
    data[0:1] = 1
    # wkb type, number of rings, number of points
    data[1:13].view(np.uint32)[:] = (3, 1, array.shape[0])
    # set the points
    data[13:].view(np.float64)[:] = array.ravel()
    return ogr.CreateGeometryFromWkb(data.tostring())


def points2polygon(points):
    """
    Return a polygon geometry.

    Adds points individually. Faster for small amounts of points.
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in points:
        ring.AddPoint_2D(x, y)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon


def dataset2outlinepolygon(dataset):
    """ Return polygon formed by pixel edges. """
    nx, ny = dataset.RasterXSize, dataset.RasterYSize
    npoints = 2 * (nx + ny) + 1
    # Construct a wkb array for speed
    #nbytes = 13 + 16 * npoints
    #data = np.empty(nbytes, dtype=np.uint8)
    #data[0:1] = 1  # little endian
    ## wkb type, number of rings, number of points
    #data[1:13].view(np.uint32)[:] = [3, 1, npoints]
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

    return array2polygon(array)


def dataset2pixelpolygon(dataset):
    """ Return polygon corresponding to the first pixel of dataset. """
    xul, dxx, dxy, yul, dyx, dyy = dataset.GetGeoTransform()
    points = ((xul, yul),
              (xul + dxx, yul + dyx),
              (xul + dxx + dxy, yul + dyx + dyy),
              (xul + dxy, yul + dyy),
              (xul, yul))
    return points2polygon(points)


def extent2polygon(xmin, ymin, xmax, ymax):
    """ Return an extent polygon. """
    points = ((xmin, ymin),
              (xmax, ymin),
              (xmax, ymax),
              (xmin, ymax),
              (xmin, ymin))
    return points2polygon(points)


def geometry2envelopepolygon(geometry):
    """ Return the rectangular polygon of geometry's envelope. """
    xmin, xmax, ymin, ymax = geometry.GetEnvelope()
    return extent2polygon(xmin, ymin, xmax, ymax)


def geometry2envelopepoints(geometry):
    """ Return array. """
    return np.array(geometry.GetEnvelope()).reshape(2, 2).transpose()


def geometry2envelopesize(geometry):
    """ Return size tuple. """
    xmin, xmax, ymin, ymax = geometry.GetEnvelope()
    return xmax - xmin, ymax - ymin


def get_status(dataset):
    """ Return dictionary. """
    band = dataset.GetRasterBand(1)
    return dict(datatype=band.DataType,
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
    diff = (geometry2envelopepoints(raster_trf) -
            geometry2envelopepoints(raster_org))
    transform = (100 * diff > pixel_trf_size).any()

    # return
    if transform:
        pixel = pixel_trf
        raster = raster_trf.Buffer(-0.01 * min(pixel_trf_size))
    else:
        pixel = pixel_org
        raster = raster_org.Buffer(-0.01 * min(pixel_trf_size))
    
    return dict(dataset=raster, pixel=pixel)


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


def get_top_tile(geometry, tilesize, blocksize=(256, 256)):
    """ 
    Get the first tile for which a block completely contains geometry.    
    """
    # Determine at which level it would fit
    envelopesize = geometry2envelopesize(geometry)
    size = tuple(e / b for e, b in zip(envelopesize, blocksize))
    level = int(math.floor(math.log(min(size), 2)))

    # Find intersecting tile at that level
    point = geometry.Centroid().GetPoint(0)
    indices = tuple(int(math.floor(c / (2 ** level * b)))
                    for b, c in zip(blocksize, point))
    tile = Tile(size=size, level=level, indices=indices)

    # Get higher tiles until tile contains geometry
    while not tile.polygon.Contains(geometry):
        tile = get_tiles(tilesize=tilesize, 
                         level=tile.level + 1, extent=tile.extent).next()
    
    return tile


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
        return '<Location: size {}, level {}, indices {}>'.format(
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
        pixelsize = 2 ** address.level
        xmin, ymin, xmax, ymax = self.get_extent(address)
        return xmin, pixelsize, 0, ymax, 0, -pixelsize

    @property
    def polygon(self):
        """ Return extent geometry. """
        return extent2polygon(*self.extent)

    



class Pyramid():
    """
    Pyramid datastore.
    """

    OPTIONS = [
        'BLOCKXSIZE=256',
        'BLOCKYSIZE=256',
        'COMPRESS=DEFLATE',
        'SPARSE_OK=TRUE',
        'TILED=TRUE',
    ]

    def __init__(self, path):
        """
        Initialize.

        The idea is to initialize almost nothing, so that any other
        instances will always be up-to-date.
        """
        self.path = path

    @property
    def status(self):
        """ 
        Return status dictionary or None if empty pyramid.
        """
        logger.info('status accessed!')
        # See if any contents in the pyramid
        levels = glob.glob(os.path.join(self.path, '*'))
        if not levels:
            return
        
        # Derive extreme levels and top tile properties from filesystem
        top_path = glob.glob(os.path.join(self.path, max(levels), '*', '*'))[0]
        min_level = int(min(levels))
        max_level, x, y = map(int, top_path[:-4].split(os.path.sep)[-3:])
        top_tile = Tile(level=max_level, indices=(x,y))

        status = dict(
            max_level=max_level,
            min_level=min_level,
            top_tile=top_tile,
        )
        
        # Update with dataset specific properties from top tile dataset
        status.update(get_status(gdal.Open(top_path)))
        return status
    
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
    # Operations
    # -------------------------------------------------------------------------

    def add(self, dataset=None, **kwargs):
        """
        If there is no dataset, check if locked and reload.
        
        Any kwargs are used to override dataset projection and datatype,
        nodatavalue, tilesize. Kwargs are ignored if data already exists
        in the pyramid.
        """
        # lock
        self.lock()
        if dataset is None:
            # unlock and return
            return self.unlock()

        # use pyramid status if possible, otherwise use dataset and kwargs
        status = self.status
        if status is None:
            status = get_status(dataset)
            status.update(kwargs)
        

        # get bounds in pyramids projection
        bounds = get_bounds(dataset=dataset, projection=status['projection'])

        # check baselevel
        min_level = get_level(bounds['pixel'])
        if min_level != status.get('min_level', min_level):
            raise LevelError('Incompatible resolution.')

        # find new top tile
        top_tile = get_top_tile(geometry=bounds['dataset'], tilesize=status['tilesize'])
        
        # Walk and warp
        for tile

        # Bring old and new toplevels to same level
        # Level up until they are the same tile.
        self.unlock()



    def get_datasets(self, tiles):
        """ Return dataset generator. """
        return tile

    def get_tiles(self, level, geometry):
        """ Return tiles intersecting with geometry for a level. """

        


    
    def get_dataset(self, level, tile, mode='r'):
        """
        Return a gdal dataset.

        If the file corresponding to level and tile does not exist:
            In (r)ead mode, return mem dataset with nodata
            In (w)rite mode, create and return tif dataset with nodata
        """
        path = os.path.join(self.path, str(level), '{}_{}.tif'.format(*tile))
        if os.path.exists(path):
            # Open existing file with correct gdal access mode
            if mode == 'w':
                access = gdal.GA_Update
                logging.debug('Update {}'.format(path))
            else:
                access = gdal.GA_ReadOnly
            return gdal.Open(str(path), access)

        create_args = [str(path),
                       self.tilesize[0],
                       self.tilesize[1],
                       1,
                       self.datatype,
                       ['TILED=YES',
                        'COMPRESS={}'.format(self.compression)]]

        if mode == 'w':
            # Use gtiff driver
            driver = gdal.GetDriverByName(b'gtiff')
            logging.debug('Create {}'.format(path))

            # Create directory if necessary
            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass  # It existed.
        else:  # mode == 'r'
            # Use mem driver
            driver = gdal.GetDriverByName(b'mem')
            create_args.pop()  # No compression for mem driver

        # Actual create
        dataset = driver.Create(*create_args)
        dataset.SetProjection(
            projections.get_spatial_reference(self.projection).ExportToWkt(),
        )
        dataset.SetGeoTransform(
            self._geometry(level=level, tile=tile).geotransform(),
        )
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(self.nodatavalue)
        band.Fill(self.nodatavalue)
        return dataset

    def get_tile_dataset(self):
        pass


    def get_topaddress(self, bbox, index=0):
        """ Return the toplevel address. """
        point = bbox.Centroid().GetPoint(0)
        level = self.maxlevel
        tile = tuple(int(math.floor(c / (2 ** level * t)))
                     for t, c in zip(self.tilesize, point))
        return Address(index=index, level=level, tile=tile)


    def walk(self, address, polygon, level):
            """
            Return generator of addresses at lower levels that intersect
            with polygon, stopping at level.
            """
            if not self.get_polygon(address).Intersects(polygon):
                return
            if address.level > level:
                subaddresses = self.iteraddresses(address=address,
                                                  level=address.level - 1)
                for subaddress in subaddresses:
                    walkaddresses = self.walk(level=level,
                                              polygon=polygon,
                                              address=subaddress)
                    for walkaddress in walkaddresses:
                        yield walkaddress
            yield address

    # =========================================================================
    # Storage
    # -------------------------------------------------------------------------

    def get_empty(self):
        """ Return empty numpy array. """
        return self.nodata * np.ones(self.shape, self.dtype)

    def get_array(self, address):
        """ Return numpy array. """
        data = self.data[address.key]
        return np.fromstring(data, self.dtype).reshape(self.shape)

    def put_array(self, address, array):
        """ Store numpy array. """
        self.data[address.key] = array.tostring()

    def pgn_tile(self, address, tile):
        """
        Configure tile dataset according to self and address.

        Sets (p)rojection, (g)eotransform and (n)odata.
        """
        tile.SetProjection(projections.get_wkt(self.projection))
        tile.SetGeoTransform(self.get_geotransform(address))
        if np.dtype(self.dtype).kind == 'f':
            nodata = float(self.nodata)
        else:
            nodata = int(self.nodata)
        tile.GetRasterBand(1).SetNoDataValue(nodata)

    def get_tile(self, address):
        """
        Return container namedtuple,

        The tuple contains a numpy array and a writable gdal dataset,
        sharing the same memory. Keep a reference to the array,
        or cause a segfault...
        """
        try:
            array = self.get_array(address)
        except KeyError:
            array = self.get_empty()
        tile = array2dataset(array)
        self.pgn_tile(address, tile)
        return Container(array=array, dataset=tile)

    def get_tiles(self, addresses):
        """ Return generator of readonly gdal datasets. """
        for address in addresses:
            try:
                array = self.get_array(address)
            except KeyError:
                continue
            tile = gdal_array.OpenArray(array)
            self.pgn_tile(address, tile)
            yield tile

    # =========================================================================
    # Operations
    # -------------------------------------------------------------------------


    def oldadd(self, dataset, index=0):
        """ Add data. """
        report = self.investigate(conf=conf, dataset=dataset)
        lowest_level = self.get_level(report.pixel)
        bbox = geometry2rectangle(report.dataset)
        topaddress = self.get_topaddress(bbox)

        # Use outer envelope to walk the tiles
        addresses = self.walk(polygon=bbox,
                              level=lowest_level,
                              address=topaddress)

        cache = collections.defaultdict(list)
        previous_level = lowest_level
        for address in addresses:
            # Get the data
            container = self.get_tile(address=address)
            target = container.dataset

            # To aggregate or not
            if address.level == previous_level + 1:
                sources = [s.dataset for s in cache[previous_level]]
            else:
                sources = [dataset]
            for source in sources:
                rasters.reproject(source, target, gdal.GRA_NearestNeighbour)
                target.FlushCache()

            if address.level == previous_level + 1:
                del cache[previous_level]
                o_or_a = 'A'
            else:
                o_or_a = 'O'

            logger.debug('{} {} {}'.format(
                len(sources), o_or_a, address,
            ))

            cache[address.level].append(container)

            self.put_array(address, container.array)
            previous_level = address.level

    def warpinto(self, dataset, index=0):
        """
        # Warp from lowest level and count read tiles.
        # If no read tiles, use an address to seek some levels up until
        # data is found, and warp that.
        """
        report = self.investigate(dataset)
        level = self.get_level(report.pixel)
        extent = tuple(polygon2envelopepoints(report.outline).ravel())
        warps = 0
        while warps == 0:
            addresses = self.iteraddresses(level=level,
                                           index=index,
                                           extent=extent)
            tiles = self.get_tiles(addresses)
            for tile in tiles:
                rasters.reproject(tile, dataset, gdal.GRA_NearestNeighbour)
                warps += 1
            level += 1
