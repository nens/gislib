# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import hashlib
import logging
import logging.config
import math
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
Report = collections.namedtuple('Report', ('outline', 'pixel', 'transform'))
Container = collections.namedtuple('Container', ('array', 'dataset'))


def array2dataset(array):
    """ Return gdal dataset. """
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


def dataset2polygon(dataset):
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


def pixel2polygon(dataset):
    """ Return polygon corresponding to the first pixel of dataset. """
    xul, dxx, dxy, yul, dyx, dyy = dataset.GetGeoTransform()
    points = ((xul, yul),
              (xul + dxx, yul + dyx),
              (xul + dxx + dxy, yul + dyx + dyy),
              (xul + dxy, yul + dyy),
              (xul, yul))
    return points2polygon(points)


def extent2rectangle(xmin, ymin, xmax, ymax):
    """ Return an extent polygon. """
    points = ((xmin, ymin),
              (xmax, ymin),
              (xmax, ymax),
              (xmin, ymax),
              (xmin, ymin))
    return points2polygon(points)


def geometry2rectangle(geometry):
    """ Return the rectangular polygon of geometry's envelope. """
    xmin, xmax, ymin, ymax = geometry.GetEnvelope()
    return extent2rectangle(xmin, ymin, xmax, ymax)


def polygon2envelopepoints(polygon):
    """ Return array. """
    return np.array(polygon.GetEnvelope()).reshape(2, 2).transpose()


def polygon2envelopesize(polygon):
    """ Return size tuple. """
    xmin, xmax, ymin, ymax = polygon.GetEnvelope()
    return xmax - xmin, ymax - ymin


class Address(object):
    def __init__(self, level, tile, index=0):
        """ x, y, and t refer to block indices. """
        self.index = index
        self.level = level
        self.tile = tile

        self.key = hashlib.md5(self.tostring()).hexdigest()

    def tostring(self):
        """ Return string. """
        return self.toarray().tostring()

    def toarray(self):
        """ Return array. """
        return np.int64((self.index,) + (self.level,) + self.tile)

    def __str__(self):
        return '<Location: index {}, level {}, tile {}>'.format(
            self.index, self.level, self.tile,
        )

    def __repr__(self):
        return str(self)


class Pyramid():
    """ Non locking smart updating geospatial pyramid. """

    DATA = 'data'
    CONF = 'conf'

    def __init__(self,
                 path,
                 dtype=np.float32,
                 tilesize=(256, 256),
                 projection=3857):
        """ Initialize. """
        # Storage
        self.storage = storages.FileStorage(path=path)
        self.conf = self.storage.get_schema(self.CONF)
        self.data = self.storage.get_schema(self.DATA, split=True)

        # Configuration, (how to persist this?)
        self.dtype = dtype
        self.projection = projection
        self.tilesize = tilesize

        self.maxlevel = self.get_maxlevel(projection)

        (pixels, lines), bands = tilesize, 1
        self.shape = (bands, lines, pixels)

        if np.dtype(dtype).kind == 'f':
            self.nodata = np.finfo(dtype).min
        else:
            self.nodata = np.iinfo(dtype).min

    # =========================================================================
    # Location navigation
    # -------------------------------------------------------------------------

    def get_maxlevel(self, projection):
        """
        Return maxlevel for a projection.

        Trick is to take a unit pixel and transform it to google.
        """
        transformation = osr.CoordinateTransformation(
            projections.get_spatial_reference(projection),
            projections.get_spatial_reference(3857),
        )
        points = (0, 0), (1, 1)
        (x1, y1, x1), (x2, y2, z2) = transformation.TransformPoints(points)
        length = min(x2 - x1, y2 - y1)
        return int(math.ceil(math.log(2 ** 25 / length, 2)))

    def get_level(self, pixel):
        """
        Get the minimum level for a pixel polygon.
        """
        size = polygon2envelopesize(pixel)
        return int(math.floor(math.log(min(size), 2)))

    def get_extent(self, address):
        """
        Return extent tuple.
        """
        return tuple(2 ** address.level *
                     self.tilesize[i] *
                     (address.tile[i] + j)
                     for j in (0, 1) for i in (0, 1))

    def get_geotransform(self, address):
        """ Return the geotransform for use with gdal dataset. """
        pixelsize = 2 ** address.level
        xmin, ymin, xmax, ymax = self.get_extent(address)
        return xmin, pixelsize, 0, ymax, 0, -pixelsize

    def get_polygon(self, address):
        """ Return a polygon represnting the extent of the address. """
        return extent2rectangle(*self.get_extent(address))

    def get_topaddress(self, bbox, index=0):
        """ Return the toplevel address. """
        point = bbox.Centroid().GetPoint(0)
        level = self.maxlevel
        tile = tuple(int(math.floor(c / (2 ** level * t)))
                     for t, c in zip(self.tilesize, point))
        return Address(index=index, level=level, tile=tile)

    def iteraddresses(self, level, index=0, extent=None, address=None):
        """
        Return location iterator.

        index is only passed to the output locations.
        """
        if extent is None:
            xmin, ymin, xmax, ymax = self.get_extent(address)
        else:
            xmin, ymin, xmax, ymax = extent

        # new level pixel- and tilesizes
        tilesize = tuple(2 ** level * s for s in self.tilesize)

        # Determine the ranges for the indices
        tiles_x, tiles_y = map(
            xrange,
            (int(math.floor((e) / t))
             for e, t in zip((xmin, ymin), tilesize)),
            (int(math.ceil((e) / t))
             for e, t in zip((xmax, ymax), tilesize)),
        )

        for tile_y in tiles_y:
            for tile_x in tiles_x:
                yield Address(index=address.index if extent is None else index,
                              level=level,
                              tile=(tile_x, tile_y))

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

    def investigate(self, dataset):
        """
        Return Report tuple.

        To transform or not to transform, that is the question. If
        the outcome is not to transform, original pixel and outline
        coordinates are returned, transformed versions otherwise.
        Transformation is considered unnecessary if the outline changes
        less then 1 % of the pixelsize by transformation.

        The outline is shrunk by 1% of the shortest pixelside, to prevent
        touching tiles to be retrieved.
        """
        transformation = osr.CoordinateTransformation(
            projections.get_spatial_reference(dataset.GetProjection()),
            projections.get_spatial_reference(self.projection),
        )

        # outline
        outline_org = dataset2polygon(dataset)
        outline_trf = outline_org.Clone()
        outline_trf.Transform(transformation)

        # pixel
        pixel_org = pixel2polygon(dataset)
        pixel_trf = pixel_org.Clone()
        pixel_trf.Transform(transformation)

        # verdict
        pixel_trf_size = polygon2envelopesize(pixel_trf)
        diff = (polygon2envelopepoints(outline_trf) -
                polygon2envelopepoints(outline_org))
        transform = (100 * diff > pixel_trf_size).any()
        if transform:
            pixel = pixel_trf
            outline = outline_trf.Buffer(-0.01 * min(pixel_trf_size))
        else:
            pixel = pixel_org
            outline = outline_org.Buffer(-0.01 * min(pixel_trf_size))
        return Report(pixel=pixel, outline=outline, transform=transform)

    def add(self, dataset, index=0):
        """ Add data. """
        report = self.investigate(dataset)
        lowest_level = self.get_level(report.pixel)
        bbox = geometry2rectangle(report.outline)
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
