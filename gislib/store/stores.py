# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections
import hashlib
import io
import logging
import logging.config
import math
import pickle

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


class Location(object):
    def __init__(self, level, indices, time=0):
        """ x, y, and t refer to block indices. """
        self.time = time
        self.level = level
        self.indices = indices
        self.key = hashlib.md5(self.tostring()).hexdigest()

    @classmethod
    def fromstring(cls, data):
        """ Return location. """
        time, level, x, y = np.fromstring(data, np.uint64)
        return cls(level=level, time=time, indices=(x, y))

    def tostring(self):
        """ Return string. """
        return self.toarray().tostring()

    def toarray(self):
        """ Return array. """
        return np.int64((self.time,) + (self.level,) + self.indices)

    def __str__(self):
        return '<Location: level {}, indices {}, time {}>'.format(
            self.level, self.indices, self.time,
        )

    def __repr__(self):
        return str(self)


class Slicer(object):
    """ Methods common to Data and Time classes. """
    @property
    def timesize(self):
        return self.chunks[2]

    def itertimes(self, indices):
        """ Return generator of time block positions. """
        if isinstance(indices, slice):
            start = indices.start // self.timesize
            stop = (indices.stop - 1) // self.timesize + 1
            for i in xrange(start, stop):
                yield i
        else:
            yield indices // self.timesize

    def get_timeslice(self, indices, time):
        """ Return inter time block slice. """
        offset = time * self.timesize
        if isinstance(indices, slice):
            start = max(0, indices.start - offset)
            stop = min(self.timesize, indices.stop - offset)
            return slice(start, stop)
        else:
            return slice(indices - offset, indices - offset + 1)


class Container(object):
    """ Container for tiledata. """
    def __init__(self, array, dataset, original, location):
        self.array = array
        self.dataset = dataset
        self.location = location
        self.original = original

    def tostring(self):
            buf = io.BytesIO()
            buf.write(self.location.tostring())
            buf.write(np.uint8(self.original).tostring())
            buf.write(self.array.tostring())
            buf.seek(0)
            return buf.read()


class Raster(Slicer):

    LOCATION = slice(32)
    ORIGINAL = 32
    ARRAY = slice(33, None)

    def __init__(self,
                 schema,
                 dtype=np.float32,
                 chunks=(256, 256, 1),
                 projection=3857,
                 base=2,
                 factor=1,
                 offset=(0, 0)):
        """ How to config? """
        self.base = base
        self.chunks = chunks
        self.dtype = dtype
        self.factor = factor
        self.offset = offset
        self.projection = projection
        self.schema = schema

        if np.dtype(dtype).kind == 'f':
            self.nodata = np.finfo(dtype).min
        else:
            self.nodata = np.iinfo(dtype).min

    @property
    def tilesize(self):
        return self.chunks[:2]

    # =========================================================================
    # Location navigation
    # -------------------------------------------------------------------------

    def get_level(self, pixel):
        """
        Get the minimum level for a pixel polygon.
        """
        size = polygon2envelopesize(pixel)
        return int(math.floor(math.log(min(size) / self.factor, self.base)))

    def get_pixelsize(self, level):
        """ Return the size of a pixel at level. """
        return self.factor * self.base ** level

    def get_extent(self, location):
        """
        """
        pixelsize = self.get_pixelsize(location.level)

        return tuple(pixelsize * self.tilesize[i]
                     * (location.indices[i] + j) + self.offset[i]
                     for j in (0, 1) for i in (0, 1))

    def get_geotransform(self, location):
        """ Return the geotransform for use with gdal dataset. """
        pixelsize = self.get_pixelsize(location.level)
        xmin, ymin, xmax, ymax = self.get_extent(location)
        return xmin, pixelsize, 0, ymax, 0, -pixelsize

    def get_polygon(self, location):
        """ Return a polygon represnting the extent of the location. """
        return extent2rectangle(*self.get_extent(location))

    def get_toplocation(self, bbox):
        """ Return first location entirely containing polygon. """
        # Determine minimum level to start at
        length = max(polygon2envelopesize(bbox))
        level = int(math.ceil(math.log(length / self.factor, self.base)))

        # Find an intersecting tile at this level
        point = bbox.Centroid().GetPoint(0)
        pixelsize = self.get_pixelsize(level)
        tilesize_offset_point = self.tilesize, self.offset, point
        indices = tuple(int(math.floor((c - o) / (t * pixelsize)))
                        for t, o, c in zip(*tilesize_offset_point))

        # Level up until location polygon contains dataset polygon
        location = Location(level=level, indices=indices)
        while not self.get_polygon(location).Contains(bbox):
            location = self.iterlocations(location=location,
                                          level=location.level + 1).next()
        return location

    def iterlocations(self, level, extent=None, location=None, time=None):
        """ Return location iterator. """
        if extent is None:
            xmin, ymin, xmax, ymax = self.get_extent(location)
        else:
            xmin, ymin, xmax, ymax = extent

        # new level pixel- and tilesizes
        pixelsize = self.get_pixelsize(level)
        tilesize = tuple(s * pixelsize for s in self.tilesize)

        # Determine the ranges for the indices
        indices_x, indices_y = map(
            xrange,
            (int(math.floor((e - o) / t))
             for e, o, t in zip((xmin, ymin), self.offset, tilesize)),
            (int(math.ceil((e - o) / t))
             for e, o, t in zip((xmax, ymax), self.offset, tilesize)),
        )

        for index_y in indices_y:
            for index_x in indices_x:
                yield Location(time=location.time if extent is None else time,
                               level=level,
                               indices=(index_x, index_y))

    def walk(self, location, polygon, level):
            """
            Return generator of sublocations that intersect with polygon,
            stopping at level.
            """
            if not self.get_polygon(location).Intersects(polygon):
                return
            if location.level > level:
                sublocations = self.iterlocations(location=location,
                                                  level=location.level - 1)
                for sublocation in sublocations:
                    walklocations = self.walk(level=level,
                                              polygon=polygon,
                                              location=sublocation)
                    for walklocation in walklocations:
                        yield walklocation
            yield location

    # =========================================================================
    # Storage
    # -------------------------------------------------------------------------

    def get_array(self, data):
        """ Return array from data """
        array = np.frombuffer(data[self.ARRAY], dtype=self.dtype)
        array.shape = self.chunks[::-1]
        array.flags.writeable = True  # This may be a bug.
        return array

    def get_original(self, data):
        """ Return original flag from data. """
        return bool(np.fromstring(data[self.ORIGINAL], np.uint8))

    def get_container(self, location=None, data=None):
        """ Return container. """
        # Fill in missing information
        if data is None:
            try:
                data = self.schema[location.key]
                original = self.get_original(data)
                array = self.get_array(data)
            except KeyError:
                original = False
                array = (self.dtype(self.nodata) *
                         np.ones(self.chunks[::-1], self.dtype))
        else:
            original = self.get_original(data)
            array = self.get_array(data)
            location = Location.fromstring(data[self.LOCATION])

        # Dataset
        dataset = array2dataset(array)
        dataset.SetProjection(projections.get_wkt(self.projection))
        dataset.SetGeoTransform(self.get_geotransform(location))
        if np.dtype(self.dtype).kind == 'f':
            nodata = float(self.nodata)
        else:
            nodata = int(self.nodata)
        for i in range(dataset.RasterCount):
            dataset.GetRasterBand(i + 1).SetNoDataValue(nodata)

        return Container(array=array,
                         dataset=dataset,
                         location=location,
                         original=original)

    def put_container(self, container):
        """ Save data. """
        if (container.array == self.nodata).all():
            try:
                del self.schema[container.location.key]
            except KeyError:
                pass
        else:
            self.schema[container.location.key] = container.tostring()

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

    def __setitem__(self, indices, dataset):
        """ Add data. """
        report = self.investigate(dataset)
        lowest_level = self.get_level(report.pixel)
        bbox = geometry2rectangle(report.outline)
        toplocation = self.get_toplocation(bbox)

        # Use outer envelope to walk the tiles
        locations = self.walk(polygon=bbox,
                              level=lowest_level,
                              location=toplocation)

        cache = collections.defaultdict(list)
        previous_level = lowest_level
        for location in locations:
            # Get the tile
            container = self.get_container(location=location)
            target = container.dataset

            # To aggregate or not
            if location.level == previous_level + 1:
                sources = [s.dataset for s in cache[previous_level]]
                container.original = False
            else:
                sources = [dataset]
                container.original = True

            for source in sources:
                rasters.reproject(source, target, gdal.GRA_NearestNeighbour)
                target.FlushCache()
            target = container.dataset
            logger.debug('{} {} {}'.format(
                len(sources), 'O' if container.original else 'A', location,
            ))
            if not container.original:
                del cache[previous_level]

            cache[location.level].append(container)

            self.put_container(container)
            previous_level = location.level

    def warpinto(self, dataset, times):
        """ """
        pass


class Time(Slicer):
    """ Just a large array! """
    def __init__(self, schema, units=None, datatype=None, size=None):
        """ Writes config, or reads it from storage. """

    def __setitem__(self, times, data):
        pass

    def __getitem__(self, times):
        pass


class Store(object):
    # Schema names
    RASTER = 'data'
    CONF = 'conf'

    def __init__(self, path):
        """ Initialize a store from a path. """
        self.storage = storages.FileStorage(path)
        self.conf = self.storage.get_schema(self.CONF, split=False)

        try:
            self.raster = pickle.loads(self.conf[self.RASTER])
        except KeyError:
            pass

    def init_raster(self, **kwargs):
        schema = self.storage.get_schema(self.RASTER, split=True)
        self.raster = Raster(schema=schema, **kwargs)
        self.conf[self.RASTER] = pickle.dumps(self.raster)
