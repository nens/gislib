# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import hashlib
import math
import pickle

from osgeo import gdal
from osgeo import ogr

import numpy as np

from gislib.store import storages

gdal.UseExceptions()
ogr.UseExceptions()

def points2polygon(points):
    """ 
    Return a polygon geometry.

    This method numpy to prepare a wkb string. Seems only faster for
    larger polygons, compared to adding points individually.
    """
    # 13 bytes for the header, 16 bytes per point
    nbytes = 13 + 16 * points.shape[0]
    data = np.empty(nbytes, dtype=np.uint8)
    # little endian
    data[0:1] = 1
    # wkb type, number of rings, number of points
    data[1:13].view(np.uint32)[:] = (3, 1, points.shape[0])
    # set the points
    data[13:].view(np.float64)[:] = points.ravel()
    return ogr.CreateGeometryFromWkb(data.tostring())


def extent2polygon(xmin, ymin, xmax, ymax):
    """ Return an extent polygon. """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint_2D(xmin, ymin)
    ring.AddPoint_2D(xmax, ymin)
    ring.AddPoint_2D(xmax, ymax)
    ring.AddPoint_2D(xmin, ymax)
    ring.AddPoint_2D(xmin, ymin)
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    return polygon
    

def dataset2polygon(dataset):
    """ Return polygon formed by pixel edges. """
    nx, ny = dataset.RasterXSize, dataset.RasterYSize
    npoints = 2 * (nx + ny) + 1
    # Construct a wkb array for speed
    nbytes = 13 + 16 * npoints
    data = np.empty(nbytes, dtype=np.uint8)
    data[0:1] = 1  # little endian
    # wkb type, number of rings, number of points
    numbers = data[1:13].view(np.uint32)[:] = [3, 1, npoints]
    points = data[13:].view(np.float64).reshape(npoints, 2)

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
    points[:, 0] = xul + dxx * px + dxy * py
    points[:, 1] = yul + dyx * px + dyy * py

    return points2polygon(points)
    

class Location(object):
    def __init__(self, time, level, indices):
        """ x, y, and t refer to block indices. """
        self.time = time
        self.level = level
        self.indices = indices
        self.key = hashlib.md5(self.tostring()).hexdigest()

    def tostring(self):
        """ Return string. """
        return self.toarray().tostring()

    def toarray(self):
        """ Return array. """
        return np.int64((self.time,) + (self.level,) + self.indices)
    
    def __str__(self):
        return '<Location: {}>'.format(self.toarray())

    def __repr__(self):
        return '<Location: {}>'.format(self.toarray())


class Store(object):

    # Names
    DATA = 'data'

    def __init__(self, path):
        """ Initialize a store from a path. """
        self.storage = storages.FileStorage(path)
        self.conf = self.storage.get_schema('conf', split=False)

        try:
            self.data = pickle.loads(self.conf[self.DATA])
        except KeyError:
            pass

    
    def create_data(self, **kwargs):
        schema = self.storage.get_schema(self.DATA, split=True)
        self.data = Data(schema=schema, **kwargs)
        self.conf[self.DATA] = pickle.dumps(self.data)

    def __setitem__(self, indices, dataset):
        """ Previously the pyramids add method """
        polygon = dataset2polygon(dataset)
        location = Location(time=0, level=0, indices=(0,0))
        print(self.data.grid.get_polygon(location))
        from arjan.monitor import Monitor; mon = Monitor() 
        for i in range(1):
            location = self.data.grid.get_location(polygon)
        mon.check('') 

        exit()

    def datasets(self, dataset):
        transformation = osr.CoordinateTransformation(
            projections.get_spatial_reference(28992), 
            projections.get_spatial_reference(4326),
        )
        polygon = dataset2polygon(dataset)
        polygon.transform(transform)

    def __getitem__(self, indices):
        """ Return a generator of datasets for times.
            They come in the blocks defined by data."""
    
    def warpinto(self, dataset, times):
        """ """
        pass

class Grid(object):
    """
    How tiles correspond to extents and sizes.
    """
    def __init__(self, size, base=2, factor=1, offset=(0, 0)):
        self.size = size 
        self.base = base
        self.factor = factor
        self.offset = offset

    def get_extent(self, location):
        """
        """
        pixelside = self.factor * self.base ** location.level

        return tuple(pixelside * self.size[i]
                     * (location.indices[i] + j) + self.offset[i]
                     for j in (0, 1) for i in (0, 1))

    def get_polygon(self, location):
        """ Return a polygon represnting the extent of the location. """
        return extent2polygon(*self.get_extent(location))

    def get_location(self, polygon):
        """ Return lowest-level location in which polygon fits entirely. """
        # Pick a location at level 0 that intersects with the polygon

        print('yeah?')
        # prepare for fast intersections using a rectangle.
        xmin, xmax, ymin, ymax = polygon.GetEnvelope()
        rect = extent2polygon(xmin, ymin, xmax, ymax)  # faster intersections

        import ipdb; ipdb.set_trace() 

        
        
        # Go smartly levels up until a location that contains the polygon



        

    def get_geotransform(location):
        """ Return the geotransform for use with gdal dataset. """

    def iter_children(location):
        """ Return location iterator. """


class Data(object):
    
    def __init__(self, 
                 schema, 
                 dtype=np.float32, 
                 chunks=(64, 64, 16),
                 projection=4326,
                 frame=None
                 ):
        """ Writes config, or reads it from storage. """
        self.schema = schema
        self.projection = projection
        self.dtype = dtype
        self.chunks = chunks
        self.grid = Grid(size=chunks[:2])


class Time(object):
    """ Just a large array! """
    def __init__(self, schema, units=None, datatype=None, size=None):
        """ Writes config, or reads it from storage. """

    def __setitem__(self, times, data):
        pass

    def __getitem__(self, times):
        pass
