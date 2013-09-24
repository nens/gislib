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
from osgeo import gdal_array
from osgeo import ogr
from osgeo import osr

import numpy as np

from gislib import projections
from gislib import rasters
from gislib import utils
from gislib import vectors


class BaseStore(object):
    """ Base class for anything that has a warpinto method. """
    
    def get_array(self, extent, size, projection):
        """
        Return data for the pyramids.

        Datatype and nodatavalue are taken from first pyramid.

        extent: xmin, xmax, ymin, ymax-tuple
        size: width, height-tuple
        projection: something like 'epsg:28992' or a wkt or proj4 string.
        """
        info = self.info

        # Create a dataset
        array = np.ones(
            (
                1,
                size[1],
                size[0],
            ),
            dtype=gdal_array.flip_code(info['datatype']),
        ) * info['nodatavalue']
        dataset = rasters.array2dataset(array)

        # Add georeferencing
        xmin, ymin, xmax, ymax = extent
        geotransform = (xmin, (xmax - xmin) / array.shape[-1], 0,
                        ymax, 0, (ymin - ymax) / array.shape[-2])
        dataset.SetProjection(projections.get_wkt(projection))
        dataset.SetGeoTransform(geotransform)

        # Get data
        self.warpinto(dataset)
        dataset.FlushCache()

        return np.ma.masked_equal(array, info['nodatavalue'], copy=False)

    def get_profile(self, line, size, projection):
        """
        Return a distances, values tuple of numpy arrays.

        line: ogr.wkbLineString
        size: integer
        projection: something like 'epsg:28992' or a wkt or proj4 string.
        """
        # Buffer line with 1 percent of length to keep bbox a polygon
        extent = utils.geometry2envelopeextent(
            line.Buffer(line.Length() / 100)
        )
        x1, y1, x2, y2 = extent
        span = x2 - x1, y2 - y1

        # Determine width, height and cellsize
        if max(span) == span[0]:
            width = size
            cellsize = span[0] / size
            height = int(math.ceil(span[1] / cellsize))
        else:
            height = size
            cellsize = span[1] / size
            width = int(math.ceil(span[0] / cellsize))

        # Determine indices for one point per pixel on the line
        vertices = line.GetPoints()
        magicline = vectors.MagicLine(vertices).pixelize(cellsize)
        origin = np.array([x1, y2])
        points = magicline.centers
        indices = tuple(np.uint64(
            (points - origin) / cellsize * np.array([1, -1]),
        ).transpose())[::-1]

        # Get the values from the array
        array = self.get_array(extent, (width, height), projection)
        values = array[0][indices]

        # make array with distance from origin (x values for graph)
        magnitudes = vectors.magnitude(magicline.vectors)
        distances = magnitudes.cumsum() - magnitudes[0] / 2

        return distances, values


class MultiStore(BaseStore):
    """ Pyramid wrapper for data extraction from a list of pyramids. """
    def __init__(self, stores):
        self.stores = stores

    @property
    def info(self):
        """ Return store info. """
        return self.stores[0].info
    
    def warpinto(self, dataset):
        """ Multistore version of warpinto. """
        for store in stores:
            store.warpinto(dataset)
