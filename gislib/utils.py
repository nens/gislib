# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import datetime
import json
import logging
import os

from osgeo import gdal
from osgeo import osr
import numpy as np


# Enable gdal exceptions
gdal.UseExceptions()

def get_transformed_extent(extent, source_projection, target_projection):
        """
        Return new reprojected geometry.
        Must keep cellsize square.

        Projections can be epsg, or proj4, or wkt
        Keep the size constant, for now.
        """
        # Turn extent into array of corner points
        points_source = np.array(extent)[np.array([[0, 1],
                                                   [2, 1],
                                                   [2, 3],
                                                   [0, 3]])]
        # Transform according to projections
        x_target, y_target = np.array(osr.CoordinateTransformation(
            get_spatial_reference(source_projection),
            get_spatial_reference(target_projection),
        ).TransformPoints(points_source))[:, 0:2].T

        # Return as extent
        return (x_target.min(),
                y_target.min(),
                x_target.max(),
                y_target.max())


def get_extent_intersection(extent1, extent2):
    """ Return the intersecting extent. """
    return (max(extent1[0], extent2[0]),
            max(extent1[1], extent2[1]),
            min(extent1[2], extent2[2]),
            min(extent1[3], extent2[3]))


def reproject(source, target, algorithm):
    """ Reproject source to target. """
    gdal.ReprojectImage(
        source, target,
        get_wkt(source.GetProjection()),
        get_wkt(target.GetProjection()),
        algorithm,
        0.0,
        0.125,
    )


class Geometry(object):
    def __init__(self, extent, size):
        self.extent = extent
        self.size = size

    def width(self):
        return self.size[0]

    def height(self):
        return self.size[1]

    def shape(self):
        return self.size[::-1]

    def delta(self):
        """ Return size tuple in extent units. """
        left, bottom, right, top = self.extent
        return right - left, top - bottom

    def cellsize(self):
        """ Return cellsize tuple. """
        return tuple(np.array(self.delta()) / np.array(self.size))

    def geotransform(self):
        """ Return geotransform tuple. """
        left, top = self.extent[0], self.extent[3]
        cellwidth, cellheight = self.cellsize()
        return left, cellwidth, 0, top, 0, -cellheight

    def gridpoints(self):
        """ Return array of shape with * height, 2. """
        x1, y1, x2, y2 = self.extent
        width, height = self.size
        x_step, y_step = self.cellsize()

        mgrid = np.mgrid[y2 - y_step / 2:y1 + y_step / 2:height * 1j,
                         x1 + x_step / 2:x2 - x_step / 2:width * 1j]

        return mgrid[::-1].transpose(1, 2, 0).reshape(-1, 2)

    def gridcoordinates(self):
        """ Return x, y arrays of length width, height. """
        x1, y1, x2, y2 = self.extent
        width, height = self.size
        x_step, y_step = self.cellsize()

        ogrid = np.ogrid[y2 - y_step / 2:y1 + y_step / 2:self.height * 1j,
                         x1 + x_step / 2:x2 - x_step / 2:self.width * 1j]
        return ogrid[1].reshape(-1, 1), ogrid[0].reshape(-1, 1)


class DatasetGeometry(Geometry):
    """ Add methods specific to pyramid building and transformations. """

    @classmethod
    def from_dataset(cls, dataset):
        x, dxx, dxy, y, dxy, dyy = dataset.GetGeoTransform()
        size = dataset.RasterXSize, dataset.RasterYSize
        extent = (x,
                  y + dyy * size[1],
                  x + dxx * size[0],
                  y)
        return cls(extent=extent, size=size)

    def to_dataset(self, datatype=1, bands=1, projection=None):
        """ Return in-memory gdal dataset. """
        driver = gdal.GetDriverByName(b'mem')
        dataset = driver.Create(b'',
                                self.size[0], self.size[1], bands, datatype)
        dataset.SetGeoTransform(self.geotransform())
        dataset.SetProjection(get_wkt(projection))
        return dataset

    def to_transformed_geometry(self, source_projection, target_projection):
        """
        Return a geometry object.

        The extent of the returned geometry is such that it encloses the
        original extent. The cellsize of the returned geometry fits in
        the cellsize of the original geometry.
        """

    def transformed_cellsize(self, source_projection, target_projection):
        """ Return transformed cellsize. """
        left, bottom, right, top = get_transformed_extent(
            self.extent, source_projection, target_projection,
        )
        return min((right - left) / self.size[0],
                   (top - bottom) / self.size[1])

