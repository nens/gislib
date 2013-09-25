# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import gdal
from osgeo import osr
import numpy as np

from gislib import projections
from gislib import vectors

# Enable gdal exceptions
gdal.UseExceptions()


def extent2polygon(extent):
    """ Return an extent polygon. """
    xmin, ymin, xmax, ymax = extent
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


def get_extent_intersection(extent1, extent2):
    """ Return the intersecting extent. """
    return (max(extent1[0], extent2[0]),
            max(extent1[1], extent2[1]),
            min(extent1[2], extent2[2]),
            min(extent1[3], extent2[3]))


def get_transformed_extent(extent, source_projection, target_projection):
    """
    Return extent transformed from source projection to target projection.
    """
    polygon = extent2polygon(extent)
    transformation = osr.CoordinateTransformation(
        projections.get_spatial_reference(source_projection),
        projections.get_spatial_reference(target_projection),
    )
    polygon.Transform(transformation)
    return geometry2envelopeextent(polygon)
