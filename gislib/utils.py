# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import gdal
from osgeo import osr
import numpy as np

from gislib import projections


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
            projections.get_spatial_reference(source_projection),
            projections.get_spatial_reference(target_projection),
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
