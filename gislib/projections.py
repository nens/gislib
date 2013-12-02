# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import osr


# Enable gdal exceptions
osr.UseExceptions()


# Projections and transformations
GOOGLE = 'EPSG:3857'  # And not 900913!!! Gdal does not understand it.
RD = ("+proj=sterea +lat_0=52.15616055555555 +lon_0=5.38763888888889 "
      "+k=0.999908 +x_0=155000 +y_0=463000 +ellps=bessel "
      "+towgs84=565.237,50.0087,465.658,-0.406857,0.350733,-1.87035,4.0812 "
      "+units=m +no_defs")  # Copied from lizard_map/coordinates.py


# Can and should override this from importing application.
default_projection = RD


def get_spatial_reference(projection):
    """
    Return a SpatialReference object.

    projection can be:
        None or empty string: returns default_projection.
        integer epsg code
        epsg:integer string
        proj4 string
        wkt string
    """
    sr = osr.SpatialReference()
    if projection is None or projection == '':
        return get_spatial_reference(default_projection)
    else:
        sr.SetFromUserInput(str(projection))
    return sr


def get_coordinate_transformation(source, target):
    """ Return a coordinate tranformation. """
    return osr.CoordinateTransformation(
        get_spatial_reference(source),
        get_spatial_reference(target),
    )


def get_wkt(projection):
    """ Convenience function. """
    return get_spatial_reference(projection).ExportToWkt()


def get_proj4(projection):
    """ Convenience function. """
    return get_spatial_reference(projection).ExportToProj4()


def get_authority(projection):
    """ Convenience function. """
    sr = get_spatial_reference(projection)
    return '{}:{}'.format(sr.GetAttrValue(b"AUTHORITY", 0),
                          sr.GetAttrValue(b"AUTHORITY", 1))
