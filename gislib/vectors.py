# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from math import radians, sin, cos, asin, sqrt, acos
from itertools import izip, tee

from osgeo import ogr

import numpy as np

from gislib import projections


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


def point2geometry(point):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbPoint)
    geometry.AddPoint_2D(*map(float, point))
    return geometry


def line2geometry(line):
    """ Return geometry. """
    geometry = ogr.Geometry(ogr.wkbLineString)
    for point in line:
        geometry.AddPoint_2D(*map(float, point))
    return geometry


def magnitude(vectors):
    """ Return magnitudes. """
    return np.sqrt((vectors ** 2).sum(1))


def normalize(vectors):
    """ Return unit vectors. """
    return vectors / magnitude(vectors).reshape(-1, 1)


def rotate(vectors, degrees):
    """ Return vectors rotated by degrees. """
    return np.vstack([
        +np.cos(np.radians(degrees)) * vectors[:, 0] +
        -np.sin(np.radians(degrees)) * vectors[:, 1],
        +np.sin(np.radians(degrees)) * vectors[:, 0] +
        +np.cos(np.radians(degrees)) * vectors[:, 1],
    ]).transpose()


class Geometry(object):
    """ Wrapper around ogr geometry for working with extents. """
    def __init__(self, geometry):
        self.geometry = geometry

    def transform(self, source, target):
        """
        Transform geometry from source projection to target projection.
        """
        transformation = projections.get_coordinate_transformation(
            source=source, target=target,
        )
        self.geometry.Transform(transformation)

    @classmethod
    def fromextent(cls, x1, y1, x2, y2):
        points = ((x1, y1),
                  (x2, y1),
                  (x2, y2),
                  (x1, y2),
                  (x1, y1))
        return cls(geometry=points2polygon(points))

    @property
    def extent(self):
        """ Return x1, y1, x2, y2. """
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        return x1, y1, x2, y2

    @property
    def envelope(self):
        """ Return polygon representing envelope. """
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
        return points2polygon(points)

    @property
    def size(self):
        """ Return width, height. """
        x1, x2, y1, y2 = self.geometry.GetEnvelope()
        return x2 - x1, y2 - y1


class MagicLine(object):
    """
    LineString with handy parameterization and projection properties.
    """
    def __init__(self, points):
        # Data
        self.points = np.array(points)
        # Views
        self.p = self.points[:-1]
        self.q = self.points[1:]
        self.lines = np.hstack([self.p, self.q]).reshape(-1, 2, 2)
        # Derivatives
        self.length = len(points) - 1
        self.vectors = self.q - self.p
        self.centers = (self.p + self.q) / 2

    def __getitem__(self, parameters):
        """ Return points corresponding to parameters. """
        i = np.uint64(np.where(parameters == self.length,
                               self.length - 1, parameters))
        t = np.where(parameters == self.length,
                     1, np.remainder(parameters, 1)).reshape(-1, 1)
        return self.p[i] + t * self.vectors[i]

    def _pixelize_to_parameters(self, size):
        """
        Return array of parameters where pixel boundary intersects self.
        """
        extent = np.array([self.points.min(0), self.points.max(0)])
        size = np.array([1, 1]) * size  # Coerce scalar size
        parameters = []
        # Loop dimensions for intersection parameters
        for i in range(extent.shape[-1]):
            intersects = np.arange(
                size[i] * np.ceil(extent[0, i] / size[i]),
                size[i] * np.ceil(extent[1, i] / size[i]),
                size[i],
            ).reshape(-1, 1)
            # Calculate intersection parameters for each vector
            nonzero = self.vectors[:, i].nonzero()
            lparameters = ((intersects - self.p[nonzero, i]) /
                           self.vectors[nonzero, i])
            # Add integer to parameter and mask outside line
            global_parameters = np.ma.array(
                np.ma.array(lparameters + np.arange(nonzero[0].size)),
                mask=np.logical_or(lparameters < 0, lparameters > 1),
            )
            # Only unmasked values must be in parameters
            parameters.append(global_parameters.compressed())

        # Add parameters for original points
        parameters.append(np.arange(self.length + 1))

        return np.sort(np.unique(np.concatenate(parameters)))

    def pixelize(self, size, endsonly=False):
        """
        Return pixelized MagicLine instance.
        """
        all_parameters = self._pixelize_to_parameters(size)
        if endsonly:
            index_points = np.equal(all_parameters,
                                    np.round(all_parameters)).nonzero()[0]
            index_around_points = np.sort(np.concatenate([
                index_points,
                index_points[:-1] + 1,
                index_points[1:] - 1,
            ]))
            parameters = all_parameters[index_around_points]
        else:
            parameters = all_parameters

        return self.__class__(self[parameters])

    def project(self, points):
        """
        Return array of parameters.

        Find closest projection of each point on the magic line.
        """
        pass


def great_circle_distance(coordinates, formula='haversine'):
    """
    Calculate the (great circle) distance between two or more points
    on the earth (specified in decimal degrees). If more than two
    points are given the accumulated distance will be calculated.

    :param coordinates: list of lists or list of tuples of
                        geographic coordinates as decimal
                        fractions (decimal degrees)
    :param formula: haversine or cosine (use cosine cosine for
                    points with a greater distance between them)

    :return distance in km

    example usage::
        coords = [(4.8896900,52.3740300),      # long/lat Amsterdam
                  (13.4105300, 52.5243700)     # long/lat Berlin
                  ]
        great_circle_distance(coords)
        577.358

    """
    if not isinstance(coordinates[0], (list, tuple)):
        raise ValueError('Coordinates must be either provided as a '
                         'list of lists or list of tuples')

    izip_pairs = _pairwise(coordinates)

    accumulated_km = 0
    for pair in izip_pairs:
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = pair[0][0], pair[0][1], pair[1][0], pair[1][1]
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        if formula.lower() == 'haversine':
            # haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            km = 6371 * c
            accumulated_km += km
        elif formula.lower() == 'cosine':
            km = acos(
                sin(lat1)*sin(lat2)+cos(lat1)*cos(lat2)*cos(lon2-lon1)
            )*6371
            accumulated_km += km
        else:
            raise AttributeError("The formula parameter must either be "
                                 "'haversine' or 'cosine'")
    return accumulated_km


def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)
