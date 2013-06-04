# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import ogr

import numpy as np


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
        parameters = []
        # Loop dimensions for intersection parameters
        for i in range(extent.shape[-1]):
            intersects = np.arange(
                size * np.ceil(extent[0, i] / size),
                size * np.ceil(extent[1, i] / size),
                size,
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
