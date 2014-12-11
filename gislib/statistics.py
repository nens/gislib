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
    polygon = vectors.Geometry.fromextent(*extent).envelope
    transformation = osr.CoordinateTransformation(
        projections.get_spatial_reference(source_projection),
        projections.get_spatial_reference(target_projection),
    )
    polygon.Transform(transformation)
    return vectors.Geometry(geometry=polygon).extent


def get_curve(array, bins=256):
    """
    TODO Better name.

    Return array with graph points (in Dutch, maaiveldcurve)

    :param array: 3d numpy masked array
    :param bins: number of histogram bins
    :returns: curve_x, list of x values; curve_y, list of y values
    """
    # compute histogram
    histogram, edges = np.histogram(array.compressed(), bins)

    # convert to percentiles
    percentile_x = np.cumsum(histogram) / float(histogram.sum()) * 100
    percentile_y = edges[1:]  # right edges of bins.
    curve_x = np.arange(0, 101)
    curve_y = np.interp(curve_x, percentile_x, percentile_y)
    return curve_x, curve_y


def get_counts(array, bins=256):
    """

    Return array with counts of classified raster.

    :param array: 3d numpy masked array
    :param bins: number of histogram bins
    :returns: pairs, list of top 10 (class, count);
              rest, list of rest (class, count)

    """
    bins = np.arange(0, bins)
    histograms = [np.histogram(d.compressed(), bins)[0] for d in array]
    nonzeros = [h.nonzero() for h in histograms]
    nbins = [bins[:-1][n] for n in nonzeros]
    nhistograms = [h[n] for n, h in zip(nonzeros, histograms)]
    # Determine the ordering
    argsorts = [h.argsort()[::-1] for h in nhistograms]
    # Use it to group
    pairs = [np.array([b[argsorts], h[argsorts]]).transpose().tolist()
             for b, h in zip(nbins, nhistograms)]

    return pairs
