# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import netCDF4

from gislib import rasters

class DiscreteKind(object):
    pass

class ContinuousKind(object):
    pass

class Space(ContinuousKind):
    """ Gdal kind. """
    def __init__(self, proj):
        self.proj = proj

    def __eq__(self, other):
        return self.proj == other.proj

    def __str__(self):
        return '<{cls}:{proj}>'.format(
            cls=self.__class__.__name__, **self.__dict__
        )

    def __repr__(self):
        return self.__str__()

    def transform(self, kind, size, extent, axis=None):
        """
        Return dictionary.

        It contains transoformed size and extent when input size and
        extent are transformed from kind to kind. This is used for
        location determination.

        Gotchas
        - Some x, y-swapping is taking place
        - No pincushion deforms are taken into account
        - Size is returned unaltered
        """
        if kind == self:
            return dict(size=size, extent=extent)
        trans_extent_flat = rasters.get_transformed_extent(
            extent=tuple(y for x in extent for y in x[::-1]),
            source_projection=self.proj,
            target_projection=kind.proj,
        )
        trans_extent = trans_extent_flat[1::-1], trans_extent_flat[:1:-1]
        return dict(size=size, extent=trans_extent, axis=axis)


class Time(DiscreteKind):
    """ Timeseries kind. """
    def __init__(self, unit):
        self.unit = unit

    def __eq__(self, other):
        return self.unit == other.unit

    def __str__(self):
        return '<{cls}:{unit}>'.format(
            cls=self.__class__.__name__, **self.__dict__
        )

    def __repr__(self):
        return self.__str__()

    def transform(self, kind, size, extent, axis=None):
        """ 
        Return dictionary.

        It contains transoformed size and extent when input size and
        extent are transformed from kind to kind. This is used for
        location determination.
        """
        if kind == self:
            return dict(size=size, extent=extent)
        trans_extent = tuple(
            map(
                tuple,
                netCDF4.date2num(
                    netCDF4.num2date(extent, self.unit),
                    kind.unit,
                ),
            ),
        )
        return dict(size=size, extent=trans_extent, axis=axis)
