# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import netCDF4

from gislib import rasters

class NonEquidistantKind(object):
    pass


class Space(object):
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

    def transform(self, kind, size, extent):
        """ Return dictionary. """
        if kind == self:
            return dict(size=size, extent=extent)
        trans_extent_flat = rasters.get_transformed_extent(
            extent=tuple(y for x in extent for y in x),
            source_projection=self.proj,
            target_projection=kind.proj,
        )
        trans_extent = trans_extent_flat[:2], trans_extent_flat[2:]
        return dict(size=size, extent=trans_extent)

    def transform_data(self, kind, size, extent, data):
        """ Return dictionary. """
        return dict(size='todo', extent='todo', data='todo', axes='todo')


class Time(NonEquidistantKind):
    """ Timeseries kind. """
    def __init__(self, unit):
        self.unit = unit

    def __eq__(self, other):
        return self.unit == other.unit

    def __str__(self):
        return '<{cls}:{unit}>'.format(
            cls=self.__class__.__name__, **self.__dict__
        )

    def transform(self, kind, size, extent):
        """ Return dictionary. """
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
        return dict(size=size, extent=trans_extent)

    def transform_data(self, kind, size, extent, data):
        """ Return dictionary. """
        return dict(size='todo', extent='todo', data='todo', axes='todo')
