# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import math

import netCDF4
import numpy as np

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

    def get_transformed(self, kind, size, extent, axes):
        """
        Return dictionary.

        It contains transoformed size and extent when input size and
        extent are transformed from self to kind. This is used for
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
        return dict(size=size, extent=trans_extent, axes=axes)

    def get_sliced(self, axes, size, extent, clip_axes, clip_extent):
        """ 
        Return dictionary.

        size: clipped size
        extent: clipped extent
        axes: unchanged axes
        slices: clipping slices based on extent & size
        """
        slices = []
        clipped_size = []
        clipped_extent = []
        stuff = zip(zip(*extent), zip(*clip_extent), size)
        for (e1, e2), (c1, c2), s in stuff:
            espan = e2 - e1
            start = max(0, min(s, int(math.floor((c1 - e1) / espan * s))))
            stop = max(0, min(s, int(math.ceil((c2 - e1) / espan * s))))
            slices.append(slice(start, stop))
            clipped_size.append(stop - start)
            clipped_extent.append(e1 + start * espan, e1 + stop * espan)

        tuple(map(lambda l: e1 + l / sz * (e2 - e1), (sl.start, sl.stop))
        return dict(
            size=tuple(clipped_size),
            extent=tuple(clipped_extent),
            axces=(),
            slices=tuple(slices),
        )


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

    def get_transformed(self, kind, size, extent, axes):
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
        return dict(size=size, extent=trans_extent, axes=axes)
    
    def get_sliced(self, axes, size, extent, clip_axes, clip_extent):
        """ 
        Return dictionary.

        size: clipped size
        extent: unchanged extent
        axes: clipped axes
        slices: clipping slices based on axes & extent
        """
        ((e1,), (e2,)) = extent
        values = e1 + axes[0] * (e2 - e1)
        indices = np.where(np.logical_and(
            values >= clip_extent[0],
            values < clip_extent[1],
        ))[0]
        if indices.size:
            slices =  (slice(indices[0], indices[-1] + 1),)
        else:
            slices =  (slice(0, 0),)

        return dict(
            size=tuple(s.stop - s.start for s in slices)
            extent=extent,
            axes=tuple(a[s] for a, s in zip(axes, slices))
            slices=slices,
        )
