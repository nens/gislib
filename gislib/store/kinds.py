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

    def transform(self, extent, kind):
        """
        Return transformed extent.

        Gotcha's
        - Some x, y-swapping is taking place
        - No pincushion deforms are taken into account
        """
        trans_extent_flat = rasters.get_transformed_extent(
            extent=tuple(y for x in extent for y in x[::-1]),
            source_projection=self.proj,
            target_projection=kind.proj,
        )
        return trans_extent_flat[1::-1], trans_extent_flat[:1:-1]

    def get_prep(self, source_size, source_extent, 
                 source_axes, target_extent, target_axes):
        """ 
        Return dictionary.

        Source and target must be of kind self.

        size: cropped size
        extent: clipped extent
        axes: empty tuple, for this kind.
        slices: clipping slices based on extent & size
        """
        slices = []
        size = []
        extent = []
        data = zip(zip(*source_extent), zip(*target_extent), source_size)
        for (e1, e2), (c1, c2), s in data:
            espan = e2 - e1
            start = max(0, min(s, int(math.floor((c1 - e1) / espan * s))))
            stop = max(0, min(s, int(math.ceil((c2 - e1) / espan * s))))
            slices.append(slice(start, stop))
            size.append(stop - start)
            extent.append((e1 + start * espan, e1 + stop * espan))

        return dict(
            size=tuple(size),
            extent=tuple(extent),
            axes=(),
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

    def transform(self, extent, kind):
        """ 
        Return dictionary.

        It contains transformed size and extent when input size and
        extent are transformed from kind to kind. This is used for
        location determination.
        """
        return tuple(
            map(
                tuple,
                netCDF4.date2num(
                    netCDF4.num2date(extent, self.unit),
                    kind.unit,
                ),
            ),
        )
    
    def get_prep(self, source_size, source_extent, 
                 source_axes, target_extent, target_axes):
        """ 
        Return dictionary.

        Source and target must be of kind self.

        Remove values from source axes that are outside targets extent
        Adjust source extent and scale axes accordingly

        size: cropped size
        extent: cropped extent?
        axes: clipped axes
        slices: clipping slices based on axes & extent

        Here comes the fun: The time kind gets the responsibility of
        checking available space in the axes.

        The check is only performed if the extents after clipping are
        completely aligned, which will be the case after the source_view
        is used to create a target_view.
        """
        ((e1,), (e2,)) = extent
        ((c1,), (c2,)) = clip
        values = e1 + axes[0] * (e2 - e1)
        indices = np.where((values >= c1) * (values < c2) * (axes[0] != -1))[0]
        if indices.size:
            slices =  (slice(indices[0], indices[-1] + 1),)
        else:
            slices =  (slice(0, 0),)

        import ipdb; ipdb.set_trace() 

        return dict(
            size=(slices[0].stop - slices[0].start,),
            extent=extent,
            axes=tuple(a[s] for a, s in zip(axes, slices)),
            slices=slices,
        )
