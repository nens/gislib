# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.


class BaseAggregator(object):
    """
    A class that handles aggregation from one zoomlevel to another. Examples are:
    - Percentile aggregator
    - Median aggregator
    - Maximum aggregator
    - Interpolating aggregator

    Aggregators are also responsible for cleaning of nodata chunks. It is
    still open if the aggregator returns a chunk with nodata or removes
    a chunk via the storage. Probably the first one.
    """
    pass


class SimplifyingAggregator(BaseAggregator):
    """ Use ogr simplify to simplify timeseries data. """
    pass


class ExtremumAggregator(BaseAggregator):
    """ Use numpy to pick the extreme value from 1D or 2D data. """
    pass


