# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

def AbstractDimension(object):
    def __init__(self, **kwargs):
        """
        Handle dimension mapping to files and blocks

        Non-equidistant dimensions must have at least one aggregator.
        """
        self.equidistant = kwargs.get(equidistant, True)
        self.aggregators = kwargs.get(aggregators, [])
        self.offset = kwargs.get(offset, 0)


class Dimension(object):
    """
    Base for all dimension classes
    """
    def __init__(self, unit, **kwargs):
        self.unit = unit
        super(self, Dimension).__init__(**kwargs)
        

class TimeDimension(Dimension):
    """ 
    Dimension with coards calender units.
    """
    def __init__(self, calendar):
        self.calendar = calendar
        super(self, TimeDimension).__init__(**kwargs)


class SpatialDimension(Dimension):
    """
    Dimension with projection units, suitable for gdal reprojection.
    """
    def __init__(self, unit):
        self.projection = projection
        super(self, SpatialDimension).__init__(**kwargs)
