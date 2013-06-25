# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.



class BaseDimension(object):
    def __init__(self, **kwargs):
        """
        Handle dimension mapping to files and blocks

        Non-equidistant dimensions must have at least one aggregator.
        """
        self.equidistant = kwargs.get('equidistant', True)
        self.aggregators = kwargs.get('aggregators', [])
        self.offset = kwargs.get('offset', 0)


class UnitDimension(BaseDimension):
    """
    Base for all dimension classes
    """
    def __init__(self, unit, **kwargs):
        self.unit = unit
        super(UnitDimension, self).__init__(**kwargs)
        

class TimeDimension(BaseDimension):
    """ 
    Dimension with coards calender units.
    """
    def __init__(self, calendar, **kwargs):
        self.calendar = calendar
        super(TimeDimension, self).__init__(**kwargs)


class SpatialDimension(BaseDimension):
    """
    Dimension with projection units, suitable for gdal reprojection.
    """
    def __init__(self, projection, **kwargs):
        self.projection = projection
        super(SpatialDimension, self).__init__(**kwargs)
