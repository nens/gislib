# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from gislib.store import stores

import numpy as np


class BaseRadar2013(stores.Grid):
    """ Template for radar 2013 data. """
    PROJECTION = 28992

    def __init__(self):
        self.dtype = 'f4'
        self.fill = np.finfo(self.dtype).min
        self.guides = [
            stores.SpaceGuide(
                size=self.SIZE['space'],
                projection=self.PROJECTION,
            ),
            stores.TimeGuide(
                base=4,
                equidistant=False,                  
                size=self.SIZE['time'],             
                calendar='minutes since 20130101',
            ),
        ]


class HybridRadar2013(BaseRadar2013):
    SIZE = dict(time=(256,),
                space=(16, 16))



class TimeRadar2013(BaseRadar2013):
    SIZE = dict(time=(65536,),
                space=(1, 1))


class SpaceRadar2013(BaseRadar2013):
    SIZE = dict(time=(1,),
                space=(256, 256))


class Ahn2(stores.Grid):
    """ Template for AHN2 data """
    def __init__(self):
        self.dtype = 'f4'
        self.fill = np.finfo(self.dtype).min
        self.space = stores.SpaceDomain(
            size=(256, 256),
            projection=28992,
        )
