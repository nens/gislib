# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from gislib.store import stores
from gislib.store import kinds

import numpy as np


def union(*dicts):
    """ Return union of dictionaries. """
    return reduce(
        lambda x, y: dict(x.items() + y.items()),
        dicts,
    )


def get_radar_guides(layout):
    """ Return guides tuple. """
    return dict(
        guides=(
            stores.Guide(
                kind=kinds.Space(proj=3857),
                size={'time': (1, 1),
                      'hybrid': (16, 16),
                      'space': (256, 256)}[layout]
            ),
            stores.Guide(
                kind=kinds.Time(unit='minutes since 2013-01-01'),
                base=4,
                size={'space': (1,),
                      'time': (65536,),
                      'hybrid': (256,)}[layout]
            ),
        ),
    )

base_radar = dict(dtype='f4', fill=np.finfo('f4').min)
time_radar = union(base_radar, get_radar_guides(layout='time'))
space_radar = union(base_radar, get_radar_guides(layout='space'))
hybrid_radar = union(base_radar, get_radar_guides(layout='hybrid'))

# Obsolete, use kwargs.
#class Ahn2(stores.Grid):
    #""" Template for AHN2 data """
    #def __init__(self):
        #self.dtype = 'f4'
        #self.fill = np.finfo(self.dtype).min
        #self.space = stores.SpaceDomain(
            #size=(256, 256),
            #proj=28992,
        #)
