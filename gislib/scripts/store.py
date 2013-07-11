# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import shutil

import numpy as np

from gislib.store import core
from gislib.store import stores
from gislib.store import storages

description = """
Commandline tool for working with nens/gislib stores.
"""

logging.root.level = logging.DEBUG
logger = logging.getLogger(__name__)

def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument('targetpath', metavar='TARGET')
    parser.add_argument('sourcepaths',
                        nargs='+',
                        metavar='SOURCE')
    return parser


def command(targetpath, sourcepaths):
    """ Do something spectacular. """
    storage = storages.FileStorage(targetpath)
    spatial_scale = core.SpatialScale(projection=28992, size=(256,256))
    time_scale = core.TimeScale(calendar='minutes since 20130401', size=(1,))
    scales = [core.FrameScale(spatial_scale), core.FrameScale(time_scale)]
    metric = core.FrameMetric(scales=scales)
    frame = core.Frame(metric=metric,
                       dtype='f4',
                       nodatavalue=np.finfo('f4').min)

    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass

    store = stores.Store(storage=storage, frame=frame)
    location = core.Location(parts=(
        core.Sublocation(level=1, indices=(1, 1)),
        core.Sublocation(level=1, indices=(1,)),
    ))
    for dataset in store.get_datasets(extent=metric.get_extent(location), 
                                      size=metric.size):
        print(dataset.data)



def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))
