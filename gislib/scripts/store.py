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

from gislib.store import stores
from gislib.store import storages
from gislib.store import structures

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
    frame = structures.Frame(
        dimensions=[
            structures.SpatialDimension(projection=28992, size=256),
            structures.TimeDimension(calendar='minutes since 20130401', size=1),
        ],
        dtype='f4',
        nodatavalue=np.finfo('f4').min,
    )

    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass

    store = stores.Store(storage=storage, frame=frame)
    location = structures.Location(frame=frame, sublocations=(
        structures.Sublocation(level=1, indices=(1, 1)),
        structures.Sublocation(level=1, indices=(1,)),
    ))
    for dataset in store.get_datasets(extent=location.extent, 
                                      resolution=frame.shape):
        print(dataset.data)



def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))
