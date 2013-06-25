# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging

import numpy as np

from gislib.datastore import datastores
from gislib.datastore import dimensions
from gislib.datastore import storages
from gislib.datastore import structures

description = """
Commandline tool for working with nens/gislib datastores.
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
    structure = structures.Structure(
        dimensions=[
            dimensions.SpatialDimension(projection=28992),
            dimensions.TimeDimension(calendar='minutes since 200130401'),
        ],
        chunkshape=(256, 256, 1),
        dtype='f4',
        nodatavalue=np.finfo('f4').min,
    )

    #datastore = datastores.Datastore(storage=storage, structure=structure)
    datastore = datastores.Datastore(storage=storage)
    #datastore.add(sourcepaths)


def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))
