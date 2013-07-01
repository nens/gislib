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
            dimensions.SpatialDimension(projection=28992, size=256),
            dimensions.TimeDimension(calendar='minutes since 200130401', size=1),
        ],
        dtype='f4',
        nodatavalue=np.finfo('f4').min,
    )
    location = (
        dimensions.Location(level=0, indices=(0, 0)),
        dimensions.Location(level=0, indices=(0,)),
    )
    print(structure.get_extent(location))
    locations = structure.get_locations(
        structure.get_extent(location),
        resolution=(513, 1),
    )
    for c, l in enumerate(locations):
        print(l)
        if c > 20:
            break
    exit()

    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass
    datastore = datastores.Datastore(storage=storage, structure=structure)

    


def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))
