# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import shutil
import sys

from osgeo import gdal

from gislib.store import stores
from gislib.store import storages

description = """
Commandline tool for working with nens/gislib stores.
"""

logging.basicConfig(stream=sys.stderr , level=logging.DEBUG)
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


def fill(targetpath, sourcepaths):
    """ Do something spectacular. """
    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass
    
    store = stores.Store(path=targetpath)
    store.init_raster()

    for i, sourcepath in enumerate(sourcepaths):
        dataset = gdal.Open(sourcepath)
        store.raster[0] = dataset
        if i == 0:
            exit()

def load(targetpath, sourcepaths):
    """ Do something spectacular. """



def main():
    """ Call command with args from parser. """
    fill(**vars(get_parser().parse_args()))
    #load(**vars(get_parser().parse_args()))
