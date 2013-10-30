# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import multiprocessing
import sys

from osgeo import gdal
from osgeo import gdal_array
import numpy as np

from gislib import rasters
from gislib import pyramids

TIF_DRIVER = gdal.GetDriverByName(b'gtiff')
MEM_DRIVER = gdal.GetDriverByName(b'mem')

logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        'sourcepath',
        metavar='SOURCE',
    )
    # Add arguments here.
    return parser


def func(smd):
    """ Modify the array. """
    gsdp.levels[-1].array[:] = 2
    pass


def init(sdp):
    """ Create a home for the single dataset pyramid. """
    global gsdp
    gsdp = sdp


def command(sourcepath):
    """ Do something spectacular. """
    dataset = gdal.Open(sourcepath)
    sdp = pyramids.SingleDatasetPyramid(dataset)
    from arjan.monitor import Monitor; mon = Monitor() 
    pool = multiprocessing.Pool(initializer=init, initargs=(sdp,))
    pool.map(func, range(1000))
    mon.check('') 
    print(sdp.levels[-1].array)


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    command(**vars(get_parser().parse_args()))
