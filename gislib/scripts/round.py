# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

from osgeo import gdal
import numpy as np

TIF_DRIVER = gdal.GetDriverByName(b'gtiff')
MEM_DRIVER = gdal.GetDriverByName(b'mem')

logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description="Round to 2 decimals."
    )
    parser.add_argument('sourcepath', metavar='SOURCE')
    parser.add_argument('targetpath', metavar='TARGET')
    return parser


def command(sourcepath, targetpath):
    """ Do something spectacular. """
    # Read
    source = gdal.Open(sourcepath)
    target = MEM_DRIVER.CreateCopy('', source)

    # Round
    array = np.ma.masked_equal(
        target.GetRasterBand(1).ReadAsArray(),
        target.GetRasterBand(1).GetNoDataValue(),
    ).round(2)
    target.GetRasterBand(1).WriteArray(array.filled(
        target.GetRasterBand(1).GetNoDataValue(),
    ))

    # Write
    TIF_DRIVER.CreateCopy(targetpath, target, 1, [
        'COMPRESS=DEFLATE',
    ])


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    command(**vars(get_parser().parse_args()))
