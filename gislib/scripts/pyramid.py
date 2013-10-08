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

from gislib import pyramids
from gislib import progress

MEM_DRIVER = gdal.GetDriverByName(b'mem')

description = """
Commandline tool for working with gislib pyramids.
"""

logger = logging.getLogger(__name__)


def get_parser():
    """ Return argument parser. """
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument('targetpath', metavar='TARGET')
    parser.add_argument('sourcepaths',
                        nargs='*',
                        metavar='SOURCE')
    parser.add_argument('-b', '--blocksize',
                        nargs=2,
                        metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('-d', '--datatype')
    parser.add_argument('-n', '--nodatavalue')
    parser.add_argument('-p', '--projection')
    parser.add_argument('-t', '--tilesize',
                        nargs=2,
                        metavar=('WIDTH', 'HEIGHT'))
    return parser


def rounded(sourcepath, precision=2):
    """ Return rounded memory dataset. """
    # Read
    source = gdal.Open(sourcepath)
    target = MEM_DRIVER.CreateCopy('', source)

    # Round
    array = np.ma.masked_equal(
        target.GetRasterBand(1).ReadAsArray(),
        target.GetRasterBand(1).GetNoDataValue(),
    ).round(precision)
    target.GetRasterBand(1).WriteArray(array.filled(
        target.GetRasterBand(1).GetNoDataValue(),
    ))
    return target


def pyramid(targetpath, sourcepaths, blocksize,
            datatype, nodatavalue, projection, tilesize):
    """ Create or update pyramid. """
    indicator = progress.Indicator(len(sourcepaths), steps=1)
    pyramid = pyramids.Pyramid(path=targetpath)

    if not sourcepaths:
        return pyramid.add()

    kwargs = {}
    if blocksize:
        kwargs.update(blocksize=tuple(int(t) for t in blocksize))
    if datatype:
        kwargs.update(datatype=gdal.GetDataTypeByName(datatype))
    if nodatavalue:
        kwargs.update(nodatavalue=float(nodatavalue))
    if projection:
        kwargs.update(projection=projection)
    if tilesize:
        kwargs.update(tilesize=tuple(int(t) for t in tilesize))

    for i, sourcepath in enumerate(sourcepaths):
        logger.debug('Add: {}'.format(sourcepath))
        #dataset = gdal.Open(sourcepath)
        dataset = rounded(sourcepath)
        pyramid.add(dataset, sync=False, **kwargs)
        indicator.update()
    pyramid.sync()


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    pyramid(**vars(get_parser().parse_args()))
