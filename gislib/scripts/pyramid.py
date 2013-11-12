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
    parser.add_argument('target_path', metavar='TARGET')
    parser.add_argument('source_paths',
                        nargs='*',
                        metavar='SOURCE')
    parser.add_argument('-b', '--block-size',
                        nargs=2,
                        metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('-d', '--data-type')
    parser.add_argument('-n', '--no-data-value')
    parser.add_argument('-p', '--projection')
    parser.add_argument('-r', '--raster-size',
                        nargs=2,
                        metavar=('WIDTH', 'HEIGHT'))
    return parser


def rounded(source_path, precision=2):
    """ Return rounded memory dataset. """
    # Read
    source = gdal.Open(source_path)
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


def progress(count, total):
    """ Log a debug line about the progress. """
    logger.debug('{} / {} ({:.1f}%)'.format(
        count,
        total,
        100 * count / total,
    ))


def pyramid(target_path, source_paths, block_size,
            data_type, no_data_value, projection, raster_size):
    """ Create or update pyramid. """
    pyramid = pyramids.Pyramid(path=target_path)

    kwargs = {}
    if block_size:
        kwargs.update(block_size=tuple(int(t) for t in block_size))
    if data_type:
        kwargs.update(data_type=gdal.GetDataTypeByName(data_type))
    if no_data_value:
        kwargs.update(no_data_value=float(no_data_value))
    if projection:
        kwargs.update(projection=projection)
    if raster_size:
        kwargs.update(raster_size=tuple(int(t) for t in raster_size))

    total = len(source_paths)
    for count, source_path in enumerate(source_paths):
        progress(count=count, total=total)
        logger.debug('Add: {}'.format(source_path))
        #dataset = gdal.Open(source_path)
        dataset = rounded(source_path)
        pyramid.add(dataset, sync=False, **kwargs)
    progress(count=total, total=total)
    pyramid.sync()


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    pyramid(**vars(get_parser().parse_args()))
