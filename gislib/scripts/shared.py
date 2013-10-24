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


def read_as_shared_array(dataset):
    """
    Return numpy array.

    Puts the array data in shared memory for use with multiprocessing.
    """
    # Derive array properties from the dataset
    dtype = np.dtype(gdal_array.flip_code(
        dataset.GetRasterBand(1).DataType,
    ))
    shape = (dataset.RasterCount,
             dataset.RasterYSize,
             dataset.RasterXSize)
    size = shape[0] * shape[1] * shape[2] * dtype.itemsize
    # Create shared memory array and read into it
    return dataset.ReadAsArray(buf_obj=np.frombuffer(
        multiprocessing.RawArray('b', size), dtype
    ).reshape(*shape))


def func(shared):
    """ Modify the array. """
    print(shared.ReadAsArray()[0,0])
    shared.GetRasterBand(1).Fill(24)
    shared.FlushCache()


def command(sourcepath):
    """ Do something spectacular. """
    dataset = gdal.Open(sourcepath)
    array = read_as_shared_array(dataset)
    shared = rasters.array2dataset(array, crs=None, extent=None)
    shared.SetProjection(dataset.GetProjection())
    shared.SetGeoTransform(dataset.GetGeoTransform())
    process = multiprocessing.Process(
        target=func,
        kwargs=dict(shared=shared),
    )
    process.start()
    process.join()
    print(shared.ReadAsArray()[0,0])


def main():
    """ Call command with args from parser. """
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    command(**vars(get_parser().parse_args()))
