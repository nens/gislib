# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging

from osgeo import gdal

from gislib import rasters
from gislib import pyramids

description = """
    Commandline tool for working with nens/gislib pyramid datasets.
"""

logging.root.level = logging.DEBUG

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


def pyramid(targetpath, sourcepaths):
    """ Do something spectacular. """
    from arjan.monitor import Monitor; mon = Monitor()
    #source = gdal.Open(sourcepaths[0])
    #pyramid = raster.Pyramid(targetpath)
    #pyramid.add(source)
    pyramid = pyramids.Pyramid(targetpath)
    pyramid.add(sourcepaths)

    mon.check('Pyramid2') 


def main():
    """ Call command with args from parser. """
    pyramid(**vars(get_parser().parse_args()))
