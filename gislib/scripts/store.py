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

description = """
Commandline tool for working with nens/gislib stores.
"""

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
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


def clean(targetpath, sourcepaths):
    """ Clean up. """
    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass


def fill(targetpath, sourcepaths):
    """ Fill. """
    store = stores.Pyramid(path=targetpath)
    from arjan.monitor import Monitor; mon = Monitor()
    for i, sourcepath in enumerate(sourcepaths):
        dataset = gdal.Open(sourcepath)
        store.add(dataset)
        mon.check(sourcepath)


def load(targetpath, sourcepaths):
    """ Load. """
    store = stores.Pyramid(path=targetpath)
    driver = gdal.GetDriverByName(b'mem')
    for i, sourcepath in enumerate(sourcepaths):
        original = gdal.Open(sourcepath)
        dataset = driver.CreateCopy('', original)
        band = dataset.GetRasterBand(1)
        band.Fill(band.GetNoDataValue())
        store.warpinto(dataset)


def main():
    """ Call command with args from parser. """
    #clean(**vars(get_parser().parse_args()))
    fill(**vars(get_parser().parse_args()))
    #load(**vars(get_parser().parse_args()))
