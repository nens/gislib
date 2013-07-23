# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import shutil

from osgeo import gdal

from gislib.store import stores
from gislib.store import adapters
from gislib.store import storages
from gislib.store import datasets
from gislib.store import grids

description = """
Commandline tool for working with nens/gislib stores.
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


def fill(targetpath, sourcepaths):
    """ Do something spectacular. """
    storage = storages.FileStorage(targetpath)
    grid = grids.HybridRadar2013()
    print(grid.guides)
    exit()
    
    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass

    store = stores.Store(storage=storage, grid=grid)
    adapter = adapters.GDALAdapter(sourcepaths=sourcepaths)
    store.add_from(adapter=adapter)


def load(targetpath, sourcepaths):
    """ Do something spectacular. """
    storage = storages.FileStorage(targetpath)
    store = stores.Store(storage=storage)

    from PIL import Image
    from matplotlib import cm, colors
    from gislib import rasters
    widths = [1999, 2000]
    extent = (((640250, 6804915), (461495,6806956)), ((0,), (1,)))
    size = ((1900, 2400), (1,))
    dataset = store.frame.get_empty_dataset(extent=extent, size=size)
    import ipdb; ipdb.set_trace() 
    store.fill_into(dataset)
    repro = dataset.data[:,:,0].transpose()
    normalize = colors.Normalize()
    Image.fromarray(cm.gist_earth(normalize(repro[::4,::4]), bytes=True)).show()


def main():
    """ Call command with args from parser. """
    fill(**vars(get_parser().parse_args()))
    #load(**vars(get_parser().parse_args()))

