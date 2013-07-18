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
from osgeo import gdal

from gislib.store import domains
from gislib.store import frames
from gislib.store import stores
from gislib.store import adapters
from gislib.store import storages
from gislib.store import datasets

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
    spatial_domain = domains.Space(projection=28992)
    time_domain = domains.Time(calendar='minutes since 20130401')
    config = frames.Config(
        domains=[
            frames.Domain(domain=spatial_domain, size=(256, 256)),
            frames.Domain(domain=time_domain, size=(1,))
        ],
        fill=np.finfo('f4').min,
    )
    frame = frames.Frame(config=config, dtype='f4')

    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass

    store = stores.Store(storage=storage, frame=frame)
    adapter = adapters.GDALAdapter(sourcepaths=sourcepaths)
    store.add_from(adapter=adapter)


def load(targetpath, sourcepaths):
    """ Do something spectacular. """
    from gislib import rasters
    gdal_dataset = gdal.Open(sourcepaths[0])
    gdal_extent = rasters.DatasetGeometry.from_dataset(gdal_dataset).extent
    store_extent = (gdal_extent[:2], gdal_extent[2:])
    original = gdal_dataset.ReadAsArray()
    

    storage = storages.FileStorage(targetpath)
    store = stores.Store(storage=storage)
    extent = (
        store_extent,
        ((0,), (1,)),
    )
    size = (
        (2000, 2500),
        (1,),
    )
    dataset = store.frame.get_empty_dataset(extent=extent, size=size)
    from PIL import Image
    from matplotlib import cm, colors
    normalize = colors.Normalize()
    from arjan.monitor import Monitor; mon = Monitor() 
    store.fill_into(dataset)
    mon.check('overall')
    repro = dataset.data[:,:,0].transpose()
    Image.fromarray(cm.jet(normalize(original[::4,::4]), bytes=True)).show()
    Image.fromarray(cm.jet(normalize(repro[::4,::4]), bytes=True)).show()


def main():
    """ Call command with args from parser. """
    fill(**vars(get_parser().parse_args()))
    load(**vars(get_parser().parse_args()))

