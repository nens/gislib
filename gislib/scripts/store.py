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
    config = frames.Config(domains=[
        frames.Domain(domain=spatial_domain, size=(256, 256)),
        frames.Domain(domain=time_domain, size=(1,))],
    )
    frame = frames.Frame(config=config,
                         dtype='f4',
                         fill_value=np.finfo('f4').min)

    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass

    store = stores.Store(storage=storage, frame=frame)
    adapter = adapters.GDALAdapter(sourcepaths=sourcepaths)
    store.add_from(adapter=adapter)


def load(targetpath, sourcepaths):
    """ Do something spectacular. """
    storage = storages.FileStorage(targetpath)
    store = stores.Store(storage=storage)
    extent = (
        #((89000.0, 472500.0), (90000.0,473750.0)),
        ((85000.0, 471250.0), (86000.0, 472500.0)),
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
    import ipdb; ipdb.set_trace() 
    Image.fromarray(cm.jet(normalize(dataset.data[:, :, 0]), bytes=True)).show()
    mon.check('overall') 


def main():
    """ Call command with args from parser. """
    #fill(**vars(get_parser().parse_args()))
    load(**vars(get_parser().parse_args()))

