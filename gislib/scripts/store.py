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
from gislib.store import storages

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


def command(targetpath, sourcepaths):
    """ Do something spectacular. """
    storage = storages.FileStorage(targetpath)
    spatial_domain = domains.Space(projection=28992, size=(256, 256))
    time_domain = domains.Time(calendar='minutes since 20130401', size=(1,))
    config = frames.Config(domains=[
        frames.Domain(spatial_domain),
        frames.Domain(time_domain)],
    )
    frame = frames.Frame(config=config,
                         dtype='f4',
                         nodatavalue=np.finfo('f4').min)

    try:
        shutil.rmtree(targetpath)
    except OSError:
        pass

    store = stores.Store(storage=storage, frame=frame)
    location = frames.Location(parts=(
        frames.Sublocation(level=1, indices=(1, 1)),
        frames.Sublocation(level=1, indices=(1,)),
    ))
    for dataset in store.get_datasets(extent=config.get_extent(location),
                                      size=((256, 256), (32,))):
        print(dataset.config.extent)


def main():
    """ Call command with args from parser. """
    command(**vars(get_parser().parse_args()))
