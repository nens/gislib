# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import pickle

from gislib.store import aggregators
from gislib.store import chunks
from gislib.store import utils

class Store(object):
    """
    Raster object with optional arguments including
    time. Choices to be made:

    chunksize, filesize
    dimensions: ['time', 'spatial', 'spatial']
    calendars:
    equidistant time or not?
    equidistant x y or not?
    z or not?
    coupled x y
    """

    FRAME = 'frame'

    def __init__(self, storage, frame=None):
        """
        Separate schemas in the storage are placed as attributes on
        the store.
        """
        # Init schemas
        for name in ('databox', 'metabox', 'config', 'metadata'):
            setattr(self, name, storage.get_schema(name))
        
        if frame is None:
            self.frame = pickle.loads(self.config[self.FRAME])
        return

        # Write config
        self.verify_not_initialized()
        self.config[self.FRAME] = pickle.dumps(frame)
        self.structure = structure

    def verify_not_initialized(self):
        """ If the store already has a structure, raise an exception. """
        try:
            self.config[self.STRUCTURE]
        except KeyError:
            return  # That's expected.
        raise IOError('Store already has a structure!')
