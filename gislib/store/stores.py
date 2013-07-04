# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import pickle

from gislib.store import aggregators
from gislib.store import chunks
from gislib.store import dimensions
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

    STRUCTURE = 'structure'

    def __init__(self, storage, structure=None):
        """ initialize. """
        self.storage = storage

        if structure is None:
            self.structure = pickle.loads(
                self.storage.common[self.STRUCTURE],
            )
        else:
            self.verify_not_initialized()
            self.storage.common[self.STRUCTURE] = pickle.dumps(structure)
            self.structure = structure

    


    def verify_not_initialized(self):
        """ If the store already has a structure, raise an exception. """
        try:
            self.storage.common[self.STRUCTURE]
        except KeyError:
            return  # That's expected.
        raise IOError('Store already has a structure!')
