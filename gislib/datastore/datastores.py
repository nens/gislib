# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import pickle

from gislib.datastore import aggregators
from gislib.datastore import chunks
from gislib.datastore import dimensions
from gislib.datastore import utils

class Datastore(object):
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

    STRUCTURE_KEY = 'structure'

    def __init__(self, storage, structure=None):
        """ initialize. """
        self.storage = storage

        if structure is None:
            self.structure = pickle.loads(
                self.storage.common[self.STRUCTURE_KEY],
            )
        else:
            self.verify_not_initialized()
            self.storage.common[self.STRUCTURE_KEY] = pickle.dumps(structure)
            self.structure = structure

    def verify_not_initialized(self):
        """ If the datastore already has a structure, raise an exception. """
        try:
            self.storage.common['structure']
        except IndexError:
            return  # That's expected.
        raise IOError('Datastore already has a structure!')
