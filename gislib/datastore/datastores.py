# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

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

        chunk1 = chunks.Chunk(storage=self.storage, location='onzin')
        chunk1['data'] = 'de data'
        chunk2 = chunks.Chunk(storage=self.storage, location='aaaaaa')
        chunk2['data'] = 'de data2'
        chunk = chunks.Chunk.first(storage=self.storage)
        print(chunk['data'])
        chunk1['data'] = None
        chunk2['data'] = None



    def verify_not_initialized(self):
        """ If the datastore already has a structure, raise an exception. """
        try:
            self.storage.common[self.STRUCTURE]
        except KeyError:
            return  # That's expected.
        raise IOError('Datastore already has a structure!')
