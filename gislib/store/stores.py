# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import pickle


class Store(object):
    """
    Raster object with optional arguments including.
    """

    FRAME = 'frame'

    def __init__(self, storage, frame=None):
        """
        Separate schemas in the storage are placed as attributes on
        the store.
        """
        # Init schemas
        for schema in ('databox', 'metabox', 'config', 'metadata'):
            split = False if schema == 'config' else True
            setattr(self, schema, storage.get_schema(schema, split=split))

        # Add a schema for each aggregator when they are added.
        if frame is None:
            self.frame = pickle.loads(self.config[self.FRAME])
            return

        # Write config
        self.verify_not_initialized()
        self.config[self.FRAME] = pickle.dumps(frame)
        self.frame = frame

    def verify_not_initialized(self):
        """ If the store already has a structure, raise an exception. """
        try:
            self.config[self.FRAME]
        except KeyError:
            return  # That's expected.
        raise IOError('Store already has a structure!')


    def get_datasets(self, extent, size):
        """ Return dataset generator. """
        for location in self.frame.get_locations(extent, size):
            try:
                data = self.databox[location.key]
                yield self.frame.to_dataset(data)
            except KeyError:
                yield self.frame.get_empty_dataset(location)

                
                
        
