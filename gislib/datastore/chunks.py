# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import hashlib
import pickle


class Chunk(object):
    """
    A chunk of data stored with compression.
    """

    @classmethod
    def first(cls, storage):
        """ Return an arbitrary chunk from storage. """
        location = pickle.loads(storage.location.first())
        return cls(storage=storage, location=location)

    def __init__(self, storage, location):
        self.location = location
        self.storage = storage
        self.key = hashlib.md5(str(self.location)).hexdigest()

    def __getitem__(self, name):
        """ Get data for this chunk. """
        return self.storage.chunks.get(chunk=self, name=name)

    def __setitem__(self, name, data):
        """ Set or delete data for this chunk. """
        # Delete
        if data is None:
            return self.storage.chunks.delete(chunk=self, name=name)
        # Write
        self.storage.chunks.put(chunk=self, name=name, data=data)

    def get_parent(self, dimension=None, aggregator=None):
        """
        It depends on the dimension and the aggregator which superchunk
        is returned.
        """
        pass

    def get_children(self):
        """
        def get_parent(self, dimension, aggregator):
        """

    def get_key(self):
        """
        Return a string.

        The string can be used to identify the chunk in the storage.
        """
