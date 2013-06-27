# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import hashlib
import pickle


class Chunk(object):
    """
    A chunk of data stored with compression.
    """
    DATA = 'data'
    META = 'meta'
    LOCATION = 'location'

    @classmethod
    def first(cls, storage):
        """ Return an arbitrary chunk from storage. """
        location = pickle.loads(storage.first(name=cls.LOCATION))
        return cls(storage=storage, location=location)

    def __init__(self, storage, location):
        self.location = location
        self.storage = storage
        self.key = hashlib.md5(str(self.location)).hexdigest()

    def __getitem__(self, name):
        """ Get data for this chunk. """
        try:
            return self.storage.chunks.get(chunk=self, name=name)
        except IOError:
            raise KeyError(name, chunk.key)

    def __setitem__(self, name, data):
        """ Set or delete data for this chunk. """
        if data is None:
            # Delete
            self.storage.chunks.delete(chunk=self, name=name)
            if name == self.DATA:
                self[self.LOCATION] = None
        else:
            # Put
            self.storage.chunks.put(chunk=self, name=name, data=data)
            if name == self.DATA:
                self[self.LOCATION] = pickle.dumps(self.location)
        

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
