# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import hashlib
import pickle


class Data(object):
    """ Uses the chunks storage to store and retrieve data."""
    def __init__(self, chunk):
        self.chunk = chunk

    def put(self, data):
        """ 
        Store data in storage.
        Here we make sure that location is written, too. And convert stuff.
        """
        self.chunk.storage.data.put(chunk=self.chunk, data=data)
        self.chunk.storage.location.put(
            chunk=self.chunk, 
            data=pickle.dumps(self.chunk.location),
        )

    def get(self):
        """ Get data from storage. """
        return self.chunk.storage.data.get(self.chunk)


class Meta(object):
    """ Uses the chunks storage to store and retrieve meta."""
    def __init__(self, chunk):
        self.chunk = chunk

    def put(self, data):
        """ Store meta in storage. """
        self.chunk.storage.meta.put(
            chunk=self.chunk,
            data=pickle.dumps(data))

    def get(self):
        """ Get meta from storage. """
        return pickle.loads(self.chunk.storage.meta.get(chunk=self.chunk))


class Chunk(object):
    """
    A chunk of data stored with compression.
    """
    def __init__(self, storage, location):
        self.location = location
        self.storage = storage
        self.data = Data(self)
        self.meta = Meta(self)

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
        return hashlib.md5(str(self.location)).hexdigest()
