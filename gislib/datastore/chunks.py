# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import hashlib

"""
So, it is a bit complex, but here's how it works. The storage has attributes
We want to be able to
chunk.data.put(data)  # puts location as well ((3,6), 2)
chunk.data.get()
chunk.meta.put(meta)
chunk.meta.get()

chunk can be instantiated from a file. In that case, the location must be unpickled.
location must always be written when putting data.

storage must be able to produce a single chunk? Na, then we need to import the chunks in the storages. Don't want to.

When creating a chunk from the storage, we need to get the meta and the data for an arbitrary chunk from the storage. single chunk from the storage.

Hey, what about separate storage for base chunks and aggregated chunks? Base chunks can easily be iterated over for copying, then. Maybe later.

Store must be able to produce one or more tiles.

Store produces locations; with a location a chunk can be instantiated. It must receive a storage as well. 

"""


class Data(object):
    """ Uses the chunks storage to store and retrieve data."""
    def __init__(self, chunk):
        self.chunk = chunk

    def put(self, data):
        """ Store data in storage. """
        self.chunk.storage.data.put(self.chunk, data)

    def get(self):
        """ Get data from storage. """
        self.chunk.storage.data.get(self.chunk)

class Meta(object):
    """ Uses the chunks storage to store and retrieve meta."""
    def __init__(self, chunk):
        self.chunk = chunk

    def put(self, meta):
        """ Store meta in storage. """
        self.chunk.storage.meta.put(self.chunk, meta)

    def get(self):
        """ Get meta from storage. """
        self.chunk.storage.meta.get(self.chunk)


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
