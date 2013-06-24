# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import hashlib


class Chunk(object):
    """
    A chunk of data stored with compression.
    """
    structure # Do we need to know what 
    extent # A tuple containing the limits for the chunks dimensions.
    data

    def get_superchunk(self, aggregator):
        """ It depends on the aggregator which superchunk is returned. """

    def get_subchunks(self):
        """ """

    def get_key(self):
        """
        Return a key for the storage. Note that currently there is no
        way to determine the extents of a chunk using the key. Given
        the key, the data can be found, but not the extents.
        """
        return hashlib.md5(str(self.extent)).hexdigest()
        


