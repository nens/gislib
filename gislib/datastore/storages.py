# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

class BaseStorage(object):
    """
    BaseClass for storage of chunks and datastore configuration.

    A storage guarantees storage for a chunk.

    

    
    Storage facility for loading and sa
    """
    def load(self, chunk):
        """ Return a chunk object. """
        raise NotImplementedError

    def save(self, chunk) 


class FileStorage(object):
    

