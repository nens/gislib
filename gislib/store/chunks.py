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

    @classmethod
    def first(cls, store):
        """ Return an arbitrary chunk from storage. """
        location = store.structure.loads(store.storage.first(name=cls.DATA))
        return cls(store=store, location=location)
    
    @property
    def chunksize(self):
        """ Return bytesize. """
        raise NotImplementedError

    @property
    def shape(self):
        """ Return shape based on structure. """
        raise NotImplementedError

    def __init__(self, store, location):
        self.location = location
        self.store = store
        self.key = hashlib.md5(str(self.location)).hexdigest()

    def __getitem__(self, name):
        """ Get data for this chunk. """
        try:
            return self.store.storage.chunks.get(
                chunk=self, name=name
            )[self.store.structure.loc_size:]
        except IOError:
            raise KeyError(name, self.key)

    def __setitem__(self, name, data):
        """ Set data for this chunk. """
        self.store.storage.chunks.put(
            chunk=self,
            name=name,
            data=self.store.structure.dumps(self.location) + data,
        )

    
    def __delitem__(self, name, data):
        """ Del data for this chunk. """
        self.store.storage.chunks.delete(chunk=self, name=name, data=data)

    def __unicode__(self):
        return ';'.join(','.join([unicode(l)] + map(unicode, i)) 
                        for l, i in self.location)

    def __str__(self):
        return self.__unicode__()
        
    def get_parent(self, dimension=0, levels=1):
        """ Return parent chunk for a dimension. """
        parent_location = self.store.structure.get_parent_location(
            location=self.location, dimension=dimension, levels=levels,
        )
        return Chunk(store=self.store, location=parent_location)

    def get_children(self, dimension=0):
        """ Return child chunks for a dimension. """
        child_locations = self.store.structure.get_child_locations(
            location = self.location, dimension=dimension,
        )
        return [Chunk(store=self.store,
                      location=location)
                for location in child_locations]

    def get_root(self):
        """ 
        Return the root chunk. It only works if the data is fully
        aggregated in all dimensions to a single chunk at the highest
        level.
        """
        self[self.DATA]  # Crash if we don't even have data ourselves.
        # Get parents in all dimensions until nodata.
        root = self
        for i in range(len(self.store.structure.dimensions)):
            begin = 0
            end = 1
            # First begin extending end until non-existing chunk found.
            while True:
                try:
                    root.get_parent(dimension=i, levels=end)[self.DATA]
                except KeyError:
                    break
                end = end * 2
            while end - begin != 1:
                # Now begin testing the middle of end until end - begin == 1 again.
                middle = (begin + end) // 2
                try:
                    root.get_parent(dimension=i, levels=middle)[self.DATA]
                except KeyError:
                    end = middle
                    continue
                begin = middle
            if begin:
                root = root.get_parent(dimension=i, levels=begin)
        return root
            
