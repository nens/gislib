# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import datetime
import hashlib
import lz4
import os
import textwrap

# =============================================================================
# File storage classes
# -----------------------------------------------------------------------------
class BaseFileStorage(object):
    """
    Base class for file storage. Handles the absolute path creation,
    compression and atomic replacement of files.
    """
    def __init__(self, path):
        """ Set the path. """
        self.path = path

    def put(self, path, data):
        """ Safe writing using tempfile and atomic move. """
        # Create directory if necessary
        try:
            os.makedirs(os.path.dirname(path))
        except OSError:
            pass

        # Prepare temporary file using path and a timestamp
        timestamp = datetime.datetime.now().isoformat()
        temppath = os.path.join(
            self.path,
            '.' + hashlib.md5(timestamp + path).hexdigest()
        )
        with open(temppath, 'wb') as tempfile:
            tempfile.write(lz4.dumps(data))

        # By using an atomic move / rename operation, there is no risk
        # of reading corrupted data, only outdated data.
        os.rename(temppath, path)

    def get(self, path):
        """ Get decompressed data """
        with open(path, 'rb') as _file:
            return lz4.loads(_file.read())

    def delete(self, path):
        """ Removal """
        os.remove(path)
        os.removedirs(os.path.dirname(path))


class ChunkFileStorage(BaseFileStorage):
    """ Store chunk level data. """

    NAME = 'chunks'

    def split_key(self, key, size=4, count=4):
        """ Split a key in directory parts and a file part. """
        total = size * count
        return textwrap.wrap(key[:total], size) + [key[total:]]

    def make_path(self, name, chunk):
        """
        Return a filepath for a chunk.

        The path consists of:
        - Fixed part indicating chunk data
        - Variable part based on name
        - Variable parts based on chunk key

        The last part is also the filename.
        """
        paths = [self.path, self.NAME, name]
        #paths.extend(self.split_key(chunk.key))
        paths.append(chunk.key)
        return os.path.join(*paths)

    def put(self, name, chunk, data):
        """ Put chunk data. """
        path = self.make_path(name=name, chunk=chunk)
        super(ChunkFileStorage, self).put(path=path, data=data)

    def get(self, name, chunk):
        """ Get chunk data. """
        path = self.make_path(name=name, chunk=chunk)
        return super(ChunkFileStorage, self).get(path=path)

    def delete(self, name, chunk):
        """ Delete chunk data. """
        path = self.make_path(name=name, chunk=chunk)
        return super(ChunkFileStorage, self).delete(path=path)

    def create_link(self, name, chunk, link):
        """ Create a symbolic link to a chunk """
        chunkpath = self.make_path(name=name, chunk=chunk)
        linkpath = self.make_path(name=name, chunk=link)
        os.symlink(chunk, link)
        


class CommonFileStorage(BaseFileStorage):
    """ Store datastore common data. """

    NAME = 'common'

    def make_path(self, name):
        return os.path.join(self.path, self.NAME, name)

    def __getitem__(self, name):
        """ Get a common value. """
        path = self.make_path(name=name)
        try:
            return super(CommonFileStorage, self).get(path=path)
        except IOError:
            raise KeyError(name)

    def __setitem__(self, name, data):
        """ Set a common value. """
        path = self.make_path(name=name)
        return super(CommonFileStorage, self).put(path=path, data=data)

    def __delitem__(self, name, data):
        """ Delete a common value. """
        return super(CommonFileStorage, self).delete(path=path)


class FileStorage(object):
    """
    A container for various storages that together form the file storage.
    """
    def __init__(self, path):
        self.path = path
        self.base = BaseFileStorage(path)
        self.common = CommonFileStorage(path)
        self.chunks = ChunkFileStorage(path)

    def first(self, name):
        """ Return named data of arbitrary chunk. """
        path = os.path.join(self.path, self.chunks.NAME, name)
        for basedir, dirnames, filenames in os.walk(path):
            if filenames:
                return self.base.get(os.path.join(basedir, filenames[0]))
