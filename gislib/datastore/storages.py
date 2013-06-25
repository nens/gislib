# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import datetime
import hashlib
import lz4
import os

# ==============================================================================
# File storage classes
# ------------------------------------------------------------------------------
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
    def make_path(self, chunk):
        """ Return a filepath for a chunk. """
        key = chunk.get_key()
        paths = [self.path]
        # Add some directory structure
        paths.extend([key[4 * i: 4 * i + 4] for i in range(8)])
        paths.append(key + self.EXT)
        return os.path.join(*paths)

    def put(self, data, chunk=None):
        super(ChunkFileStorage, self).put(data=data,
                                          path=self.make_path(chunk))

    def get(self, chunk=None):
        return super(ChunkFileStorage, self).get(path=self.make_path(chunk))

    def delete(self, chunk=None):
        super(ChunkFileStorage, self).delete(path=self.make_path(chunk))


class StructureFileStorage(ChunkFileStorage):
    """ Store data in a single file. """
    FILENAME = 'structure'

    def make_path(self, chunk):
        return os.path.join(self.path, self.FILENAME)


class LocationFileStorage(ChunkFileStorage):
    """ Storage class for metadata. """
    EXT = '.meta'


class DataFileStorage(ChunkFileStorage):
    """ Storage class for chunkdata. """
    EXT = '.data'


class MetaFileStorage(ChunkFileStorage):
    """ Storage class for metadata. """
    EXT = '.meta'


class FileStorage(object):
    """
    A container for various storages that together form the file storage.
    """
    def __init__(self, path):
        self.structure = StructureFileStorage(path)
        self.location = LocationFileStorage(path)
        self.data = DataFileStorage(path)
        self.meta = MetaFileStorage(path)
