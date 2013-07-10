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

    def put(self, path, value):
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
            tempfile.write(lz4.dumps(value))

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


class SchemaFileStorage(BaseFileStorage):
    """ Store key value data for a schema. """
    def __init__(self, path, schema, split):
        """ Sets the schema name. """
        super(SchemaFileStorage, self).__init__(path=path)
        self.schema = schema
        self.split = split

    def split_key(self, key, size=4, count=4):
        """ Split a key in directory parts and a file part. """
        total = size * count
        return textwrap.wrap(key[:total], size) + [key[total:]]

    def make_path(self, key):
        """
        Return a filepath corresponding to key.
        """
        paths = [self.path, self.schema]
        if self.split:
            paths.extend(self.split_key(key))
        else:
            paths.append(key)
        return os.path.join(*paths)

    def __setitem__(self, key, value):
        """ Put value. """
        path = self.make_path(key=key)
        return super(SchemaFileStorage, self).put(path=path, value=value)

    def __getitem__(self, key):
        """ Get value. """
        path = self.make_path(key=key)
        try:
            return super(SchemaFileStorage, self).get(path=path)
        except IOError:
            raise KeyError(key)

    def __delitem__(self, key):
        """ Delete value. """
        path = self.make_path(key=key)
        try:
            return super(SchemaFileStorage, self).delete(path=path)
        except IOError:
            raise KeyError('key')

    def first(self):
        """ Return named data of arbitrary chunk. """
        path = os.path.join(self.path, self.schema)
        for basedir, dirnames, filenames in os.walk(path):
            if filenames:
                return super(SchemaFileStorage, self).get(
                    path=os.path.join(basedir, filenames[0]),
                )

    def create_link(self, name, chunk, link):
        """ Create a symbolic link to a chunk """
        chunkpath = self.make_path(name=name, chunk=chunk)
        linkpath = self.make_path(name=name, chunk=link)
        os.symlink(chunkpath, linkpath)


class FileStorage(object):
    """
    A container for various storages that together form the file storage.
    """
    def __init__(self, path):
        """ Set storage path. """
        self.path = path

    def get_schema(self, schema, split=False):
        """ Return SchemaFileStorage object. """
        return SchemaFileStorage(path=self.path, schema=schema, split=split)
