# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import datetime
import hashlib
import lz4
import pickle

class BaseStorage(object):
    """
    BaseClass for storage of chunks and datastore configuration.

    A storage guarantees storage for a chunk.
    Storage facility for loading and sa
    """
    def get(self, chunk):
        """ Fill a chunk object with data. """
        raise NotImplementedError

    def put(self, chunk):
        """ Save chunks data in the store. """
        raise NotImplementedError

    def delete(self, chunk):
        """ Delete a chunk of data from store. """
        raise NotImplementedError

    def save(self, structure):
        """ Store a structure object """
        raise NotImplementedError

    def load(self):
        """ Return a structure object. """
        raise NotImplementedError


class FileStorage(object):

    STRUCTURE_FILENAME = 'structure.pkl'
    def __init__(self, path):
        self.path = path
        self.structure_filepath = os.path.join(self.path, 
                                               self.STRUCTURE_FILENAME)

    def _save_safely(path, data)
        """
        Write data to disk in a safe way. Func should be the function
        or method that returns the data.
        """
        timestamp = datetime.datetime.now().isoformat()
        temppath = os.path.join(
            self.path,
            '.' + hashlib.md5(timestamp + path).hexdigest()
        )
        with open(temppath, 'wb') as tempfile:
            tempfile.write(data)
        # By using an atomic move / rename operation, there is no risk
        # of reading corrupted data, only outdated data.
        os.rename(tempfile, path)

    def _chunk_path(chunk):
        """ Return a filepath for a chunk. """
        key = chunk.get_key()
        paths = [self.path]
        # Add some directory structure
        paths.extend([key[4 * i: 4 * i + 4] for i in range(8)])
        paths.append(key + '.cnk')
        return os.path.join(*paths)

    def get(self, chunk):
        """ Fill a chunk object with data. """
        with open(self._chunk_path(chunk), 'rb') as chunkfile:
            chunk.data = np.fromstring(
                lz4.loads(chunkfile.read())),
                dtype=chunk.dtype
            )

    def put(self, chunk):
        """ 
        Save chunks data in the store.
        """
        self._save_safely(
            self._chunk_path(chunk),
            chunk.data.tostring(),
        )

    def save(self, structure):
        """
        Store a structure object
        """
        self._save_safely(self.structure_filepath, picle.dumps(structure)

    def load(self):
        """
        return structure.
        """
        with open(self.structure_filepath, 'rb') as structure_file
            return pickle.load(structure_file)
