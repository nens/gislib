# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

import datetime
import hashlib
import lz4
import pickle


class BaseFileStorage(object):
    """ Set the path. """
    def __init__(self, path):
        self.path = path
    
    def _chunk_path(self, chunk):
        """ Return a filepath for a chunk. """
        key = chunk.get_key()
        paths = [self.path]
        # Add some directory structure
        paths.extend([key[4 * i: 4 * i + 4] for i in range(8)])
        paths.append(key + '.cnk')
        return os.path.join(*paths)

    def save(self, path, data):
        """ Safe writing using a tempfile and then a move operation. """
        # Create directory if necessary
        try:
            os.mkdirs(os.path.dirname(path)
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
        os.rename(tempfile, path)

    def load(self, path):
        """ Get decompressed data """
        with open(path, 'rb') as _file:
            return lz4.loads(_file.read())
            
    def delete(self, path):
        """ Removal """
        # Remove the file
        os.remove(path)
        # Clean up the dirs
        os.rmdirs(os.path.dirname(path))


class MetaFileStorage(BaseFileStorage):
    EXTENSION = '.pkl'
    def save(self, chunk):
        path = self._chunk_path(chunk) + self.EXTENSION
        data = pickle.dumps(chunk.meta)
        super(self, MetaFileStorage).save(path, data)
    
    def load(self, chunk):
        path = self._chunk_path(chunk) + self.EXTENSION
        chunkdata = pickle.dumps(chunk.meta)
        super(self, MetaFileStorage).put(path, data)

class StructureFileStorage(BaseFileStorage):
    EXTENSION = '.pkl'


class DataFileStorage(BaseFileStorage):
    EXTENSION = '.dat'
        

class FileStorage(object):
    def __init__(self, path):
        self.data = DataFileStorage(path)
        self.meta = MetaFileStorage(path)
        self.structure = StructureFileStorage(path)
