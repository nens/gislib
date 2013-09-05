# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import hashlib
import shutil
import tempfile
import unittest

import numpy as np

from gislib.store import core
from gislib.store import storages
from gislib.store import stores


class TestStorage(unittest.TestCase):
    """ Integration tests. """
    def setUp(self):
        # Create store
        self.tempdir = tempfile.mkdtemp()
        storage = storages.FileStorage(self.tempdir)
        metric = core.FrameMetric(scales=[])
        frame = core.Frame(metric=metric,
                           dtype='f4',
                           nodatavalue=np.finfo('f4').min)
        self.store = stores.Store(storage=storage, frame=frame)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_storage(self):
        """ Test key value storage. """
        key = hashlib.md5().hexdigest()

        # Splitting store
        value = 'testvalue'
        self.store.databox[key] = value
        self.assertEqual(self.store.databox[key], value)

        # Not splitting store
        value += 'two'
        self.store.config[key] = value
        self.assertEqual(self.store.config[key], value)
