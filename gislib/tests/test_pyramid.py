from unittest import TestCase

import numpy as np
import mock

from gislib import pyramids


def manager_factory(levels=[]):
    with mock.patch('os.listdir', return_value=levels):
        return pyramids.Manager('')


class FakeRasterBand(object):
    DataType = np.int8

    def GetBlockSize(self):
        return 256

    def GetNoDataValue(self):
        return -999


class FakeDataSet(object):
    RasterCount = 1
    RasterYSize = 1024
    RasterXSize = 2048

    def GetRasterBand(self, i):
        return FakeRasterBand()

    def GetProjection(self):
        return "Huh?"


class TestManager(TestCase):
    def test_empty_manager_factory_has_no_levels(self):
        manager = manager_factory()
        self.assertEquals(manager.levels, [])

    def test_manager_sets_config_from_first_dataset(self):
        with mock.patch(
            'gislib.pyramids.Manager.get_datasets',
            return_value=iter([FakeDataSet()])) as mocked_get_datasets:
            manager = manager_factory(levels=["14"])

            mocked_get_datasets.assert_called_with(-1)

            self.assertEquals(manager.block_size, 256)
            self.assertEquals(manager.data_type, np.int8)
            self.assertEquals(manager.no_data_value, -999)
            self.assertEquals(manager.projection, "Huh?")
            self.assertEquals(manager.raster_count, 1)
            self.assertEquals(manager.raster_size, (1024, 2048))
