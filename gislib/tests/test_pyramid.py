from unittest import TestCase

from osgeo import gdal_array
import numpy as np
import mock

from gislib import pyramids


def manager_factory(levels=[]):
    with mock.patch('os.listdir', return_value=levels):
        return pyramids.Manager('')


class TestPointFromDataset(TestCase):
    def setUp(self):
        a = np.array([
            [1,   2,  3,  4],
            [5,   6,  7,  8],
            [9,  10, 11, 12],
            [13, 14, 15, 16]])
        self.dataset = gdal_array.OpenArray(a)
        self.dataset.SetGeoTransform(
            [50, 2, 0, 75, 0, -2])

    def test_lefttop(self):
        self.assertEquals(
            pyramids.point_from_dataset(self.dataset, (51, 74)),
            np.array([1]))

    def test_outsiderightcorner_raises(self):
        self.assertRaises(
            RuntimeError,
            lambda: pyramids.point_from_dataset(self.dataset, (58, 67)))

    def test_on_edge(self):
        self.assertEquals(
            pyramids.point_from_dataset(self.dataset, (54, 71)),
            np.array([11]))


class TestGetTiles(TestCase):
    def test_example(self):
        self.assertEquals(
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            list(pyramids.get_tiles((2, 2), (1, 1, 3, 3))))


class GetTile(TestCase):
    def test_point_inside_spacing_returns_00(self):
        self.assertEquals(
            pyramids.get_tile((2, 2), (1, 1)),
            (0, 0))

    def test_point_outside(self):
        self.assertEquals(
            pyramids.get_tile((2, 2), (3, 3)),
            (1, 1))


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
