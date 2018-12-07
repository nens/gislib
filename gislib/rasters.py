# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import datetime
import json
import logging
import multiprocessing
import os

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
import numpy as np

from gislib import projections
from gislib import utils

gdal.UseExceptions()
osr.UseExceptions()


def array2dataset(array):
    """
    Return gdal dataset.

    Fastest way to get a gdal dataset from a numpy array,
    but keep a refernce to the array around, or a segfault will
    occur. Also, don't forget to call FlushCache() on the dataset after
    any operation that affects the array.
    """
    # Prepare dataset name pointing to array
    datapointer = array.ctypes.data
    bands, lines, pixels = array.shape
    datatypecode = gdal_array.NumericTypeCodeToGDALTypeCode(array.dtype.type)
    datatype = gdal.GetDataTypeName(datatypecode)
    bandoffset, lineoffset, pixeloffset = array.strides

    dataset_name_template = (
        'MEM:::'
        'DATAPOINTER={datapointer},'
        'PIXELS={pixels},'
        'LINES={lines},'
        'BANDS={bands},'
        'DATATYPE={datatype},'
        'PIXELOFFSET={pixeloffset},'
        'LINEOFFSET={lineoffset},'
        'BANDOFFSET={bandoffset}'
    )
    dataset_name = dataset_name_template.format(
        datapointer=datapointer,
        pixels=pixels,
        lines=lines,
        bands=bands,
        datatype=datatype,
        pixeloffset=pixeloffset,
        lineoffset=lineoffset,
        bandoffset=bandoffset,
    )
    # Acces the array memory as gdal dataset
    dataset = gdal.Open(dataset_name, gdal.GA_Update)
    return dataset


def dict2dataset(dictionary):
    """
    Like array2dataset, but set additional properties on the dataset.

    Dictionary must contain 'array' and may contain: 'geotransform',
    'projection', 'nodatavalue'
    """
    # dataset
    array = dictionary['array']
    dataset = array2dataset(array)
    # geotransform
    geotransform = dictionary.get('geotransform')
    if geotransform is not None:
        dataset.SetGeoTransform(geotransform)
    # projection
    projection = dictionary.get('projection')
    if projection is not None:
        dataset.SetProjection(projection)
    # nodatavalue
    nodatavalue = dictionary.get('nodatavalue')
    if nodatavalue is not None:
        for i in range(len(array)):
            dataset.GetRasterBand(i + 1).SetNoDataValue(nodatavalue)
    return dataset


def reproject(source, target, algorithm=gdal.GRA_NearestNeighbour):
    """ Reproject source to target. """
    gdal.ReprojectImage(
        source, target,
        projections.get_wkt(source.GetProjection()),
        projections.get_wkt(target.GetProjection()),
        algorithm,
        0.0,
        0.125,
    )


def get_dtype(dataset):
    """ Return the numpy dtype. """
    return np.dtype(gdal_array.flip_code(
        dataset.GetRasterBand(1).DataType,
    ))


def get_no_data_value(dataset):
    """ Return the no data value from the first band. """
    return dataset.GetRasterBand(1).GetNoDataValue()


def get_shape(dataset):
    """ Return the numpy shape. """
    return dataset.RasterCount, dataset.RasterYSize, dataset.RasterXSize


def get_geotransform(extent, width, height):
    """ Return geotransform tuple. """
    x1, y1, x2, y2 = extent
    return x1, (x2 - x1) / width, 0, y2, 0, (y1 - y2) / height


class Dataset(object):
    """ Wrapper aroung gdal dataset. """
    def __init__(self, dataset):
        self.dataset = dataset
        self.projection = dataset.GetProjection()
        self.raster_count = dataset.RasterCount
        self.raster_size = (dataset.RasterYSize, dataset.RasterXSize)
        self.geo_transform = dataset.GetGeoTransform()

        band = dataset.GetRasterBand(1)
        self.block_size = band.GetBlockSize()
        self.data_type = band.DataType
        self.no_data_value = band.GetNoDataValue()

    def get_offsets(self, block):
        """ Return offsets tuple. """
        u1 = block[0] * self.block_size[0]
        u2 = min(self.raster_size[0], (block[0] + 1) * self.block_size[0])
        v1 = block[1] * self.block_size[1]
        v2 = min(self.raster_size[1], (block[1] + 1) * self.block_size[1])
        return u1, u2, v1, v2

    def read_block(self, block):
        """ Read a block as a dataset. """
        # read the data
        u1, u2, v1, v2 = self.get_offsets(block)
        band_list = range(1, self.raster_count + 1)
        array = np.frombuffer(self.dataset.ReadRaster(
            u1, v1, u2 - u1, v2 - v1, band_list=band_list,
        ), dtype=gdal_array.flip_code(self.data_type))
        array.shape = self.raster_count, v2 - v1, u2 - u1

        # dataset properties
        p, a, b, q, c, d = self.geo_transform
        geo_transform = p + a * u1 + b * v1, a, b, q + c * u1 + d * v1, c, d
        dataset = dict2dataset(dict(
            array=array,
            geotransform=geo_transform,
            projection=self.projection,
            nodatavalue=self.no_data_value,
        ))
        return dict(dataset=dataset, array=array)

    def write_block(self, block, array):
        """ Write a dataset to a block. """
        u1, u2, v1, v2 = self.get_offsets(block)
        band_list = range(1, self.raster_count + 1)
        self.dataset.WriteRaster(
            u1, v1, u2 - u1, v2 - v1, array.tostring(), band_list=band_list,
        )
        # self.dataset.FlushCache() wasn't good enough. This method
        # closes and reopens the dataset to really flush any cache.
        self.dataset = gdal.Open(self.dataset.GetFileList()[0], gdal.GA_Update)

    def get_extent(self, projection=None):
        pass

    def get_pixel_size(self, projection=None):
        """ Get the size of a pixel, assuming square. """
        return self.geo_transform[1], -self.geo_transform[5]

    def get_pixel_geometry(indices=(0, 0), projection=None):
        """ Return a geometry corresponding to a raster pixel. """
        pass

    def get_outline_geometry(projection=None):
        """ Return a polygon corresponding to the outline of the raster. """
        pass


class SharedMemoryDataset(object):
    """
    Wrapper for a gdal dataset in a shared memory buffer.
    """
    def __init__(self, dataset):
        """ Initialize a dataset. """
        dtype = get_dtype(dataset)
        shape = get_shape(dataset)
        no_data_value = get_no_data_value(dataset)

        geotransform = dataset.GetGeoTransform()

        # Create underlying numpy array with data in shared buffer
        size = shape[0] * shape[1] * shape[2] * dtype.itemsize
        self.array = np.frombuffer(
            multiprocessing.RawArray('b', size), dtype,
        ).reshape(*shape)

        # Create a dataset from it
        self.dataset = dict2dataset(dict(array=self.array,
                                    geotransform=geotransform,
                                    nodatavalue=no_data_value,
                                    projection=dataset.GetProjection()))

        # Directly read the dataset into the shared memory array
        dataset.ReadAsArray(buf_obj=self.array)


class Geometry(object):
    def __init__(self, extent, size):
        self.extent = extent
        self.size = size

    def width(self):
        return self.size[0]

    def height(self):
        return self.size[1]

    def shape(self):
        return self.size[::-1]

    def delta(self):
        """ Return size tuple in extent units. """
        left, bottom, right, top = self.extent
        return right - left, top - bottom

    def cellsize(self):
        """ Return cellsize tuple. """
        return tuple(np.array(self.delta()) / np.array(self.size))

    def geotransform(self):
        """ Return geotransform tuple. """
        left, top = self.extent[0], self.extent[3]
        cellwidth, cellheight = self.cellsize()
        return left, cellwidth, 0, top, 0, -cellheight

    def gridpoints(self):
        """ Return array of shape with * height, 2. """
        x1, y1, x2, y2 = self.extent
        width, height = self.size
        x_step, y_step = self.cellsize()

        mgrid = np.mgrid[y2 - y_step / 2:y1 + y_step / 2:height * 1j,
                         x1 + x_step / 2:x2 - x_step / 2:width * 1j]

        return mgrid[::-1].transpose(1, 2, 0).reshape(-1, 2)

    def gridcoordinates(self):
        """ Return x, y arrays of length width, height. """
        x1, y1, x2, y2 = self.extent
        width, height = self.size
        x_step, y_step = self.cellsize()

        ogrid = np.ogrid[y2 - y_step / 2:y1 + y_step / 2:self.height * 1j,
                         x1 + x_step / 2:x2 - x_step / 2:self.width * 1j]
        return ogrid[1].reshape(-1, 1), ogrid[0].reshape(-1, 1)


class DatasetGeometry(Geometry):
    """ Add methods specific to pyramid building and transformations. """

    @classmethod
    def from_dataset(cls, dataset):
        x, dxx, dxy, y, dxy, dyy = dataset.GetGeoTransform()
        size = dataset.RasterXSize, dataset.RasterYSize
        extent = (x,
                  y + dyy * size[1],
                  x + dxx * size[0],
                  y)
        return cls(extent=extent, size=size)

    def to_dataset(self, datatype=1, bands=1, projection=None):
        """ Return in-memory gdal dataset. """
        driver = gdal.GetDriverByName(b'mem')
        dataset = driver.Create(b'',
                                self.size[0], self.size[1], bands, datatype)
        dataset.SetGeoTransform(self.geotransform())
        dataset.SetProjection(projections.get_wkt(projection))
        return dataset

    def transformed_cellsize(self, source_projection, target_projection):
        """ Return transformed cellsize. """
        left, bottom, right, top = utils.get_transformed_extent(
            self.extent, source_projection, target_projection,
        )
        return min((right - left) / self.size[0],
                   (top - bottom) / self.size[1])


class LockError(Exception):
    pass


class AbstractGeoContainer(object):
    """ Abstract class with locking mechanism. """

    def _lock(self):
        """ Create a lockfile. """
        # Create directory if it does not exist in a threadsafe way
        try:
            os.makedirs(os.path.dirname(self._lockpath))
        except:
            pass
        # Make a lockfile. Raise LockException if not possible.
        try:
            fd = os.open(self._lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            self._raise_locked_exception()

        # Write current date in the lockfile.
        with os.fdopen(fd, 'w') as lockfile:
            lockfile.write(str(datetime.datetime.now()))

    def _unlock(self):
        """ Remove a lockfile """
        os.remove(self._lockpath)

    def _raise_locked_exception(self):
        """ Raise locking specific OSError. """
        raise LockError('Object is locked.')

    def is_locked(self):
        """ Return if the container is locked for updating. """
        return os.path.exists(self._lockpath)

    def verify_not_locked(self):
        """ Return None or raise exception. """
        if self.is_locked():
            self._raise_locked_exception()


class Pyramid(AbstractGeoContainer):
    """
    Add and get geodata to and from a paramid of datafiles.
    """
    _LOCKFILE = 'pyramid.lock'
    CONFIG_FILE = 'pyramid.json'
    CONFIG_ATTRIBUTES = ['algorithm',
                         'cellsize',
                         'compression',
                         'datatype',
                         'nodatavalue',
                         'projection',
                         'tilesize']

    def __init__(self, path, projection=projections.GOOGLE,
                 algorithm=0, compression='NONE',
                 tilesize=(1024, 1024), cellsize=None):
        """
        Fully initialize existing pyramid, or defer
        initialization to first add for new pyramid.
        """
        self.config_path = os.path.join(path, self.CONFIG_FILE)
        self._lockpath = os.path.join(path, self._LOCKFILE)
        self.path = path

        # If another process creates the pyramid, it first locks the
        # pyramid and then creates the config file. Checking for both
        # files in reverse order makes init threadsafe.
        if os.path.exists(self.config_path):
            self.verify_not_locked()
            # Config from path
            with open(self.config_path) as config_file:
                config = json.load(config_file)
                for attr in self.CONFIG_ATTRIBUTES:
                    setattr(self, attr, config[attr])

            # Determine toplevel
            for basedir, dirnames, filenames in os.walk(self.path):
                self.toplevel = max(map(int, dirnames))
                break

            # Determine extent
            self.extent = self._extent()

        else:
            # Config from kwargs
            self.algorithm = algorithm
            self.cellsize = cellsize
            self.compression = compression
            self.projection = projection
            self.tilesize = tilesize
            # vvv Determined when first dataset gets added
            self.datatype = None
            self.nodatavalue = None
            self.toplevel = None

    def has_data(self):
        """ Return boolean. """
        return self.toplevel is not None

    def _extent(self):
        """ Return extent of the toplevel tile. """
        topleveldir = os.path.join(self.path, str(self.toplevel))
        toplevelpath = os.path.join(topleveldir,
                                    os.listdir(topleveldir)[0])
        dataset = gdal.Open(str(toplevelpath))
        return DatasetGeometry.from_dataset(dataset).extent

    def _config_from_dataset(self, dataset):
        """ Determine missing attributes from dataset and write config. """
        # Determine nodatavalue and datatype from dataset
        band = dataset.GetRasterBand(1)
        self.nodatavalue = band.GetNoDataValue()
        self.datatype = band.DataType

        if self.cellsize is None:
            self.cellsize = self._cellsize(dataset)

        config = {k: getattr(self, k) for k in self.CONFIG_ATTRIBUTES}
        with open(self.config_path, 'w') as config_file:
            json.dump(config, config_file)

    def _cellsize(self, dataset):
        """ Return approximate cellsize of dataset in pyramid coordinates. """
        dataset_geometry = DatasetGeometry.from_dataset(dataset)
        return dataset_geometry.transformed_cellsize(
            source_projection=dataset.GetProjection(),
            target_projection=self.projection,
        )

    def _dataset(self, level, tile, mode='r'):
        """
        Return a gdal dataset.

        If the file corresponding to level and tile does not exist:
            In (r)ead mode, return mem dataset with nodata
            In (w)rite mode, create and return tif dataset with nodata
        """
        path = os.path.join(self.path, str(level), '{}_{}.tif'.format(*tile))
        if os.path.exists(path):
            # Open existing file with correct gdal access mode
            if mode == 'w':
                access = gdal.GA_Update
                logging.debug('Update {}'.format(path))
            else:
                access = gdal.GA_ReadOnly
            return gdal.Open(str(path), access)

        create_args = [str(path),
                       self.tilesize[0],
                       self.tilesize[1],
                       1,
                       self.datatype,
                       ['TILED=YES',
                        'COMPRESS={}'.format(self.compression)]]

        if mode == 'w':
            # Use gtiff driver
            driver = gdal.GetDriverByName(b'gtiff')
            logging.debug('Create {}'.format(path))

            # Create directory if necessary
            try:
                os.makedirs(os.path.dirname(path))
            except OSError:
                pass  # It existed.
        else:  # mode == 'r'
            # Use mem driver
            driver = gdal.GetDriverByName(b'mem')
            create_args.pop()  # No compression for mem driver

        # Actual create
        dataset = driver.Create(*create_args)
        dataset.SetProjection(
            projections.get_spatial_reference(self.projection).ExportToWkt(),
        )
        dataset.SetGeoTransform(
            self._geometry(level=level, tile=tile).geotransform(),
        )
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(self.nodatavalue)
        band.Fill(self.nodatavalue)
        return dataset

    def _geometry(self, level=0, tile=(0, 0)):
        """ Return geometry for a tile at level. """
        width, height = self.tilesize
        cellsize = self.cellsize * np.power(2, level)
        x1 = cellsize * width * tile[0]
        x2 = cellsize * width * (tile[0] + 1)
        y1 = cellsize * height * tile[1]
        y2 = cellsize * height * (tile[1] + 1)
        extent = x1, y1, x2, y2
        return Geometry(extent=extent, size=self.tilesize)

    def _tiles(self, dataset, level, limit=False):
        """
        Return tile indices generator for the extent of dataset.

        If limit is true, do not return tiles that are fully outside
        the pyramids extent. This prevents creating lots and lots of
        empty datasets when datasets with extents much larger than our
        own are supplied to warpinto().
        """
        # Determine extent in pyramid projection from dataset
        dataset_geometry = DatasetGeometry.from_dataset(dataset)
        transformed_extent = utils.get_transformed_extent(
            extent=dataset_geometry.extent,
            source_projection=dataset.GetProjection(),
            target_projection=self.projection,
        )
        original_extent = np.array(dataset_geometry.extent)
        difference = (transformed_extent - original_extent).max()

        # If almost the same, take original because of rounding errors
        if difference:
            dataset_extent = transformed_extent
        else:
            dataset_extent = original_extent

        # Return extent, or intersected extent based on limit setting
        if limit:
            extent = np.array(utils.get_extent_intersection(self.extent,
                                                            dataset_extent))
        else:
            extent = np.array(dataset_extent)

        # Determine limits for the indices and return a generator
        delta = self._geometry(level).delta()
        x1, y1 = extent[:2] // delta
        # If extent upper bounds are on tile edge,
        # don't add neighbouring tiles.
        x2, y2 = np.where(
            extent[2:] // delta == extent[2:] / delta,
            extent[2:] // delta - 1,
            extent[2:] // delta,
        )
        # Return generator
        return ((x, y)
                for y in np.arange(y1, y2 + 1, dtype=np.int64)
                for x in np.arange(x1, x2 + 1, dtype=np.int64))

    def _add(self, dataset, level=0):
        """
        Add dataset to the pyramid. After adding this one, it's extent is
        cascaded up to the top level of the pyramid, so that the pyramid
        stays consistent.
        """
        # Adding now, so toplevel cannot be lower than this level
        self.toplevel = max(self.toplevel, level)

        # Loop tiles in this pyramid and reproject dataset into it.
        for tile in self._tiles(dataset, level):
            tile_dataset = self._dataset(level=level, tile=tile, mode='w')
            reproject(source=dataset,
                      target=tile_dataset,
                      algorithm=self.algorithm)
            tile_dataset = None  # Close the file.

        # Rebuild levels above this for involved tiles
        if level < self.toplevel:
            for tile in self._tiles(dataset, level):
                tile_dataset = self._dataset(level=level, tile=tile)
                self._add(tile_dataset, level=level + 1)
                tile_dataset = None  # Close the file

    def _extend(self):
        """
        If there are multiple tiles at the toplevel, add them to
        the next level.
        """
        topleveldir = os.path.join(self.path, str(self.toplevel))
        filenames = os.listdir(topleveldir)
        if len(filenames) > 1:
            self.toplevel += 1
            for filename in filenames:
                dataset = gdal.Open(str(os.path.join(topleveldir, filename)))
                self._add(dataset=dataset, level=self.toplevel)
                dataset = None  # Close the file
            # Call this method again, until the top is reached.
            self._extend()

        # When we're done, its safe to get extent
        self.extent = self._extent()

    def _level(self, dataset):
        """ Return level for dataset. """
        cellsize = self._cellsize(dataset)
        level = int(np.log2(cellsize / self.cellsize))
        if level < 0:
            return 0
        if level > self.toplevel:
            return self.toplevel
        return level

    def add(self, dataset):
        """ Non-recursive version of add """
        self._lock()
        try:
            # Config if this is the first
            if self.toplevel is None:
                self._config_from_dataset(dataset)

            # Add dataset
            self._add(dataset)
            self._extend()
        except KeyboardInterrupt:
            pass
        finally:
            self._unlock()

    def warpinto(self, dataset):
        """
        Warp from the pyramid into given dataset. Level of the pyramid that
        is used is taken from the resolution of the dataset.
        """
        level = self._level(dataset)
        # Set apply pyramid nodatavalue
        for i in range(dataset.RasterCount):
            band = dataset.GetRasterBand(i + 1)
            band.SetNoDataValue(self.nodatavalue)
            band.Fill(self.nodatavalue)
        # Loop tiles in this pyramid and reproject dataset into it.
        for tile in self._tiles(dataset, level, limit=True):
            tile_dataset = self._dataset(level=level, tile=tile)
            reproject(source=tile_dataset,
                      target=dataset,
                      algorithm=self.algorithm)
            tile_dataset = None  # Close the file
        return dataset


class Monolith(AbstractGeoContainer):
    """
    Simple dataset container that shares the methods add and warpinto of
    the pyramid. However, multiple adds just overwrite the old dataset,
    it only holds one dataset at a time.
    """
    TIF_FILE = 'monolith.tif'
    _LOCKFILE = 'monolith.lock'

    def __init__(self, path, memory=True,
                 algorithm=0, compression='NONE'):
        """
        Add and init use filesystem.

        If memory, the data is kept in-memory for fast warpinto.
        """
        self.algorithm = algorithm
        self.compression = compression
        self.path = path
        self.tifpath = os.path.join(path, self.TIF_FILE)
        self._lockpath = os.path.join(path, self._LOCKFILE)
        self.memory = memory

        if not os.path.exists(self.tifpath):
            # Do not set dataset attribute, monolith is empty.
            return

        self.verify_not_locked()
        tif_dataset = gdal.Open(str(self.tifpath))

        if self.memory:
            driver = gdal.GetDriverByName(b'mem')
            self.dataset = driver.CreateCopy(b'', tif_dataset)
        else:
            self.dataset = tif_dataset
        self._set_attributes_from_dataset()

    def has_data(self):
        """ Return boolean. """
        return hasattr(self, 'dataset')

    def _set_attributes_from_dataset(self):
        """ For convenience, store some dataset attributes on self. """
        band = self.dataset.GetRasterBand(1)
        self.nodatavalue = band.GetNoDataValue()
        self.datatype = band.DataType

    def add(self, dataset):
        """
        Add a copy of dataset as tif.

        If self.memory, create an in-memory copy as well.
        """
        self._lock()

        # Create a file copy of dataset
        tif_driver = gdal.GetDriverByName(b'gtiff')
        create_args = [str(self.tifpath),
                       dataset,
                       1,  # Strict, just default value
                       ['TILED=YES',
                        'COMPRESS={}'.format(self.compression)]]
        tif_dataset = tif_driver.CreateCopy(*create_args)

        # Apply default projection if there is none.
        tif_dataset.SetProjection(
            projections.get_wkt(dataset.GetProjection()),
        )

        # Close and reopen dataset to force flush.
        tif_dataset = None
        tif_dataset = gdal.Open(str(self.tifpath))

        # Set dataset attribute based on memory attribute
        if self.memory:
            mem_driver = gdal.GetDriverByName(b'mem')
            mem_dataset = mem_driver.CreateCopy(b'', tif_dataset)
            self.dataset = mem_dataset
        else:
            self.dataset = tif_dataset

        self._set_attributes_from_dataset()
        self._unlock()

    def warpinto(self, dataset):
        """ Warp our dataset into argument dataset. """
        reproject(source=self.dataset,
                  target=dataset,
                  algorithm=self.algorithm)


class Container(object):
    """ Generic container for gdal datasets with builtin warp. """
    def __init__(self, path=None, dataset=None, algorithm=0):
        """
        Use a path to a dataset, or use dataset keyword argument to
        specify an opened dataset.
        """
        if path is None:
            self.dataset = dataset
        else:
            self.dataset = gdal.Open(str(path))
        self.algorithm = algorithm
        self._set_attributes_from_dataset()

    def _set_attributes_from_dataset(self):
        """ For convenience, store some dataset attributes on self. """
        band = self.dataset.GetRasterBand(1)
        self.nodatavalue = band.GetNoDataValue()
        self.datatype = band.DataType

    def warpinto(self, dataset):
        """ Warp our dataset into argument dataset. """
        reproject(source=self.dataset,
                  target=dataset,
                  algorithm=self.algorithm)
