# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import gdal
from osgeo import gdalconst
from osgeo import ogr

from matplotlib.backends import backend_agg
from matplotlib import figure
from matplotlib import colors
from matplotlib import cm
from matplotlib import patches

from PIL import Image

import numpy as np


def ds2ma(dataset, bandnumber=1):
    """
    Return np masked array from band in gdal dataset.
    """
    band = dataset.GetRasterBand(bandnumber)
    fill_value = band.GetNoDataValue()
    array = band.ReadAsArray()
    mask = np.equal(array, fill_value)
    masked_array = np.ma.array(array, mask=mask, fill_value=fill_value)
    return masked_array


def h5ds2ma(dataset):
    """
    Return np masked array dataset.

    Expects an attribute fillvalue set on dataset.
    """
    fill_value = dataset.attrs['fill_value']
    array = dataset[:]
    mask = np.equal(array, fill_value)
    masked_array = np.ma.array(array, mask=mask, fill_value=fill_value)
    return masked_array


def default_normalize(array):
    normalize = colors.Normalize()
    return normalize(array)


class BaseGrid(object):
    """
    A grid is defined by its size, extent and projection.

    Extent is (left, right, top, bottom); cellsize is (width, height);
    projection is a wkt string.
    """
    def __init__(self, dataset=None, extent=None, size=None, projection=None):
        """
        Either use a dataset, or an extent, a cellsize and optionally
        a projection
        """
        if dataset and not (extent or size or projection):
            self._init_from_dataset(dataset)
        elif (size is not None and extent is not None) and not dataset:
            for s in size:
                if not isinstance(s, int):
                    raise TypeError('Size elements must be of type int.')
            self.size = size
            self.extent = extent
            self.projection = projection
        else:
            raise NotImplementedError('Incompatible arguments')

    def _init_from_dataset(self, dataset):
        self.size = dataset.RasterXSize, dataset.RasterYSize
        self.projection = dataset.GetProjection()

        x, a, b, y, c, d = dataset.GetGeoTransform()
        self.extent = x, x + a * self.size[0], y, y + d * self.size[1]

    def get_geotransform(self):
        left, right, top, bottom = self.extent
        cellwidth = (right - left) / self.size[0]
        cellheight = (top - bottom) / self.size[1]
        return left, cellwidth, 0, top, 0, -cellheight

    def get_center(self):
        left, right, top, bottom = self.extent
        return (right - left) / 2, (top - bottom) / 2

    def get_cellsize(self):
        left, right, top, bottom = self.extent
        cellwidth = (right - left) / self.size[0]
        cellheight = (top - bottom) / self.size[1]
        return cellwidth, cellheight

    def get_shape(self):
        return self.size[::-1]

    def get_grid(self):
        """
        Return x and y coordinates of cell centers.
        """
        cellwidth, cellheight = self.get_cellsize()
        left, right, top, bottom = self.extent

        xcount, ycount = self.size
        xmin = left + cellwidth / 2
        xmax = right - cellwidth / 2
        ymin = bottom + cellheight / 2
        ymax = top - cellheight / 2

        y, x = np.mgrid[
            ymax:ymin:ycount * 1j, xmin:xmax:xcount * 1j]
        return x, y

    def create_dataset(self, bands=1,
                       nodatavalue=-9999, datatype=gdalconst.GDT_Float64):
        """
        Return empty in-memory dataset.

        It has our size, extent and projection.
        """
        dataset = gdal.GetDriverByName(b'MEM').Create(
            b'', self.size[0], self.size[1], bands, datatype,
        )
        dataset.SetGeoTransform(self.get_geotransform())
        dataset.SetProjection(self.projection)

        rasterbands = [dataset.GetRasterBand(i + 1) for i in range(bands)]
        for band in rasterbands:
            band.SetNoDataValue(nodatavalue)
            band.Fill(nodatavalue)

        return dataset

    def create_imagelayer(self, image):
        pass

    def create_vectorlayer(self):
        """ Create and return VectorLayer. """
        return VectorLayer(self)


class AbstractLayer(BaseGrid):
    """ Add imaging methods """

    def _rgba():
        raise NotImplementedError

    def _save_img(self, filepath):
        self.image().save(filepath)

    def _save_tif(self, filepath, rgba=True):
        dataset = self._rgba_dataset() if rgba else self._single_band_dataset()
        gdal.GetDriverByName(b'GTiff').CreateCopy(
            str(filepath), dataset, 0, ['COMPRESS=DEFLATE'],
        )

    def _save_asc(self, filepath):
        """ Save as asc file. """
        dataset = self._single_band_dataset()
        gdal.GetDriverByName(b'AAIGrid').CreateCopy(filepath, dataset)

    def _rgba_dataset(self):
        dataset = self.create_dataset(bands=4, datatype=gdalconst.GDT_Byte)
        bands = [dataset.GetRasterBand(i + 1) for i in range(4)]
        data = self._rgba().transpose(2, 0, 1)
        for band, array in zip(bands, data):
            band.WriteArray(array)
        return dataset

    def _single_band_dataset(self):
        dataset = self.create_dataset()
        band = dataset.GetRasterBand(1)
        band.WriteArray(self.ma.filled())
        band.SetNoDataValue(self.ma.fill_value)
        return dataset

    def _checker_image(self):
        pattern = (np.indices(
            self.get_shape(),
        ) // 8).sum(0) % 2
        return  Image.fromarray(cm.gray_r(pattern / 2., bytes=True))

    def show(self):
        """
        Show after adding checker pattern for transparent pixels.
        """
        image = self.image()
        checker = self._checker_image()
        checker.paste(image, None, image)
        checker.show()

    def image(self):
        return Image.fromarray(self._rgba())

    def save(self, filepath, **kwargs):
        """
        Save as image file.
        """
        if filepath.endswith('.tif') or filepath.endswith('.tiff'):
            self._save_tif(filepath, **kwargs)
        elif filepath.endswith('.asc'):
            self._save_asc(filepath)
        else:
            self._save_img(filepath)


class RasterLayer(AbstractLayer):
    """
    Layer containing grid data.
    """
    def __init__(self, dataset=None, band=1, colormap=None, normalize=None,
                 array=None, extent=None, projection=None):
        if dataset and array is None and extent is None and projection is None:
            rasterband = dataset.GetRasterBand(band)
            self._init_from_dataset(dataset=dataset)
            self._ma_from_rasterband(rasterband=rasterband)
        elif array is not None and dataset is None:
            self.size = array.shape[::-1]
            if extent is None:
                self.extent = [0, self.size[0], 0, self.size[1]]
            else:
                self.extent = extent
            self.projection = projection
            self._ma_from_array(array=array)
        else:
            raise NotImplementedError('Incompatible arguments')

        self.normalize = normalize or default_normalize
        self.colormap = colormap or cm.gray

    def _ma_from_rasterband(self, rasterband):
        """
        Store masked array and gridproperties
        """
        fill_value = rasterband.GetNoDataValue()
        array = rasterband.ReadAsArray()
        mask = np.equal(array, fill_value)
        self.ma = np.ma.array(array, mask=mask, fill_value=fill_value)

    def _ma_from_array(self, array):
            self.ma = np.ma.array(
                array,
                mask=array.mask if hasattr(array, 'mask') else False,
            )

    def _rgba(self):
        return self.colormap(self.normalize(self.ma), bytes=True)



class VectorLayer(AbstractLayer):
    def __init__(self, basegrid):
        """
        """
        self.projection = basegrid.projection
        self.extent = basegrid.extent
        self.size = basegrid.size
        self._add_axes()

    def _add_axes(self):
        """
        Add matplotlib axes with coordinates setup according to geo.
        """
        dpi = 72
        figsize = tuple(c / dpi for c in self.size)
        fig = figure.Figure(figsize, dpi, facecolor='g')
        fig.patch.set_alpha(0)
        backend_agg.FigureCanvasAgg(fig)

        rect, axis = self._mpl_config()
        axes = fig.add_axes(rect, axisbg='y')
        axes.axis(axis)
        axes.autoscale(False)

        axes.patch.set_alpha(0)
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)

        self.axes = axes

    def _mpl_config(self):
        """
        Return rect, axis.

        To get the matplotlib axes to match exactly the geotransform
        coordinates, an appropriate combination of the axes rect and
        the axis limits is required.

        Moreover, a factor is applied to make the axes a little larger
        than the figure, because otherwise some edge artifacts may
        be visible.
        """
        factor = 0.1
        rect = (-factor, -factor, 1 + 2 * factor, 1 + 2 * factor)

        left, right, top, bottom = self.extent
        width = right - left
        height = top - bottom
        cellwidth, cellheight = self.get_cellsize()

        # For some reason, 2 pixels have to be added to
        axis = (
            left - width * factor + cellwidth * 0,
            right + width * factor + cellwidth * 1,
            bottom - height * factor + cellheight * 0,
            top + height * factor + cellheight * 1,
        )

        return rect, axis

    def _rgba(self):
        canvas = self.axes.get_figure().canvas
        buf, shape = canvas.print_to_buffer()
        rgba = np.fromstring(buf, dtype=np.uint8).reshape(
            *(self.get_shape() + tuple([4]))
        )
        return rgba

    def add_image(self, image_path):
        """ Add a raster image, assuming extent matches ours. """
        image = Image.open(image_path)
        self.axes.imshow(image, extent=self.extent)

    def add_line(self, shapepath, *plotargs, **plotkwargs):
        """ Plot shape as matplotlib line """
        axes = self.axes
        dataset = ogr.Open(str(shapepath))
        for layer in dataset:
            for feature in layer:
                x, y = np.array(feature.geometry().GetPoints()).transpose()
                self.axes.plot(x, y, *plotargs, **plotkwargs)

    def add_patch(self, shapepath, *plotargs, **plotkwargs):
        """ Plot shape as matplotlib line """
        axes = self.axes
        dataset = ogr.Open(shapepath)
        for layer in dataset:
            for feature in layer:
                xy = np.array(feature.geometry().GetBoundary().GetPoints())
                self.axes.add_patch(
                    patches.Polygon(xy, *plotargs, **plotkwargs)
                )

    def add_multipolygon(self, shapepath, *plotargs, **plotkwargs):
        """ Plot shape as matplotlib line """
        axes = self.axes
        dataset = ogr.Open(shapepath)
        for layer in dataset:
            for feature in layer:
                count = feature.geometry().GetGeometryCount()
                polygons = [feature.geometry().GetGeometryRef(i)
                            for i in range(count)]
                for polygon in polygons:
                    xy = np.array(polygon.GetBoundary().GetPoints())
                    self.axes.add_patch(
                        patches.Polygon(xy, *plotargs, **plotkwargs)
                    )
