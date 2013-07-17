# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import gdal
import numpy as np

from gislib import projections
from gislib import rasters
from gislib.store import datasets
from gislib.store import domains


class GDALAdapter(object):
    """ 
    """
    def __init__(self, sourcepaths):
        self.sourcepaths = sourcepaths
        
    def get_data(self, gdal_dataset):
        """ Return a masked array from dataset. """
        return np.ma.masked_equal(
            gdal_dataset.ReadAsArray(),
            gdal_dataset.GetRasterBand(1).GetNoDataValue(),
            copy=False,
        ).reshape(-1,
                  gdal_dataset.RasterXSize,
                  gdal_dataset.RasterYSize).transpose(1, 2, 0)

    def get_axes(self, gdal_dataset):
        return tuple()

    def get_config(self, gdal_dataset):
        """ Get a config for this dataset. """
        # Determine space extent
        projection = projections.get_wkt(gdal_dataset.GetProjection())
        gdal_extent = rasters.DatasetGeometry.from_dataset(gdal_dataset).extent
        space_kwargs = dict(
            domain=domains.Space(projection=projection),
            extent=(gdal_extent[:2], gdal_extent[2:]),
            size=(gdal_dataset.RasterXSize, gdal_dataset.RasterYSize),
        )
        # Fix time extent for now
        calendar = 'minutes since 20130401'
        time_kwargs = dict(
            domain=domains.Time(calendar=calendar),
            extent=((0,), (gdal_dataset.RasterCount,)),
            size=(gdal_dataset.RasterCount,),
        )
        return datasets.Config(domains=[
            datasets.Domain(**space_kwargs),
            datasets.Domain(**time_kwargs),
        ])

    def get_dataset(self, sourcepath):
        """
        Return dataset object.
        """
        gdal_dataset = gdal.Open(sourcepath)
        return datasets.Dataset(
            config=self.get_config(gdal_dataset),
            axes=self.get_axes(gdal_dataset),
            data=self.get_data(gdal_dataset),
        )

    def get_datasets(self):
        """
        Return a generator of dataset objects.
        """
        for sourcepath in self.sourcepaths:
            yield self.get_dataset(sourcepath)
    
    def get_configs(self):
        """
        Return a generator of config objects and adapter arguments
        """
        for sourcepath in self.sourcepaths:
            gdal_dataset = gdal.Open(sourcepath)
            yield sourcepath, self.get_config(gdal_dataset)
