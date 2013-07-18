# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections

import gdal
import numpy as np

from gislib import projections
from gislib import rasters
from gislib.store import datasets
from gislib.store import domains


class Cache(collections.OrderedDict):
    def __setitem__(self, *args, **kwargs):
        if len(self) >= 10:
            del self[(self.iterkeys().next())]
        return super(Cache, self).__setitem__(*args, **kwargs)

cache = Cache()


class GDALAdapter(object):
    """ 
    """
    def __init__(self, sourcepaths):
        self.sourcepaths = sourcepaths
        
    def get_data(self, gdal_dataset, config):
        """ Return a masked array from dataset. """
        data = gdal_dataset.ReadAsArray()
        return data.transpose().reshape(config.shape)

    def get_axes(self, gdal_dataset):
        return tuple()

    def get_config(self, gdal_dataset):
        """ Get a config for this dataset. """
        fill = gdal_dataset.GetRasterBand(1).GetNoDataValue()
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
        return datasets.Config(fill=fill, domains=[
            datasets.Domain(**space_kwargs),
            datasets.Domain(**time_kwargs),
        ])

    def get_dataset(self, sourcepath):
        """
        Return dataset object.
        """
        if sourcepath not in cache:
            print('create')
            gdal_dataset = gdal.Open(sourcepath)
            config = self.get_config(gdal_dataset)
            cache[sourcepath] = datasets.Dataset(
                config=config,
                axes=self.get_axes(gdal_dataset),
                data=self.get_data(gdal_dataset=gdal_dataset, config=config),
            )
        else:
            print('return from cache')
        return cache[sourcepath]

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
