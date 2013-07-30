# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections

import gdal
import h5py

from gislib import projections
from gislib import rasters
from gislib.store import datasets
from gislib.store import kinds


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


class RadarAdapter(object):
    def __init__(self, sourcepaths):
        self.sourcepaths = sourcepaths

    def _get_space_extent(self, h5):
        east = h5['east'][:]
        east_step = east[1] - east[0]
        north = h5['north'][:]
        north_step = north[0] - north[1]
        lower = east[0] - east_step / 2, north[-1] - north_step / 2
        upper = east[-1] + east_step / 2, north[0] + north_step / 2
        return lower, upper

    def _get_times(self, h5):
        chunks = h5['precipitation'].chunks
        size = h5['time'].size
        for i in range(0, size, chunks[2]):
            yield i, min(i + chunks[2], size)

    def __iter__(self):
        with h5py.File(self.sourcepaths[0]) as h5:
            time = h5['time']
            precipitation = h5['precipitation']
            space_extent = self._get_space_extent(h5)
            space_size = precipitation.shape[-2::-1]
            unit = time.attrs['units']
            proj = 28992
            fill = -9999
            #dtype = precipitation.dtype
            space_kind = kinds.Space(proj=proj)
            time_kind = kinds.Time(unit=unit)
            space_domain = datasets.Domain(kind=space_kind,
                                           size=space_size,
                                           extent=space_extent)
            for i1, i2 in self._get_times(h5):
                time_extent = tuple(map(lambda x: (x,),time[[i1, i2 - 1]]))
                time_lower = time_extent[0][0]
                time_span = time_extent[1][0] - time_lower
                time_domain = datasets.Domain(kind=time_kind,
                                              size=(i2 - i1,),
                                              extent=time_extent)
                time_axes = (time[i1:i2] - time_lower) / time_span
                dataset_kwargs = dict(
                    domains=(space_domain, time_domain),
                    axes=(None, time_axes),
                    data=precipitation[:, :, i1:i2].transpose()[::-1],
                    fill=fill,
                )
                yield datasets.Dataset(**dataset_kwargs)
