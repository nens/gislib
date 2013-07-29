# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from scipy import ndimage
from osgeo import gdal

import numpy as np

from gislib import rasters
from gislib import projections
from gislib.store import kinds


class DoesNotFitError(Exception):
    """
    Raised by reproject functions if non-equidistant time domains do
    not fit in the extents of the dataset.
    """
    def __init__(self, dimension):
        self.dimension = dimension


def _to_gdal(dataset, i):
    """ Return in-memory gdal dataset. """
    domain = dataset.domains[i]
    size = domain.size

    # Create dataset
    a1 = tuple(j for d in dataset.domains[:i] for j in d.size)
    a2 = tuple(j for d in dataset.domains[i + 1:] for j in d.size)

    s1 = reduce(lambda x, y: x * y, a1, 1)
    s2 = reduce(lambda x, y: x * y, a2, 1)
    ysize, xsize = size
    bands = s1 * s2

    l1, l2 = len(a1), len(a2)
    gdal_dataset = gmdrv.Create(
        '',
        xsize,
        ysize,
        bands,
        gtype[dataset.data.dtype],
    )

    # Transpose args
    p1, p2, p3 = np.cumsum([l1, 2, l2])
    transpose = range(p1) + range(p2, p3) + range(p1, p2)

    # Inverse transpose and reshape args
    p1, p2, p3 = np.cumsum([l1, l2, 2])
    itranspose = range(p1) + range(p2, p3) + range(p1, p2)
    ireshape = a1 + a2 + size

    # Projection, geotransform, fill and data
    extent = tuple(e for t in domain.extent for e in t)
    geometry = rasters.Geometry(extent=extent, size=size)
    gdal_dataset.SetProjection(str(domain.kind.proj))
    gdal_dataset.SetGeoTransform(geometry.geotransform())
    shape = (-1,) + size
    data = dataset.data.transpose(*transpose).reshape(shape)
    #gdal_dataset.WriteRaster(0,0,xsize, ysize, data.tostring())
    for j in range(gdal_dataset.RasterCount):
        band = gdal_dataset.GetRasterBand(j + 1)
        band.SetNoDataValue(float(dataset.fill))
        band.WriteArray(data[j])

    return dict(data=gdal_dataset, transpose=itranspose, shape=ireshape)
   

def _transform_space(source, target, i):
    exit()
    
    
    
def _transform_time(source, target, i):
    pass


def reproject(source, target):
    """ Put relevant data from source into target. """
    # Create a view that selects relevant indices from time (and other ned)
    # Create a subview according to spatial domain
    # Raise if necessary
    # Create datasets from them and reproject.




        

def _transform_generic(source, target):
    """ Just affine transformation of equidistant grids. """
    # Source, target and intersection bounds
    l1, u1 = np.array(source.config.span).transpose()
    l2, u2 = np.array(target.config.span).transpose()
    l3, u3 = np.maximum(l1, l2), np.minimum(u1, u2)

    # Indices for intersection for both source and target
    s1, s2 = np.array(source.config.shape), np.array(target.config.shape)
    p1 = np.uint64(np.round((l3 - l1) / (u1 - l1) * s1))
    q1 = np.uint64(np.round((u3 - l1) / (u1 - l1) * s1))
    sourceview = source.data[tuple(map(slice, p1, q1))]
    p2 = np.uint64(np.round((l3 - l2) / (u2 - l2) * s2))
    q2 = np.uint64(np.round((u3 - l2) / (u2 - l2) * s2))
    targetview = target.data[tuple(map(slice, p2, q2))]

    # Bounds for views
    l4 = l1 + p1 / s1 * (u1 - l1)
    u4 = l1 + q1 / s1 * (u1 - l1)
    l5 = l2 + p2 / s2 * (u2 - l2)
    u5 = l2 + q2 / s2 * (u2 - l2)
    s4, s5 = np.array(sourceview.shape), np.array(targetview.shape)

    # Determine transform
    diagonal = (u5 - l5) / (u4 - l4) / (s5 / s4)
    if np.equal(diagonal, 1).all():
        targetview[:] = sourceview
    else:
        ndimage.affine_transform(sourceview, diagonal, mode='nearest',
                                 output_shape=targetview.shape,
                                 output=targetview, order=0)


class Domain(object):
    def __init__(self, kind, size, extent):
        self.kind = kind
        self.size = size
        self.extent = extent

    def __str__(self):
        return '<{cls}:{kind}:{size}:{extent}>'.format(
            cls=self.__class__.__name__, **self.__dict__
        )

    def __repr__(self):
        return self.__str__()

    def transform(self, kind):
        """ Return a new transformed domain. """
        kwargs = self.kind.transform(kind=kind,
                                     size=self.size,
                                     extent=self.extent)
        return Domain(kind=kind, **kwargs)


class Dataset(object):
    def __init__(self, domains, axes, data, fill):
        """ Dataset. """
        self.domains = domains
        self.fill = fill
        self.axes = axes  # A tuple of numpy arrays for the ned domains
        self.data = data  # The numpy array

    def __str__(self):
        return '<{cls}:{shape}>'.format(
            cls=self.__class__.__name__, shape=self.shape
        )

    def __repr__(self):
        return self.__str__()

    @property
    def extent(self):
        return tuple(d.extent for d in self.domains)

    @property
    def span(self):
        return tuple(l for e in self.extent for l in zip(*e))

    @property
    def size(self):
        return tuple(d.size for d in self.domains)

    @property
    def shape(self):
        return tuple(j for i in self.size for j in i)


class SerializableDataset(Dataset):
    """ Dataset with a location attribute and a tostring() method. """

    def __init__(self, locus, *args, **kwargs):
        self.locus = locus
        super(SerializableDataset, self).__init__(*args, **kwargs)

    def tostring(self):
        """ Return serialized dataset string. """
        return b''.join(([self.locus.tostring()] +
                         [n.tostring()
                          for n in self.axes] +
                         [self.data.tostring()]))


gdal.UseExceptions()
gmdrv = gdal.GetDriverByName(b'mem')
gtype = {np.dtype('f4'): gdal.GDT_Float32}
