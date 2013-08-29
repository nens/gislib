# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import math

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

def _slice(domain, axis, extent):
    """
    Return indices into axes for which points are in extent.

    Not suitable for multidimensional domains.
    """
    values = domain.values(axis)
    ((e1,), (e2,)) = extent
    where = np.where(np.logical_and(values >= e1, values < e2))[0]
    try:
        return slice(where.min(), where.max() + 1)
    except ValueError:
        return slice(0, 0)

def _create_source_view(source, target):
    """
    Returns view to the source dataset with the discrete variables
    already transformed to their respective kinds of the target dataset,
    clipped to contain only relevant data for reprojection into the
    target dataset.
    """
    # This whole part may not be necessary, if the domains take care
    # of building a view to the data that matches the target extent. Make
    # code prettier :)
    dicts = []
    for sd, sa, td in zip(source.domains, source.axes, target.domains):
        if isinstance(sd.kind, kinds.DiscreteKind):
            # Transform to target kind
            dicts.append(sd.transform(td.kind, axis=sa))
        else:
            # Just pass the original domains and (None) axes
            dicts.append(dict(axis=sa, domain=sd))
    
    # Convert to dataset kwargs
    kwargs = dict(fill=source.fill,
                  data=source.data,
                  axes=tuple(d['axis'] for d in dicts),
                  domains=tuple(d['domain'] for d in dicts))
    _source = Dataset(**kwargs)

    # Make a view.
    # Discrete variables: determine slice into source axes that fits in target extent
    # Continous variables: determine slice tuple from transformed domain extents
    data = zip(_source.domains, _source.axes, target.domains, target.axes)
    slices = []
    for sd, sa, td, ta in data:
        if isinstance(sd.kind, kinds.ContinuousKind):
            # Going to determine the slice from size and extent in transformed domain
            slices.extend(map(lambda i: slice(*i), sd.indices(td)))
        else:
            slices.append(_slice(domain=sd, axis=sa, extent=td.extent))

def _transform_spatial(source_view, target_view):
    pass


def reproject(source, target):
    """ 
    Put relevant data from source into target. 

    source
        Time dimension: 
            - Create a view:
                - New extents matching those of the target
                - New axes according to new extent
                - Sliced data to contain only extent of target
        Space dimension:
            - Create a view based on above view:
                - Slice based on backtransformed target extent
                - New extents according to slices

                
              

        translate extents using extents and axes for ned domains
        Create a view that only holds the extents whe're interested in for
        Then, for all ned domains:
            Use np.in1d to determine existing values
            Raise DoesNotFit if necessary
    target
        Add the non-existing values from source to axes, and sort.
        create a view according to source axes. sort should not be necessary, but check.

    Now do the spatial warp / or, later on, the aggregation operation
    """
    # Need an as fast as possible copydata operation, when kinds and
    # extents are equal, and axes are too, or empty.

    # source  |............|
    # target        |............|
    # source_view   |......|
    # target_view   |......|

    source_view = source.get_view(target)  # Raises DoesNotFitError?
    target_view = target.get_view(source) # Will that work?
    if source_view.data.size and target_view.data.size:
        import ipdb; ipdb.set_trace()
    print(source_view.shape)
    print(target_view.shape)


    #source_indices = np.not_equal(source.axes[1], -1)
    #transformed_source = _transform_spatial(source_view, target_view)


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
        """ Return transformed Domain. """
        if kind == self.kind:
            return self
        extent = self.kind.transform(extent=self.extent, kind=kind)
        return Domain(kind=kind, extent=extent, size=self.size)


    def values(self, axis):
        """ Return values. """
        ((e1,), (e2,)) = self.extent
        return e1 + axis * (e2 - e1)


def get_prep(source_domain, source_axes, target_domain, target_axes):
    """ 
    Return dictionary with slices, axes and domain.
    """
    transformed = target_domain.transform(kind=source_domain.kind)

    kwargs = source_domain.kind.get_prep(
        source_size=source_domain.size,
        source_extent=source_domain.extent,
        source_axes=source_axes,
        target_extent=transformed.extent,
        target_axes=target_axes,
    )
    
    return dict(axes=kwargs.pop('axes'),
                slices=kwargs.pop('slices'),
                domain=Domain(source_domain.kind, **kwargs))

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

    def get_view(self, dataset):
        """
        Return a dataset view that matches the extents of dataset for
        all domains that are not raster domains. The time domain is
        already converted to those of dataset.
        """
        # Have domains prepare for view creation
        preps = map(get_prep, 
                    self.domains, self.axes, dataset.domains, dataset.axes)
        axes = tuple(p['axes'] for p in preps)
        slices = tuple(s for p in preps for s in p['slices'])
        return Dataset(
            domains=tuple(p['domain'] for p in preps),
            axes=axes,
            data=self.data[slices],  # Here the actual view is created
            fill=self.fill,
        )

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
