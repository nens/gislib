# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from scipy import ndimage
import numpy as np


class DoesNotFitError(Exception):
    """
    Raised by reproject functions if non-equidistant time domains do
    not fit in the extents of the dataset.
    """
    pass  # But probably some init will be here to transport some data.


def reproject(source, target):
    """
    Use simple affine transform to place source data in target.

    However, if domains in source differ from those in target, additional
    measures must be taken.  - space: Use gdal to warp the spacedomain -
    do not affine that one. Make a big spatial dataset from

    - calendar: Use nedcdf or some coards library to convert extents
    - time:
        - Determine target extent in source calendar
        - Convert only relevant values to target
        - Determine overflow in target - what to do with it? Discard, too.
    - The add_from is responsible for picking the correct datasets for
    writing the data to. It should automatically return datasets for
    which the data fits

    It is still a bit slow. We could just copy in case the diagonal is
    some ones. Also, we could a nearest neighbour project via slicing.

    Raises DoesNotFitError if the source does not fit in the target.
    """
    import ipdb; ipdb.set_trace()
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
