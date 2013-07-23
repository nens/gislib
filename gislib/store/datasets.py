# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections

from scipy import ndimage
import numpy as np

Domain = collections.namedtuple('Domain', ('domain', 'extent', 'size'))

class DoesNotFitError(Exception):
    """
    Raised by reproject functions if non-equidistant time domains do
    not fit in the extents of the dataset.
    """
    pass


def reproject(source, target):
    """
    Use simple affine transform to place source data in target.

    However, if domains in source differ from those in target, additional measures must be taken.
    - projection: Use gdal to warp the spacedomain - do not affine that one.
    - calendar: Use nedcdf or some coards library to convert extents
    - non-equidistant time:
        - Determine target extent in source calendar
        - Convert only relevant values to target
        - Determine overflow in target - what to do with it? Discard, too.
    - The add_from is responsible for picking the correct datasets for writing the data to. It should automatically return datasets for which the data fits

    It is still a bit slow. We could just copy in case the diagonal is
    some ones. Also, we could a nearest neighbour project via slicing.

    Raises DoesNotFitError if the source does not fit in the target.
    """
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

class BaseDomain(object):
    """ Base class for dataset domains. """
    def __init__(self, size, extent):
        self.size = size
        self.extent = extent


class SpaceDomain(BaseDomain):
    """ A domain containing gdal datasets. """
    def __init__(self, projection, *args, **kwargs):
        self.projection = projection
        super(self, SpaceDomain).__init__(*args, **kwargs)
        

class TimeDomain(BaseDomain):
    """ A domain containing gdal datasets. """
    def __init__(self, calendar, equidistant=True, *args, **kwargs):
        self.projection = projection
        self.equidistant = equidistant
        super(self, TimeDomain).__init__(*args, **kwargs)


class Config(object):
    """ Collection of dataset scales. """
    def __init__(self, domains, fill):
        self.domains = domains
        self.fill = fill
        # dtype? Or not?

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


class Dataset(object):
    def __init__(self, conf, axes, data):
        """ Dataset. """
        self.conf = conf
        self.axes = axes  # A tuple of numpy arrays corresponding to the ned domains
        self.data = data  # The numpy array 


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


class Converter(object):
    """
    Convert datasets.

    You have a dataset with one domain (domains, extents) and need to
    convert to another.

    How?

    """
    def convert(self, source, target):
        """
        Adds data from source to target.

        Only target pixels that are within the extent of source are affected.

        For now, we do just resample, that is, simple affine from scipy.
        """
        # Determine the extent of source.
        # Determine the view of the target that is
        # within the extent of the source

        # Determine the view of the target array that is affected
        # Determine the transformation
        # Perform it.
        # Done.
