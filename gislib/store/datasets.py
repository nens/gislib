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


def reproject(source, target):
    """
    It's more complex.

    Determine the offset and diagonals from sizes.
    Determine the overlapping extents
    Calculate views for both of them
    Do the trick.
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
    diagonal = (u5 - l5) / (u4 - l4) / (s4 / s5)
    ndimage.affine_transform(sourceview, diagonal, output=targetview)
        
    targetview.mask = np.ma.equal(targetview.data, targetview.fill_value)
    print('reproject')


class Config(object):
    """ Collection of dataset scales. """
    def __init__(self, domains):
        self.domains = domains

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
    def __init__(self, config, axes, data):
        """
        """
        self.config = config
        self.axes = axes
        self.data = data


class SerializableDataset(Dataset):
    """ Dataset with a location attribute and a tostring() method. """

    def __init__(self, location, *args, **kwargs):
        self.location = location
        super(SerializableDataset, self).__init__(*args, **kwargs)

    def tostring(self):
        """ Return serialized dataset string. """
        return b''.join(([self.location.tostring()] +
                         [n.tostring()
                          for n in self.axes] +
                         [self.data.filled().tostring()]))


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
