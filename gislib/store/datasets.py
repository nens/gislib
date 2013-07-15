# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections

from gislib.store import configs

Domain = collections.namedtuple('Domain', ('domain', 'extent'))


class Config(configs.Config):
    """ Collection of dataset scales. """
    @property
    def extent(self):
        return tuple(d.extent for d in self.domains)


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
        return ''.join(*([self.location.tostring()] +
                         [n.tostring()
                          for n in self.axis] +
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
