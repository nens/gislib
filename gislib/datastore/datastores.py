# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from gislib.datastore import chunks
from gislib.datastore import files
from gislib.datastore import utils
from gislib.datastore import aggregators
from gislib.datastore import dimensions

""" General code about the module here """


class Datastore(object):
    """
    Raster object with optional arguments including
    time. Choices to be made:

    chunksize, filesize
    dimensions: ['time', 'spatial', 'spatial']
    calendars:
    equidistant time or not?
    equidistant x y or not?
    z or not?
    coupled x y
    """
    def __init__(self, path):
        pass

    def update(data, extent):
        """
        Update extent with data in numpy masked array
        """
        pass


