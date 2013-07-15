# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division


class Config(object):
    def __init__(self, domains):
        self.domains = domains

    @property
    def size(self):
        return tuple(s.domain.size for s in self.domains)

    @property
    def shape(self):
        return tuple(j for i in self.size for j in i)
