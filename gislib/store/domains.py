# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import collections

# A number of domains for various purposes

Time = collections.namedtuple('Time', ('calendar'))
Space = collections.namedtuple('Space', ('projection'))
