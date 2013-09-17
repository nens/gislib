#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import gdal
import numpy as np
import sys


def main():
    inputpath = sys.argv[1]
    outputpath = sys.argv[2]

    source = gdal.Open(inputpath)

    rgba = np.fromstring(
        source.ReadRaster(0, 0, source.RasterXSize, source.RasterYSize),
        dtype=np.uint8,
    ).reshape(source.RasterYSize, source.RasterXSize, 4)

    target = gdal.GetDriverByName(b'GTiff').Create(
        outputpath,
        source.RasterXSize,
        source.RasterYSize,
        4,
        gdal.GDT_Byte,
        ["COMPRESS=DEFLATE"],
    )
    target.SetProjection(source.GetProjection())
    target.SetGeoTransform(source.GetGeoTransform())
    for i in range(4):
        band = target.GetRasterBand(i + 1)
        band.WriteArray(rgba[..., i])


if __name__ == '__main__':
    exit(main())
