# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from osgeo import gdal
from osgeo import osr
from matplotlib import cm
from matplotlib import colors
import numpy as np

from gislib import projections
from gislib import vectors

# Enable gdal exceptions
gdal.UseExceptions()

LANDUSE = {
    1: '1 - BAG - Overig / Onbekend',
    2: '2 - BAG - Woonfunctie',
    3: '3 - BAG - Celfunctie',
    4: '4 - BAG - Industriefunctie',
    5: '5 - BAG - Kantoorfunctie',
    6: '6 - BAG - Winkelfunctie',
    7: '7 - BAG - Kassen',
    8: '8 - BAG - Logiesfunctie',
    9: '9 - BAG - Bijeenkomstfunctie',
    10: '10 - BAG - Sportfunctie',
    11: '11 - BAG - Onderwijsfunctie',
    12: '12 - BAG - Gezondheidszorgfunctie',
    13: '13 - BAG - Overig kleiner dan 50 m2<br />(schuurtjes)',
    14: '14 - BAG - Overig groter dan 50 m2<br />(bedrijfspanden)',
    15: '15 - BAG - None',
    16: '16 - BAG - None',
    17: '17 - BAG - None',
    18: '18 - BAG - None',
    19: '19 - BAG - None',
    20: '20 - BAG - None',
    21: '21 - Top10 - Water',
    22: '22 - Top10 - Primaire wegen',
    23: '23 - Top10 - Secundaire wegen',
    24: '24 - Top10 - Tertiaire wegen',
    25: '25 - Top10 - Bos/Natuur',
    26: '26 - Top10 - Bebouwd gebied',
    27: '27 - Top10 - Boomgaard',
    28: '28 - Top10 - Fruitkwekerij',
    29: '29 - Top10 - Begraafplaats',
    30: '30 - Top10 - Agrarisch gras',
    31: '31 - Top10 - Overig gras',
    32: '32 - Top10 - Spoorbaanlichaam',
    33: '33 - Top10 - None',
    34: '34 - Top10 - None',
    35: '35 - Top10 - None',
    36: '36 - Top10 - None',
    37: '37 - Top10 - None',
    38: '38 - Top10 - None',
    39: '39 - Top10 - None',
    40: '40 - Top10 - None',
    41: '41 - LGN - Agrarisch Gras',
    42: '42 - LGN - Mais',
    43: '43 - LGN - Aardappelen',
    44: '44 - LGN - Bieten',
    45: '45 - LGN - Granen',
    46: '46 - LGN - Overige akkerbouw',
    47: '47 - LGN - None',
    48: '48 - LGN - Glastuinbouw',
    49: '49 - LGN - Boomgaard',
    50: '50 - LGN - Bloembollen',
    51: '51 - LGN - None',
    52: '52 - LGN - Gras overig',
    53: '53 - LGN - Bos/Natuur',
    54: '54 - LGN - None',
    55: '55 - LGN - None',
    56: '56 - LGN - Water (LGN)',
    57: '57 - LGN - None',
    58: '58 - LGN - Bebouwd gebied',
    59: '59 - LGN - None',
    61: '61 - CBS - Spoorwegen terrein',
    62: '62 - CBS - Primaire wegen',
    63: '63 - CBS - Woongebied',
    64: '64 - CBS - Winkelgebied',
    65: '65 - CBS - Bedrijventerrein',
    66: '66 - CBS - Sportterrein',
    67: '67 - CBS - Volkstuinen',
    68: '68 - CBS - Recreatief terrein',
    69: '69 - CBS - Glastuinbouwterrein',
    70: '70 - CBS - Bos/Natuur',
    71: '71 - CBS - Begraafplaats',
    72: '72 - CBS - Zee',
    73: '73 - CBS - Zoet water',
    74: '74 - CBS - None',
    75: '75 - CBS - None',
    76: '76 - CBS - None',
    77: '77 - CBS - None',
    78: '78 - CBS - None',
    79: '79 - CBS - None',
    97: '97 - Overig - buitenland',
    98: '98 - Top10 - erf',
    99: '99 - Overig - Overig/Geen landgebruik'
}

def get_extent_intersection(extent1, extent2):
    """ Return the intersecting extent. """
    return (max(extent1[0], extent2[0]),
            max(extent1[1], extent2[1]),
            min(extent1[2], extent2[2]),
            min(extent1[3], extent2[3]))


def get_transformed_extent(extent, source_projection, target_projection):
    """
    Return extent transformed from source projection to target projection.
    """
    polygon = vectors.Geometry.fromextent(*extent).envelope
    transformation = osr.CoordinateTransformation(
        projections.get_spatial_reference(source_projection),
        projections.get_spatial_reference(target_projection),
    )
    polygon.Transform(transformation)
    return vectors.Geometry(geometry=polygon).extent


def get_curve(array, bins=256):
    """
    TODO Better name.

    Return array with graph points.
    """
    # compute histogram
    histogram, edges = np.histogram(array.compressed(), bins)

    # convert to percentiles
    percentile_x = np.cumsum(histogram) / float(histogram.sum()) * 100
    percentile_y = edges[1:]  # right edges of bins.
    curve_x = np.arange(0, 101)
    curve_y = np.interp(curve_x, percentile_x, percentile_y)
    return curve_x, curve_y


def get_landuse_counts(array, bins=256):
    """ 

    TODO: Retrieve colo(u)rs and labels from metadata file
          so this file does not have to have the LANDUSE global
          and for make benefit of genericity.

    Return array with counts of LANDUSE labels.

    """
    bins = np.arange(0, bins)
    histograms = [np.histogram(d.compressed(), bins)[0] for d in array]
    nonzeros = [h.nonzero() for h in histograms]
    nbins = [bins[:-1][n] for n in nonzeros]
    nhistograms = [h[n] for n, h in zip(nonzeros, histograms)]
    # Determine the ordering
    argsorts = [h.argsort() for h in nhistograms]
    arg10 = [a[:-10:-1] for a in argsorts]
    argrest = [a[-10::-1] for a in argsorts]
    # Use it to group
    rests = [h[argrest].sum() for h, a in zip(nhistograms, argsorts)]
    pairs = [np.array([b[arg10], h[arg10]]).transpose()
             for b, h, a in zip(nbins, nhistograms, argsorts)]
    # Prepare result data
    # NOTE: hardcoded style not sure what styles are supported
    styles = ['terrain']
    colormaps = [cm.get_cmap(s.split(':')[0]) for s in styles]
    result = [[dict(label=LANDUSE.get(b, b),
                    data=d,
                    color=colors.rgb2hex((c(b))))
               for b, d in p.tolist()] for p, c in zip(pairs, colormaps)]

    for r, s in zip(result, rests):
        if s:
            r.append(dict(label='Overig', data=float(s), color='#ffffff'))
    return result
