# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from matplotlib import cm
from matplotlib import colors


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

LC_LANDUSE_BEIRA = [
    (0.84, 1.0, 0.74),
    (0.19, 0.08, 0.81),
    (1.0, 0.49, 0.48),
    (1.0, 0.49, 0.48),
    (0.61, 0.62, 0.61),
]

LC_LANDUSE = [
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.741, 0.235, 0.322, 1.0),
    (0.969, 0.765, 0.776, 1.0),
    (0.969, 0.349, 0.224, 1.0),
    (0.969, 0.22, 0.353, 1.0),
    (0.808, 0.525, 0.58, 1.0),
    (0.969, 0.588, 0.482, 1.0),
    (0.741, 0.204, 0.192, 1.0),
    (1.0, 0.412, 0.518, 1.0),
    (0.969, 0.427, 0.388, 1.0),
    (0.741, 0.443, 0.388, 1.0),
    (0.741, 0.588, 0.549, 1.0),
    (0.741, 0.235, 0.322, 1.0),
    (0.741, 0.235, 0.322, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.443, 1.0, 1.0),
    (0.192, 0.204, 0.192, 1.0),
    (0.42, 0.412, 0.42, 1.0),
    (0.71, 0.698, 0.71, 1.0),
    (0.0, 0.443, 0.29, 1.0),
    (0.741, 0.235, 0.322, 1.0),
    (0.451, 0.443, 0.0, 1.0),
    (0.451, 0.443, 0.0, 1.0),
    (0.906, 0.89, 0.906, 1.0),
    (0.647, 1.0, 0.451, 1.0),
    (0.647, 1.0, 0.451, 1.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.647, 1.0, 0.451, 1.0),
    (1.0, 1.0, 0.451, 1.0),
    (0.808, 0.667, 0.388, 1.0),
    (1.0, 0.0, 0.776, 1.0),
    (0.906, 0.906, 0.0, 1.0),
    (0.451, 0.302, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.969, 0.588, 0.482, 1.0),
    (0.451, 0.443, 0.0, 1.0),
    (0.871, 0.443, 1.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.322, 1.0, 0.0, 1.0),
    (0.0, 0.443, 0.29, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.741, 0.235, 0.322, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.776, 0.365, 0.388, 1.0),
    (0.808, 0.525, 0.58, 1.0),
    (0.969, 0.22, 0.353, 1.0),
    (0.969, 0.427, 0.388, 1.0),
    (0.451, 0.541, 0.259, 1.0),
    (0.224, 0.667, 0.0, 1.0),
    (0.969, 0.588, 0.482, 1.0),
    (0.0, 0.443, 0.29, 1.0),
    (0.906, 0.89, 0.906, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.906, 0.89, 0.906, 1.0),
]


def add_cmap_transparent(name):
    """ Create and register a transparent colormap. """
    cmap = colors.ListedColormap([(0, 0, 0, 0)])
    cm.register_cmap(name, cmap)


def add_cmap_damage(name):
    """ Create and register damage colormap. """
    f = 0.05
    cdict = {'red':   [(0.0,       0, 0),
                       (00.01 * f, 0, 1),
                       (1.0,       1, 1)],
             'green': [(0.,        0.00, 0.00),
                       (00.01 * f, 0.00, 1.00),
                       (00.50 * f, 1.00, 0.65),
                       (10.00 * f, 0.65, 0.00),
                       (1.,        0.00, 0.00)],
             'blue':  [(0., 0, 0),
                       (1., 0, 0)],
             'alpha': [(0.,        0, 0),
                       (00.01 * f, 0, 1),
                       (1.,        1, 1)]}

    cmap = colors.LinearSegmentedColormap('', cdict)
    cm.register_cmap(name, cmap)


def add_cmap_shade(name, ctr):
    """ Create and register shade colormap. """
    cdict = {'red':   [(0.0, 9.9, 0.0),
                       (ctr, 0.0, 1.0),
                       (1.0, 1.0, 9.9)],

             'green': [(0.0, 9.9, 0.0),
                       (ctr, 0.0, 1.0),
                       (1.0, 1.0, 9.9)],

             'blue':  [(0.0, 9.9, 0.0),
                       (ctr, 0.0, 1.0),
                       (1.0, 1.0, 9.9)],

             'alpha': [(0.0, 9.9, 1.0),
                       (ctr, 0.0, 0.0),
                       (1.0, 1.0, 9.9)]}

    cmap = colors.LinearSegmentedColormap('', cdict)
    cm.register_cmap(name, cmap)


def add_cmap_drought(name):
    """ Create and register drought colormap. """
    cdict = {
        'red': [
            (0.0, 0.0, 0.0),
            (0.062, 0.0, 0.157),
            (0.125, 0.157, 0.51),
            (0.188, 0.51, 0.749),
            (0.25, 0.749, 0.149),
            (0.312, 0.149, 0.357),
            (0.375, 0.357, 0.576),
            (0.438, 0.576, 0.827),
            (0.5, 0.827, 0.902),
            (0.562, 0.902, 0.949),
            (0.625, 0.949, 0.98),
            (0.688, 0.98, 1.0),
            (0.75, 1.0, 0.659),
            (0.812, 0.659, 0.8),
            (0.875, 0.8, 0.91),
            (0.938, 0.91, 1.0),
            (1.0, 1.0, 0.0)
        ],
        'green': [
            (0.0, 0.0, 0.149),
            (0.062, 0.149, 0.314),
            (0.125, 0.314, 0.533),
            (0.188, 0.533, 0.824),
            (0.25, 0.824, 0.451),
            (0.312, 0.451, 0.62),
            (0.375, 0.62, 0.8),
            (0.438, 0.8, 1.0),
            (0.5, 1.0, 0.902),
            (0.562, 0.902, 0.941),
            (0.625, 0.941, 0.965),
            (0.688, 0.965, 1.0),
            (0.75, 1.0, 0.0),
            (0.812, 0.0, 0.286),
            (0.875, 0.286, 0.502),
            (0.938, 0.502, 0.749),
            (1.0, 0.749, 0.0)
        ],
        'blue': [
            (0.0, 0.0, 0.451),
            (0.062, 0.451, 0.631),
            (0.125, 0.631, 0.812),
            (0.188, 0.812, 1.0),
            (0.25, 1.0, 0.0),
            (0.312, 0.0, 0.243),
            (0.375, 0.243, 0.471),
            (0.438, 0.471, 0.749),
            (0.5, 0.749, 0.0),
            (0.562, 0.0, 0.322),
            (0.625, 0.322, 0.529),
            (0.688, 0.529, 0.749),
            (0.75, 0.749, 0.0),
            (0.812, 0.0, 0.208),
            (0.875, 0.208, 0.447),
            (0.938, 0.447, 0.749),
            (1.0, 0.749, 0.0)
        ],
    }
    cmap = colors.LinearSegmentedColormap('', cdict)
    cm.register_cmap(name, cmap)


def add_cmap_landuse(name):
    """ Create and register a transparent colormap. """
    cmap = colors.ListedColormap(LC_LANDUSE)
    cm.register_cmap(name, cmap)


def add_cmap_landuse_beira(name):
    """ Create and register a transparent colormap. """
    cmap = colors.ListedColormap(LC_LANDUSE_BEIRA)
    cm.register_cmap(name, cmap)
