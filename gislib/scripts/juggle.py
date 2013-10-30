# -*- coding: utf-8 -*-
# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from multiprocessing.pool import ThreadPool

import argparse
import itertools
import multiprocessing
import os
import re
import shlex
import signal
import subprocess


def get_parser():
    """ Return arguments dictionary. """
    parser = argparse.ArgumentParser(description='No description yet.')
    parser.add_argument('source',
                        help=('No help here.'))
    parser.add_argument('target',
                        help=('No help here.'))
    parser.add_argument('-c', '--command',
                        choices=['hillshade',
                                 'translate',
                                 'rgba',
                                 'landuse'],
                        help=('No help here.'))
    parser.add_argument('-s', '--strategy',
                        choices=['keep', 'flat'],
                        help=('No help here.'))
    parser.add_argument('-p', '--pattern',
                        choices=['aig', 'tif'],
                        help=('No help here.'))
    parser.add_argument('-w', '--workers',
                        nargs='?',
                        type=int,
                        help=('No help here.'))
    return parser


def pre():
    """ Execute this as preexec in subprocess, to prevent interruption. """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def ext(name):
    """ Replace extension by '.tif' """
    root, ext = os.path.splitext(name)
    return root + '.tif'


class Command(object):
    """ Container for commands. """
    def hillshade(self, sourcepath, targetpath):
        return ' '.join([
            'gdaldem hillshade -compute_edges -co "COMPRESS=DEFLATE" -q',
            sourcepath, targetpath,
        ])

    def translate(self, sourcepath, targetpath):
        """ Convert to compressed tiff with RD srs. """
        return ' '.join([
            #'gdal_translate -ot Byte -of gtiff -co COMPRESS=DEFLATE',
            'gdal_translate -of gtiff -co COMPRESS=DEFLATE',
            '-a_srs epsg:28992 -q',
            sourcepath, targetpath,
        ])

    def rgba(self, sourcepath, targetpath):
        """ Store float as rgba. """
        return ' '.join([
            'juggle_rgba',
            sourcepath, targetpath,
        ])

    def landuse(self, sourcepath, targetpath):
        """ Fetch rgba from geoserver and store as tif. """
        return ' '.join([
            'juggle_landuse',
            sourcepath, targetpath,
        ])


class Strategy(object):
    """
    Container for strategies.
    """
    def keep(self, sourcepath, source, target):
        """ Copy into target path with same structure. """
        targetpath = re.sub(
            '^' + source.rstrip('/'),
            target.rstrip('/'),
            sourcepath,
        )
        return ext(targetpath)

    def flat(self, sourcepath, source, target):
        """ Make a flat folder structure. """
        targetpath = os.path.join(target, os.path.basename(sourcepath))
        return ext(targetpath)


class Pattern(object):
    """ Container for patterns. """
    aig = re.compile('^i[0-9][0-9][a-z][a-z][0-9]_[0-9][0-9]$')
    tif = re.compile('.*\.tif$')


def get_jobs(source, target, command, pattern, strategy):
    """ Return sourcepath, targetpath. """
    matched, existed, processed = 0, 0, 0
    for basepath, dirnames, filenames in os.walk(source, followlinks=True):
        for name in dirnames + filenames:
            if pattern.match(name):
                matched += 1
                sourcepath = os.path.join(basepath, name)
                targetpath = strategy(sourcepath, source, target)
                try:
                    os.makedirs(os.path.dirname(targetpath))
                except OSError:
                    pass  # It is ok if it already exists
                if os.path.exists(targetpath):
                    existed += 1
                    print('matched: {}, processed: {}, existed: {}.'.format(
                        matched, processed, existed,
                    ))
                    continue
                processed += 1
                print('matched: {}, processed: {}, existed: {}.'.format(
                    matched, processed, existed,
                ))
                yield command(sourcepath=sourcepath, targetpath=targetpath)


def execute(job):
    """ Execute the requested command. """
    subprocess.Popen(shlex.split(job), preexec_fn=pre).wait()


def command(source, target, command, pattern, strategy, workers):
    """ Main command. """
    # Fetch attributes
    pattern = getattr(Pattern, pattern)
    strategy = getattr(Strategy(), strategy)
    command = getattr(Command(), command)

    # Going to supply it in batches to the thread pool.
    processes = multiprocessing.cpu_count() if workers is None else workers
    pool = ThreadPool(processes=processes)
    jobs = get_jobs(source, target, command, pattern, strategy)

    while True:
        batch = list(itertools.islice(jobs, 0, processes))
        if batch:
            pool.map(execute, batch)
        else:
            break


def main():
    return command(**vars(get_parser().parse_args()))
