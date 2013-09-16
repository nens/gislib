#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

import sys


class Indicator(object):
    """
    Progress indicator.

    Instantiate with amount of calls to update() expected. If you have
    only an estimate, be sure to call complete() afterwards, to render
    a completed progress bar.
    """

    INDICATOR = []
    for text in map(str, range(00, 100, 10)):
        INDICATOR.append(text)
        INDICATOR.extend(3 * '.')
    INDICATOR.append('100 - done.\n')

    def __init__(self, total):
        """ Set the expected length of the job. """
        self.total = total
        self.count = 0  # Update counter
        self.position = 0  # Indicator position
        self._progress()  # Display the first item

    def _progress(self):
        """ Update indicator one position. """
        sys.stdout.write(self.INDICATOR[self.position])
        sys.stdout.flush()
        self.position += 1

    def update(self):
        """
        Update progress one unit towards end.

        Because of the min(), excessive updates don't crash.
        """
        self.count += 1
        fraction = self.count / self.total
        position = min(fraction, 1) * (len(self.INDICATOR) - 1)
        while self.position <= position:
            self._progress()

    def complete(self):
        """ Complete progress. """
        while self.position < len(self.INDICATOR):
            self._progress()


def main():
    """ Kind of test suite. """
    import time
    # Few steps
    length = 5
    p = Indicator(total=length)
    for i in range(length):
        p.update()
    # Too many updates
    length = 5
    p = Indicator(total=length)
    for i in range(length * 2):
        p.update()
    # Complete before end
    length = 5
    p = Indicator(total=length)
    for i in range(length // 2):
        p.update()
    p.complete()
    # Many steps
    length = 500
    p = Indicator(total=length)
    p.update()
    time.sleep(1)
    for i in range(length - 2):
        time.sleep(0.001)
        p.update()
    time.sleep(1)
    p.update()


if __name__ == '__main__':
    exit(main())
