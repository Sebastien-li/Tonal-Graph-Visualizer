"""Utility functions for the project."""
import sys
from datetime import datetime
from pathlib import Path
import logging as log
import bisect
import numpy as np


def find_le(a, x, key = None):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right([key(y) for y in a], x)
    if i:
        return a[i-1]
    raise ValueError

def display_float(x, precision = 3):
    """ Returns a string representation of a float with a given precision. Removes trailing zeros"""
    return f'{float(x):.{precision}f}'.rstrip('0').rstrip('.')

def interval_collision(interval1_start, interval1_end, interval2_start, interval2_end):
    """ Returns True if two intervals overlap."""
    return interval1_start < interval2_end and interval1_end > interval2_start

def interval_in(interval1_start, interval1_end, interval2_start, interval2_end):
    """ Returns True if interval 1 in interval 2."""
    return interval1_start >= interval2_start and interval1_end <= interval2_end

def octave_weight(octave):
    # 1 / (1 + e^(octave-6))
    return [0.997, 0.993, 0.982, 0.952, 0.880, 0.731, 0.5, 0.268, 0.119, 0.047][octave]

def duration_weight(duration):
    return duration ** 0.5

def doubling_weight(nb_double):
    # sqrt(nb_double/3) or 1
    if nb_double == 1:
        return 0.577
    if nb_double == 2:
        return 0.816
    return 1

def get_relative_duration(rhythm_node, node):
    intersected_duration = min(rhythm_node.offset, node['offset']) - max(rhythm_node.onset, node['onset'])
    return intersected_duration / rhythm_node.duration

def cartesian_product(*arrays):
    """ Returns the cartesian product of a list of arrays."""
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def get_multilogger():
    """ Returns a logger that writes to a file and to the console."""
    logger = log.getLogger('HALight logger')
    formater = log.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger.setLevel(log.DEBUG)

    log_folder = Path('logs')
    log_folder.mkdir(exist_ok=True)
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    handler_debug = log.FileHandler(Path(f'logs/{now}.log'), mode='w')
    handler_debug.setLevel(log.DEBUG)
    handler_debug.setFormatter(formater)
    logger.addHandler(handler_debug)

    handler_info = log.StreamHandler(sys.stdout)
    handler_info.setLevel(log.INFO)
    handler_info.setFormatter(formater)
    logger.addHandler(handler_info)
    return logger
