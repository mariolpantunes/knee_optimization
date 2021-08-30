#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import re
import csv
import math
import joblib
import logging
import tempfile
import argparse
import numpy as np


from enum import Enum
from matplotlib import pyplot


import knee.rdp as rdp
import optimization.de as de
import knee.zmethod as zmethod
import knee.postprocessing as pp
import knee.evaluation as evaluation


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Metric(Enum):
    avg = 'avg'
    max = 'max'

    def __str__(self):
        return self.value


# Global variable for optimization method
args = None
traces = []
x_max = []
y_range = []
expecteds = []


# joblib cache
location = tempfile.gettempdir()
limit = 256 * 1024
memory = joblib.Memory(location, bytes_limit=limit, verbose=0)


# RDP cache
rdp_cache = memory.cache(rdp.rdp)


def compute_knee_cost(trace, expected, x, y, r, dx, dy, dz, ex, ey):
    # RDP
    points_reduced, removed = rdp_cache(trace, r)

    ## Knee detection code ##
    knees = zmethod.knees(points_reduced, dx=dx, dy=dy, dz=dz, x_max=x, y_range=y)
    knees = knees[knees > 0]
    knees = pp.add_points_even(trace, points_reduced, knees, removed, tx=ex, ty=ey)
    if len(knees) == 0:
        return float('inf'), float('inf')

    ## Average cost
    cost_a = evaluation.rmspe(trace, knees, expected, evaluation.Strategy.knees)
    cost_b = evaluation.rmspe(trace, knees, expected, evaluation.Strategy.expected)

    return cost_a, cost_b


# Cost cache
knee_cost_cache = memory.cache(compute_knee_cost)


def compute_knees_cost(r, dx, dy, dz, ex, ey):
    costs = []

    for i in range(len(traces)):
        trace = traces[i]
        expected = expecteds[i]
        x = x_max[i]
        y = y_range[i]

        cost_a, cost_b = knee_cost_cache(trace, expected, x, y, r, dx, dy, dz, ex, ey)
        
        if args.m is Metric.max:
            cost = max(cost_a, cost_b)
        else:
            cost = (cost_a+cost_b)/2.0
        costs.append(cost)
    
    costs = np.array(costs)

    if args.m is Metric.max:
        return np.amax(costs) 
    else:
        return np.average(costs)

def objective(p):
    # Round input parameters 
    r = round(p[0]*100.0)/100.0
    dx = round(p[1]*100.0)/100.0
    dy = round(p[2]*100.0)/100.0
    dz = round(p[3]*100.0)/100.0
    ex = round(p[4]*100.0)/100.0
    ey = round(p[5]*100.0)/100.0

    return compute_knees_cost(r, dx, dy, dz, ex, ey)


def main(args):
    # Get all files from the input folder
    logger.info(f'Base path: {args.i}')
    files = []
    for f in os.listdir(args.i):
        if re.match(r'w[0-9]+-(arc|lru).csv', f):
            files.append(f)
    files.sort()

    # Get the traces 
    global traces, x_max, y_range, expecteds
    for f in files:
        path = os.path.join(os.path.normpath(args.i), f)
        points = np.genfromtxt(path, delimiter=',')
        traces.append(points)

        # Get original x_max and y_ranges
        x_max.append([max(x) for x in zip(*points)][0])
        y_range.append([[max(y),min(y)] for y in zip(*points)][1]) 

        # Get the expected values
        filename = os.path.splitext(f)[0]
        expected_file = os.path.join(os.path.normpath(args.i), f'{filename}_expected.csv')

        e = []
        if os.path.exists(expected_file):
            with open(expected_file, 'r') as f:
                reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
                e = list(reader)
        expecteds.append(np.array(e))
    logger.info(f'Loaded {len(traces)} trace(s)')
    
    # Run the Differential Evolution Optimization
    logger.info(f'Running the Differential Evolution Optimization ({args.p}, {args.l}, {args.m})')
    bounds = np.asarray([[.85, .95], [.01, .2], [.01, .2], [.01, .2], [.01, .2], [.01, .2]])
    best, score, iter = de.differential_evolution(objective, bounds, n_iter=args.l, n_pop=args.p, cached=False, debug=True)

    # Round input parameters
    r = round(best[0]*100.0)/100.0
    dx = round(best[1]*100.0)/100.0
    dy = round(best[2]*100.0)/100.0
    dz = round(best[3]*100.0)/100.0
    ex = round(best[4]*100.0)/100.0
    ey = round(best[5]*100.0)/100.0
    logger.info('Best configuration (%s, %s, %s, %s, %s, %s) = %s', r, dx, dy, dz, ex, ey, score)

    # Plot the optimization evolution
    pyplot.plot(iter, '.-')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Cost Function')
    pyplot.savefig('optimization.pdf', bbox_inches='tight')
    pyplot.close()
    
    # Compute the RMSPE for all the traces using the cache
    with open(args.o, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Files', 'RMSPE(k)', 'RMSPE(E)'])

        for i in range(len(traces)):
            trace = traces[i]
            expected = expecteds[i]
            x = x_max[i]
            y = y_range[i]

            cost_a, cost_b = knee_cost_cache(trace, expected, x, y, r, dx, dy, dz, ex, ey)
            writer.writerow([files[i], cost_a, cost_b])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Z-Method Optimal Knee')
    parser.add_argument('-i', type=str, required=True, help='input folder')
    parser.add_argument('-o', type=str, help='output CSV', default='results.csv')
    parser.add_argument('-p', type=int, help='population size', default=30)
    parser.add_argument('-l', type=int, help='number of loops (iterations)', default=100)
    parser.add_argument('-m', type=Metric, choices=list(Metric), help='Metric type', default='avg')
    args = parser.parse_args()
    
    main(args)