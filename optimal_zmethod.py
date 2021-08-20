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


import knee.rdp as rdp
import optimization.de as de
import knee.zmethod as zmethod
import knee.postprocessing as pp
import knee.evaluation as evaluation


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Global variable for optimization method
args = None
traces = []
expected = []

x_max = None
y_range = None


# Ram cache
cost_cache = {}

# joblib cache
location = tempfile.gettempdir()
memory = joblib.Memory(location, verbose=0)


# RDP cache
rdp_cache = memory.cache(rdp.rdp)


def compute_knee_cost(r, dx, dy, dz, ex, ey):
    # RDP
    points_reduced, removed = rdp_cache(points, r)
        
    ## Knee detection code ##
    knees = zmethod.knees(points_reduced, dx=dx, dy=dy, dz=dz, x_max=x_max, y_range=y_range)
    knees = knees[knees > 0]
    knees = pp.add_points_even(points, points_reduced, knees, removed, tx=ex, ty=ey)

    ## Average cost
    cost_a = evaluation.rmspe(points, knees, expected, evaluation.Strategy.knees)
    cost_b = evaluation.rmspe(points, knees, expected, evaluation.Strategy.expected)
    #cost = (cost_a+cost_b)/2.0

    cost = max(cost_a, cost_b)

    return cost

# Knees cache
#cost_cache = memory.cache(compute_knee_cost)


def objective(p):
    # Round input parameters 
    r = round(p[0]*100.0)/100.0
    dx = round(p[1]*100.0)/100.0
    dy = round(p[2]*100.0)/100.0
    dz = round(p[3]*100.0)/100.0
    ex = round(p[4]*100.0)/100.0
    ey = round(p[5]*100.0)/100.0

    if (r, dx, dy, dz, ex, ey) in cost_cache:
        return cost_cache[(r, dx, dy, dz, ex, ey)]
    else:
        cost = compute_knee_cost(r, dx, dy, dz, ex, ey)
        cost_cache[(r, dx, dy, dz, ex, ey)] = cost
        return cost


def main(args):
    # Get all files from the input folder
    print(args.i)
    files = []
    for f in os.listdir(args.i):
        if re.match(r'w[0-9]+-(arc|lru).csv', f):
            files.append(f)
    files.sort()
    print(files)

    '''# Get the points
    global points 
    points = np.genfromtxt(args.i, delimiter=',')

    # Get original x_max and y_ranges
    global x_max
    x_max = [max(x) for x in zip(*points)][0]
    global y_range
    y_range = [[max(y),min(y)] for y in zip(*points)][1]

    # Get the expected values
    global expected
    dirname = os.path.dirname(args.i)
    filename = os.path.splitext(os.path.basename(args.i))[0]
    expected_file = os.path.join(os.path.normpath(dirname), f'{filename}_expected.csv')

    if os.path.exists(expected_file):
        with open(expected_file, 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            expected = list(reader)
    else:
        expected = []
    expected = np.array(expected)

    # Run the Genetic Optimization
    bounds = np.asarray([[.85, .95], [.01, .1], [.01, .1], [.01, .1], [.01, .1], [.01, .1]])
    #best, score = de.differential_evolution(objective, bounds, crossover, mutation, n_iter=args.l, n_pop=args.p)
    best, score = de.differential_evolution(objective, bounds, n_iter=args.l, n_pop=args.p)

    # Round input parameters
    r = round(best[0]*100.0)/100.0
    dx = round(best[1]*100.0)/100.0
    dy = round(best[2]*100.0)/100.0
    dz = round(best[3]*100.0)/100.0
    ex = round(best[4]*100.0)/100.0
    ey = round(best[5]*100.0)/100.0
    logger.info('%s (%s, %s, %s, %s, %s, %s) = %s', args.i, r, dx, dy, dz, ex, ey, score)'''
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Z-Method Optimal Knee')
    parser.add_argument('-i', type=str, required=True, help='input folder')
    parser.add_argument('-p', type=int, help='population size', default=20)
    parser.add_argument('-l', type=int, help='number of loops (iterations)', default=100)
    args = parser.parse_args()
    
    main(args)