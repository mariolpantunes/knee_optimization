#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mariolpantunes@gmail.com'
__status__ = 'Development'


import os
import re
import csv
#import joblib
import logging
import tempfile
import argparse
import statistics
import numpy as np
import diskcache as dc

from enum import Enum
from matplotlib import pyplot


import knee.rdp as rdp
import optimization.de as de
import knee.curvature as curvature
import knee.clustering as clustering
import knee.postprocessing as pp
import knee.evaluation as evaluation
from knee.knee_ranking import ClusterRanking


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Metric(Enum):
    mcc = 'mcc'
    rmspe = 'rmspe'

    def __str__(self):
        return self.value


class Agglomeration(Enum):
    average = 'avg'
    maximum = 'max'

    def __str__(self):
        return self.value


# Global variable for optimization method
args = None
traces = []
expecteds = []
#rdp_cache_dic = {}
#cost_cache_dic = {}



# joblib cache
location = tempfile.gettempdir()
#memory = joblib.Memory(location, verbose=0)
cache = dc.Cache(location)

def compute_rdp(idx, r):
    trace = traces[idx]
    return rdp.rdp(trace, r)


# RDP cache
#rdp_cache = memory.cache(compute_rdp)

def rdp_cache(idx, r):
    if (idx, r) not in cache:
        cache[(idx, r)] = compute_rdp(idx, r)
    return cache[(idx, r)]


def knee_cost(idx, r, cs, ct):
    trace = traces[idx]
    expected = expecteds[idx]

    # RDP
    points_reduced, removed = rdp_cache(idx, r)

    ## Knee detection code ##
    knees = curvature.multi_knee(points_reduced)
    if len(knees) == 0:
        return float('inf'), float('inf'), 0
    
    t_k = pp.filter_worst_knees(points_reduced, knees)
    t_k = pp.filter_corner_knees(points_reduced, t_k, cs)
    filtered_knees = pp.filter_clustring(points_reduced, t_k, clustering.average_linkage, ct, ClusterRanking.left)
    #knees = pp.add_points_even(trace, points_reduced, filtered_knees, removed, tx=0.05, ty=0.05)
    knees = rdp.mapping(filtered_knees, points_reduced, removed)
    if len(knees) == 0:
        return float('inf'), float('inf'), 0, -1

    ## RMSPE cost
    cost_a = evaluation.rmspe(trace, knees, expected, evaluation.Strategy.knees)
    cost_b = evaluation.rmspe(trace, knees, expected, evaluation.Strategy.expected)

    ## MCC
    cm = evaluation.cm(trace, knees, expected)
    mcc = evaluation.mcc(cm)

    #logger.error(f'Trace: {idx} -> ({r}, {cs}, {ct}) -> ({cost_a} {cost_b} {len(knees)} {mcc})')
    return cost_a, cost_b, len(knees), mcc


# Cost cache
#knee_cost_cache = memory.cache(knee_cost)
def knee_cost_cache(idx, r, cs, ct):
    if (idx, r, cs, ct) not in cache:
        cache[(idx, r, cs, ct)] = knee_cost(idx, r, cs, ct)
    return cache[(idx, r, cs, ct)]


def compute_knees_cost(r, cs, ct):
    costs = []
    nks = []
    try:
        for i in range(len(traces)):
            cost_a, cost_b, nk, mcc = knee_cost_cache(i, r, cs, ct)
            
            if args.m is Metric.mcc:
                cost = 1.0 - mcc
            elif args.m is Metric.rmspe:
                if args.a is Agglomeration.maximum:
                    cost = max(cost_a, cost_b)
                elif args.a is Agglomeration.average:
                    cost = (cost_a+cost_b)/2.0
            costs.append(cost)
            nks.append(nk)
        costs = np.array(costs)
    except Exception as e:
        logger.warning(f'Error: {e}')

    # add the cost of the number of knees
    if args.a is Agglomeration.maximum:
        return np.amax(costs) + max(max(nks)-args.k, 0)
    elif args.a is Agglomeration.average:
        return np.average(costs) + max(statistics.mean(nks)-args.k, 0)


def objective(p):
    # Round input parameters 
    r = round(p[0]*100.0)/100.0
    cs = round(p[1]*100.0)/100.0
    ct = round(p[2]*100.0)/100.0

    return compute_knees_cost(r, cs, ct)


def main(args):
    # Get all files from the input folder
    logger.info(f'Base path: {args.i}')
    files = []
    for f in os.listdir(args.i):
        if re.match(r'w[0-9]+-(arc|lru).csv', f):
            files.append(f)
    files.sort()

    # Get the traces 
    global traces, expecteds
    for f in files:
        path = os.path.join(os.path.normpath(args.i), f)
        points = np.genfromtxt(path, delimiter=',')
        traces.append(points)

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
    logger.info(f'Running the Differential Evolution Optimization ({args.p}, {args.l}, {args.m}, {args.a})')
    bounds = np.asarray([[.9, .95], [.1, 1], [.01, .1]])
    best, _, debug = de.differential_evolution(objective, bounds, variant=args.v, n_iter=args.l, n_pop=args.p, n_jobs=args.c, cached=False, debug=True)

    # Round input parameters
    r = round(best[0]*100.0)/100.0
    cs = round(best[1]*100.0)/100.0
    ct = round(best[2]*100.0)/100.0

    # Plot the optimization evolution
    pyplot.plot(debug[0], '.-', label='best')
    pyplot.plot(debug[1], '.-', label='average')
    pyplot.plot(debug[2], '.-', label='worst')
    pyplot.xlabel('Iteration')
    pyplot.ylabel('Cost Function')
    pyplot.legend()
    pyplot.xticks(range(0, args.l+1, 10))
    pyplot.savefig(args.g, bbox_inches='tight')
    pyplot.close()
    
    nk = []
    rmspe_knee = []
    rmspe_expected = []
    mccs = []

    # Compute the RMSPE for all the traces using the cache
    with open(args.o, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Files', 'RMSPE(k)', 'RMSPE(E)', 'MCC', 'N_Knees'])

        for i in range(len(traces)):
            cost_a, cost_b, n_knees, mcc = knee_cost_cache(i, r, cs, ct)
            writer.writerow([files[i], cost_a, cost_b, mcc, n_knees])
            nk.append(n_knees)
            rmspe_knee.append(cost_a)
            rmspe_expected.append(cost_b)
            mccs.append(mcc)

    # Output solution results
    logger.info(f'Best configuration ({r}, {cs}, {ct}) = [{statistics.mean(rmspe_knee)}, {statistics.mean(rmspe_expected)}, {statistics.mean(mccs)}]')

    # Output the number of knees
    logger.info(f'Average Number of knees ({statistics.median(nk)}, {statistics.mean(nk)}, {statistics.stdev(nk)})')

    # Clean cache
    #memory.clear(warn=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Z-Method Optimal Knee')
    parser.add_argument('-i', type=str, required=True, help='input folder')
    parser.add_argument('-o', type=str, help='output CSV', default='results_curvature.csv')
    parser.add_argument('-p', type=int, help='population size', default=50)
    parser.add_argument('-l', type=int, help='number of loops (iterations)', default=30)
    parser.add_argument('-c', type=int, help='number of cores', default=-1)
    parser.add_argument('-k', type=int, help='number of knees', default=10)
    parser.add_argument('-m', type=Metric, choices=list(Metric), help='Metric type', default='mcc')
    parser.add_argument('-a', type=Agglomeration, choices=list(Agglomeration), help='Agglomeration type', default='max')
    parser.add_argument('-g', type=str, help='output plot', default='plot_curvature.pdf')
    parser.add_argument('-t', type=float, help='CM threshold', default=0.01)
    parser.add_argument('-v', type=str, help='DE variant', default='best/1/bin')
    args = parser.parse_args()
    
    main(args)
