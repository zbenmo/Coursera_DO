#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import functools
import itertools
import numpy as np
import networkx as nx

import utils

Point = namedtuple("Point", ['x', 'y'])


@functools.lru_cache
def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


def calc_solution_length(points, solution):
    nodeCount = len(points)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    return obj


def trivial_solution(node_count, points):
    G = nx.DiGraph()
    nx.add_cycle(G, range(node_count))

    return G


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # solution = trivial(points)
    opt = 0

    # solution = improve(points, solution)

    G = trivial_solution(node_count, points)

    solution = improve2(G, points)

    obj = calc_solution_length(points, solution)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def trivial(points):
    nodeCount = len(points)

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    return range(0, nodeCount)


def suggest_indices(max_ind):
    while True:
        inds = np.random.randint(low=0, high=max_ind, size=2)
        if inds[0] != inds[1]:
            yield inds


def improve(points, solution):
    current_solution = solution
    current_length = calc_solution_length(points, solution)
    # using the original solution here (for selecting potentials for swap)
    for _ in range(20):
        for potential_swap in itertools.islice(suggest_indices(len(solution) - 1), 2000):
            p1, p2 = potential_swap
            alternative_solution = [
                p2 if p == p1 else (
                    p1 if p == p2 else p
                )
                for p in current_solution
            ]
            length_of_alternative_solution = calc_solution_length(points, alternative_solution)
            if length_of_alternative_solution < current_length:
                current_solution = alternative_solution
                current_length = length_of_alternative_solution
    return current_solution


def rearrange_edges(G):
    start, end = next(iter(G.edges()))
    visited = set([start, end])
    while end != start:
        node = next(G.neighbors(end), None)
        if node != None:
            end = node
            visited.add(node)
        else:
            node = next((n for n in G.predecessors(end) if n not in visited), None)
            if node == None:
                node = start
            G.remove_edge(node, end)
            G.add_edge(end, node)
            end = node
            visited.add(node)


def length_by_nodes(node1, node2, points):
    return utils.length(points[node1], points[node2])


def improve2_helper(G, points):
    length_by_nodes_p = functools.partial(length_by_nodes, points=points)

    # pbar_text = "checking pairs of edges"
    # pbar = st.progress(0.0, text=pbar_text)

    # # there should be n*(n-1)/2 pairs
    # n = len(G.edges)
    # total = n * (n - 1) // 2

    # c = itertools.count(1)

    # cur_solution = utils.graph_to_solution(G)
    # cur_length = utils.calc_solution_length(cur_solution)
    G_temp = G.copy()
    for comb in itertools.islice(itertools.combinations(G.edges(), 2), None):
        edge1, edge2 = comb
        # c_value = next(c)
        # if c_value % 100 == 0:
        #     pbar.progress(c_value / total, text=pbar_text)
        if len(set(edge1).union(set(edge2))) < 4:
            continue
        if (
            length_by_nodes_p(edge1[0], edge2[0]) + length_by_nodes_p(edge1[1], edge2[1]) >
            length_by_nodes_p(*edge1) + length_by_nodes_p(*edge2)
            ):
            continue
        G_temp.remove_edges_from(comb)
        G_temp.add_edge(edge1[0], edge2[0])
        G_temp.add_edge(edge1[1], edge2[1])

        rearrange_edges(G_temp)
        return G_temp


def improve2(G, points):
    G_improved = improve2_helper(G, points)

    while G_improved:

        G_improved = improve2_helper(G_improved, points)

    return utils.graph_to_solution(G_improved) if G_improved else utils.graph_to_solution(G)



import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

