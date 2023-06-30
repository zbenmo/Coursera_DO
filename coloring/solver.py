#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import Counter
from itertools import chain, islice
import itertools
import time
from typing import List, Tuple
import cpmpy
import numpy as np
from ortools.sat.python import cp_model
import networkx as nx

# from ortools.constraint_solver import pywrapcp


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # solution_num_colors, solution = solve_trivial(edges, node_count)
    # opt = "0"

    solution_num_colors, solution = solve_cp(edges, node_count)
    opt = "0"

    # solution_num_colors, solution = solve_cp_orttols(edges, node_count)
    # opt = "0"

    # prepare the solution in the specified output format
    output_data = str(solution_num_colors) + ' ' + opt + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


def solve_trivial(edges, node_count):
    # build a trivial solution
    # every node has its own color
    return node_count, range(0, node_count)


def print_statistics(g: nx.Graph):
    print(f'{g.number_of_nodes()=}')
    print(f'{g.number_of_edges()=}')


def solve_cp(edges, node_count):

    g = nx.Graph(edges)

    all_nodes = [x for x, _ in sorted(g.degree(), key=lambda ele: ele[1], reverse=True)]

    # all_cliques = sorted(nx.find_cliques(g), key=len, reverse=True)

    # first_clique = all_cliques[0]

    model = cpmpy.Model()

    # dict of node-names to intvar
    nodecolor = {
        node: cpmpy.intvar(1, i + 1, name=str(node))

        for i, node in enumerate(range(node_count))
    }

    # constrain adges to have differently colored nodes
    model += [nodecolor[node1] != nodecolor[node2] for node1, node2 in edges]

    # # Set the values for the nodes in the first clique.
    # # This shall never changed later in the solution
    # model += [
    #     nodecolor[node] == i + 1
    #     for i, node in enumerate(first_clique)
    # ]

    # model += [
    #     cpmpy.alldifferent(nodecolor[node] for node in clique)
    #     for clique in islice((x for x in all_cliques[1:] if len(x) > 20), None) #  if len(clique) > 6
    # ]

    # minimize number of colors
    model.minimize(cpmpy.max(nodecolor.values()))

    assert model.solve(), f"{node_count=}, {len(edges)=}"

    # print(model.objective_value())

    return model.objective_value(), [nodecolor[node].value() for node in range(node_count)]


class GraphColoringSolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, nodecolor):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__nodecolor = nodecolor
        # self.__solution_count = 0
        # self.__start_time = time.time()

    # def solution_count(self):
    #     return self.__solution_count

        self._solution = None

    def on_solution_callback(self):
        # current_time = time.time()
        # print('Solution %i, time = %f s' %
        #       (self.__solution_count, current_time - self.__start_time))
        # self.__solution_count += 1

        colors = [self.Value(self.__nodecolor[i]) for i in range(len(self.__nodecolor))]

        self._solution = max(colors) + 1, colors

        # all_queens = range(len(self.__queens))
        # for i in all_queens:
        #     for j in all_queens:
        #         if self.Value(self.__queens[j]) == i:
        #             # There is a queen in column j, row i.
        #             print('Q', end=' ')
        #         else:
        #             print('_', end=' ')
        #     print()
        # print()


def solve_cp_orttols(edges, node_count):

    g = nx.Graph(edges)

    # num_edges = Counter(chain.from_iterable(edges))

    # all_nodes = next(zip(*num_edges.most_common()))

    all_nodes = [x for x, _ in sorted(g.degree(), key=lambda ele: ele[1], reverse=True)]

    # all_triangles = list(nx.simple_cycles(g, 3))

    all_cliques = list(nx.find_cliques(g))

    ind_of_biggest_clique = next(iter(sorted(
        (ind for ind in range(len(all_cliques))),
        key=lambda index: len(all_cliques[index]),
        reverse=True
    )), None)
    size_of_biggest_clique = (
        len(all_cliques[ind_of_biggest_clique]) if ind_of_biggest_clique else 1
    )

    for max_colors in range(size_of_biggest_clique, node_count + 1):
        
        # print(f"{max_colors=}")

        model = cp_model.CpModel()

        # max_colors = node_count
        
        # dict of node-names to intvar
        nodecolor = {
            node: model.NewIntVar(0, max_colors - 1, f'color_for_{node}') # min(i, max_colors - 1)
            for i, node in enumerate(all_nodes)
        }

        if ind_of_biggest_clique:
            for i, node in enumerate(all_cliques[ind_of_biggest_clique]):
                model.Add(nodecolor[node] == i)
        
        # # constrain adges to have differently colored nodes
        # for node1, node2 in edges:
        #     model.Add(nodecolor[node1] != nodecolor[node2])
        
        # redundant all different contraints
        for clique in all_cliques:
            model.AddAllDifferent(nodecolor[x] for x in clique)

        model.AddAllDifferent


        # # minimize number of colors
        # model.minimize(max(nodecolor.values()))

        # print(model)

        # assert model.solve(time_limit=5)
        
        solver = cp_model.CpSolver()

        solution_printer = GraphColoringSolutionPrinter(nodecolor)

        # search_parameters = pywrapcp.Solver.DefaultSolverParameters()
        # search_parameters.trace_search  = True
        # search_parameters.trace_propagation  = True

        solver.Solve(model, solution_printer)

        # # Statistics.
        # print('\nStatistics')
        # print(f'  conflicts      : {solver.NumConflicts()}')
        # print(f'  branches       : {solver.NumBranches()}')
        # print(f'  wall time      : {solver.WallTime()} s')

        # print(model.objective_value())

        if solution_printer._solution is not None:

            return solution_printer._solution # model.objective_value(), [nodecolor[node].value() for node in range(node_count)]

    assert False

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

