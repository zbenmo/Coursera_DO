from collections import namedtuple
import functools
import math


Point = namedtuple("Point", ['x', 'y'])


def parse_input_data(input_data: str):
    # parse the input
    lines = input_data.split('\n')

    node_count = int(lines[0])

    points = []
    for i in range(1, node_count+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    assert len(points) == node_count, f'{len(points)} vs. {node_count}'

    return node_count, points


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


def graph_to_solution(G, show=None):
    solution = []
    node = 0
    while True:
        solution.append(node)
        neighbours = G[node]
        if len(neighbours) != 1:
            if show:
                show(neighbours) 
        node = next(iter(neighbours.keys()))
        if node == 0:
            break

    # if show:
    #     show(solution)

    assert len(solution) == len(G.nodes())

    return solution

