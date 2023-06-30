import functools
import itertools
import pathlib
import numpy as np
import streamlit as st
import networkx as nx
import utils
import matplotlib.pyplot as plt


@st.cache_data
def load_challenge(file_location):
    with open(file_location, 'r') as input_data_file:
        input_data = input_data_file.read()
    node_count, points = utils.parse_input_data(input_data)
    return node_count, points


def draw_graph(G, points):
    fig, ax = plt.subplots()
    pos = {
        i: point for i, point in enumerate(points)
    }
    nx.draw_networkx(G, pos=pos, node_size=30, font_size=10, ax=ax)
    st.pyplot(fig)


def trivial_solution(node_count, points):
    G = nx.DiGraph()
    nx.add_cycle(G, range(node_count))

    return G


@st.cache_data
def get_all_challenges():
    return sorted(
        pathlib.Path("data").glob("*"),
        key=lambda path: tuple(map(int, path.name.split("_")[1:]))
    ) 


def length_by_nodes(node1, node2, points):
    return utils.length(points[node1], points[node2])


def add_edge_and_then_remove_if_necessary(G, source, target):
    prev_predecessor = next(G.predecessors(target), None)
    neighbor = next(G.neighbors(target), None)
    G.add_edge(source, target)
    if prev_predecessor != None and neighbor != None:
        G.remove_edge(prev_predecessor, target)
    return prev_predecessor


def ensure(condition, err_msg):
    if not condition:
        st.error(err_msg)
        st.stop()


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
            ensure(node != None, f'{visited=}, {start=}, {end=}')
            G.remove_edge(node, end)
            G.add_edge(end, node)
            end = node
            visited.add(node)


def improve(G, points):
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

    # G_temp = G.copy()

    # edges_sorted = (
    #     sorted(G_temp.edges(), key=lambda edge: utils.length(points[edge[0]], points[edge[1]]))
    # )

    # edge1, edge2 = edges_sorted[-2:] # longest edge

    # st.write(edge1, edge2)

    # G_temp.remove_edge(*edge1)
    # G_temp.remove_edge(*edge2)

    # # p1, p2 = edge

    # # potential_points = set(range(len(points)))
    # # potential_points.remove(p1)
    # # potential_points.remove(p2)
    # # potential_points -= set(G_temp.predecessors(p1))

    # # other_points_sorted = sorted(
    # #     potential_points,
    # #     key=lambda p: utils.length(points[p1], points[p])
    # # )

    # # closest_point = other_points_sorted[0]

    # # prev_predecessor = add_edge_and_then_remove_if_necessary(G_temp, p1, closest_point)

    # # if prev_predecessor is not None:
    # #     prev_predecessor = add_edge_and_then_remove_if_necessary(G_temp, prev_predecessor, p2)


    # return G_temp


def write_solution_to_screen(solution):
    st.write(', '.join(map(str, solution)))


def main():
    # import sys
    # if len(sys.argv) < 2:
    #     print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
    #     exit(-1)

    # file_location = sys.argv[1].strip()

    file_location = st.selectbox("challenge", get_all_challenges())

    node_count, points = load_challenge(file_location)

    G = trivial_solution(node_count, points)

    draw_graph(G, points)

    trivial = utils.graph_to_solution(G)

    write_solution_to_screen(trivial)

    st.write(f'lenght = {utils.calc_solution_length(points, trivial)}')

    G_improved = improve(G, points)

    while G_improved:

        draw_graph(G_improved, points)

        improved_tsp = utils.graph_to_solution(G_improved, write_solution_to_screen)

        write_solution_to_screen(improved_tsp)

        st.write(f'lenght improved = {utils.calc_solution_length(points, improved_tsp)}')

        G_improved = improve(G_improved, points)


if __name__ == "__main__":
    main()