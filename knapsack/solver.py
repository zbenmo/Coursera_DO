#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
import pandas as pd
from pprint import pprint
from utils import PriorityQueue


Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # method = "dp"

    if len(items) * capacity < 10_000_000:
        method = "dp"
    # elif capacity > 20_000:
    #     method = "trivial"
    else:
        method = "bab"

    if method == "trivial":
        value, taken = trivial(items, capacity)
        opt = 0
    elif method == "dp":
        value, taken = dynamic_programming(items, capacity)
        opt = 1
    elif method == "bab":
        value_trivial, taken_trivial = trivial(items, capacity)

        value, taken = branch_and_bound(items, capacity, value_trivial, taken_trivial)
        opt = 1
    else:
        assert False, f'{method=}'

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(opt) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


def trivial(items, capacity):
    # a trivial algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, taken


def dynamic_programming(items, capacity):
    num_items = len(items)
    # in the table below we'll keep the best value we have for that combination of (capacity, last item index)
    table = np.empty((capacity, num_items + 1), dtype=int)
    table[:, 0] = 0 # with no items, the best value is 0, regardelss the capacity

    def get_value_in_cell(capacity, item_index):
        return 0 if capacity < 1 else table[capacity - 1, item_index + 1]

    def set_value_in_cell(capacity, item_index, value):
        table[capacity - 1, item_index + 1] = value

    for item_index, item in enumerate(items):
        for cur_capacity in range(1, capacity + 1):
            value_without_item = get_value_in_cell(cur_capacity, item_index - 1)
            if cur_capacity < item.weight:
                # Don't take current item
                value_found = value_without_item
            else:
                value_if_taking = get_value_in_cell(cur_capacity - item.weight, item_index - 1) + item.value
                value_found = max(
                    value_without_item,
                    value_if_taking
                )
            # assert value_found < 100
            set_value_in_cell(cur_capacity, item_index, value_found)

    taken = [0]*len(items)
    cur_capacity = capacity
    for item_index, item in reversed(list(enumerate(items))):
        if get_value_in_cell(cur_capacity, item_index) > get_value_in_cell(cur_capacity, item_index - 1):
            taken[item_index] = 1
            cur_capacity -= item.weight

    return get_value_in_cell(capacity, num_items - 1), taken


def calc_linear_relaxation(df, capacity):
    """
    Returns:
        the value
    """
    # return (
    #     df
    #     .assign(
    #         cum_weight=lambda d: d.weight.cumsum(),
    #         cum_weight_before=lambda d: d.cum_weight - d.weight, 
    #         take=lambda d: np.where(
    #             d.cum_weight <= capacity,
    #             1,
    #             np.where(d.cum_weight_before < capacity, (capacity - d.cum_weight_before) / d.weight, 0) 
    #         ),
    #         contrib_value=lambda d: d['take'] * d.value,
    #     )
    #     .contrib_value
    #     .sum()
    # )

    cur_capacity = 0
    value = 0
    for _, item in df.iterrows():
        if cur_capacity >= item.weight:
            value += item.value
            cur_capacity -= item.weight
        else:
            value += item.value * cur_capacity / item.weight
            break
    return value

def branch_and_bound(items, capacity, value_trivial, taken_trivial): # Best First
    df = (
        pd.DataFrame.from_records(
            np.array(items, dtype=float)[:, 1:],
            columns=['value', 'weight'],
        )
        .assign(
            value_per_unit_of_weight=lambda d: d['value'] / d['weight']
        )
        .sort_values(by='value_per_unit_of_weight', ascending=False)
    )

    check_options = PriorityQueue()

    check_options.push(([], 0, capacity, 0, calc_linear_relaxation(df, capacity)))

    best_solution = taken_trivial
    best_found = value_trivial

    def consider(option):
        nonlocal best_solution
        nonlocal best_found

        _, next_index, _, value, _ = option
        if next_index == len(items):
            # if best_solution == None:
            #     best_solution = option # the first solution
            #     best_found = value
            # else:
                if best_found < value:
                    best_solution = option[0] # a better solution
                    best_found = value
        else:
            check_options.push(option)

    while not check_options.empty():
        cur_evaluation = check_options.pop()
        cur_taken, cur_next_index, cur_capacity, cur_value, optimistic_value = cur_evaluation

        cur_item = items[cur_next_index]

        new_next_index = cur_next_index + 1

        # introduce the option when we do not take the item
        new_optimistic_value = cur_value + calc_linear_relaxation(df.iloc[new_next_index:], cur_capacity)
        if new_optimistic_value > best_found:
            consider((
                cur_taken[:] + [0],
                new_next_index,
                cur_capacity,
                cur_value,
                new_optimistic_value,
            ))

        if cur_capacity >= cur_item.weight:
            # introduce the option when we take the item
            new_value = cur_value + cur_item.value
            new_capacity = cur_capacity - cur_item.weight
            new_optimistic_value = optimistic_value
            if new_optimistic_value > best_found:
                consider((
                    cur_taken[:] + [1],
                    new_next_index,
                    new_capacity,
                    new_value,
                    new_optimistic_value,
                ))

    return best_found,  best_solution


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

