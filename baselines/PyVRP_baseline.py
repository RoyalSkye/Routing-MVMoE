"""
File taken from https://github.com/ai4co/rl4co/blob/main/rl4co/envs/routing/mtvrp/baselines/pyvrp.py
"""

import numpy as np
import pyvrp as pyvrp

from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve
from pyvrp.constants import MAX_VALUE
from pyvrp.stop import MaxRuntime
from tensordict.tensordict import TensorDict
from torch import Tensor

# from .constants import PYVRP_SCALING_FACTOR
# from .utils import scale

import os, sys
import time
import argparse
from datetime import timedelta
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
from utils import check_extension, load_dataset, save_dataset, run_all_in_pool
import torch
from scipy.spatial.distance import cdist

PYVRP_SCALING_FACTOR = 1_000
MAX_TW = 3.0

def scale(data: Tensor, scaling_factor: int):
    """
    Scales ands rounds data to integers so PyVRP can handle it.
    """
    array = (data * scaling_factor).numpy().round()
    array = np.where(array == np.inf, np.iinfo(np.int32).max, array)
    array = array.astype(int)

    if array.size == 1:
        return array.item()

    return array

def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves the AnyVRP instance with PyVRP.

    Args:
        instance: The AnyVRP instance to solve.
        max_runtime: The maximum runtime for the solver.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    data = instance2data(instance, PYVRP_SCALING_FACTOR)
    stop = MaxRuntime(max_runtime)
    result = _solve(data, stop)

    solution = result.best
    action = solution2action(solution)
    cost = result.cost() / PYVRP_SCALING_FACTOR

    return action, cost


def instance2data(instance: TensorDict, scaling_factor: int) -> ProblemData:
    """
    Converts an AnyVRP instance to a ProblemData instance.

    Args:
        instance: The AnyVRP instance to convert.
        scaling_factor: The scaling factor to use for the conversion.

    Returns:
        The ProblemData instance.
    """
    num_locs = instance["locs"].size()[0]

    time_windows = scale(instance["time_windows"], scaling_factor)
    pickup = scale(instance["demand_backhaul"], scaling_factor)
    delivery = scale(instance["demand_linehaul"], scaling_factor)
    service = scale(instance["service_time"], scaling_factor)
    coords = scale(instance["locs"], scaling_factor)
    capacity = scale(instance["vehicle_capacity"], scaling_factor)
    max_distance = scale(instance["distance_limit"], scaling_factor)

    depot = Depot(
        x=coords[0][0],
        y=coords[0][1],
        tw_early=time_windows[0][0],
        tw_late=time_windows[0][1],
    )

    clients = [
        Client(
            x=coords[idx][0],
            y=coords[idx][1],
            tw_early=time_windows[idx][0],
            tw_late=time_windows[idx][1],
            delivery=delivery[idx],
            pickup=pickup[idx],
            service_duration=service[idx],
        )
        for idx in range(1, num_locs)
    ]

    vehicle_type = VehicleType(
        num_available=num_locs - 1,  # one vehicle per client
        # num_available=1,  # one vehicle per client
        capacity=capacity,
        max_distance=max_distance,
    )

    matrix = scale(instance["cost_matrix"], scaling_factor)

    if instance["open_route"]:
        # Vehicles do not need to return to the depot, so we set all arcs
        # to the depot to zero.
        matrix[:, 0] = 0

    # if instance["backhaul_class"] == 1:  # VRP with backhauls
    #     # In VRPB, linehauls must be served before backhauls. This can be
    #     # enforced by setting a high value for the distance/duration from depot
    #     # to backhaul (forcing linehaul to be served first) and a large value
    #     # from backhaul to linehaul (avoiding linehaul after backhaul clients).
    #     linehaul = np.flatnonzero(delivery > 0)
    #     backhaul = np.flatnonzero(pickup > 0)
    #     # Note: we remove the constraint that we cannot visit backhauls *only* in a
    #     # a single route as per Slack discussion
    #     # matrix[0, backhaul] = MAX_VALUE
    #     matrix[np.ix_(backhaul, linehaul)] = MAX_VALUE

    return ProblemData(clients, [depot], [vehicle_type], matrix, matrix)


def solution2action(solution: pyvrp.Solution) -> list[int]:
    """
    Converts a PyVRP solution to the action representation, i.e., a giant tour.
    """
    return [visit for route in solution.routes() for visit in route.visits() + [0]]

def process_instance(td: TensorDict) -> TensorDict:
    """
    We simply transform the data to the format the current PyVRP API expects
    """
    td_ = td
    td_.set("durations", td["service_time"])
    cost_mat = td_['cost_matrix']
    num_loc = cost_mat.shape[-1]
    # note: if we don't do this, PyVRP may complain diagonal is not 0.
    # i guess it is because of some conversion from floating point to integer
    cost_mat[np.arange(num_loc), np.arange(num_loc)] = 0
    td_.set("cost_matrix", cost_mat)
    return td_


def solve_pyvrp_log(directory, name, depot, loc, demand, capacity, route_limit=None, service_time=None, tw_start=None, tw_end=None,
                       timelimit=3600, grid_size=1, seed=1234, problem="CVRP", scaling_factor=1):
    """
        OR-Tools to solve VRP variants, Ref to:
            https://developers.google.com/optimization/routing/vrptw
            https://developers.google.com/optimization/routing/routing_options
            https://github.com/google/or-tools/issues/1051
            https://github.com/google/or-tools/issues/750
    """

    tour_filename = os.path.join(directory, "{}.pyvrp.tour".format(name))
    output_filename = os.path.join(directory, "{}.pyvrp.pkl".format(name))
    log_filename = os.path.join(directory, "{}.pyvrp.log".format(name))

    instance = TensorDict({
        "locs": np.concatenate((np.array(depot)[None, :], np.array(loc)), axis=0),
        "vehicle_capacity": np.array(capacity),
        "distance_limit": np.array(route_limit) if route_limit is not None else np.array(float("inf")),
        "service_time": np.concatenate((np.array([0]), np.array(service_time)), axis=0) if service_time is not None else np.zeros(len(loc)+1),
        "demand_linehaul": np.concatenate((np.array([0.0]), np.clip(np.array(demand), a_min=0, a_max=None))),
        "demand_backhaul": np.concatenate((np.array([0.0]), np.clip(-1 * np.array(demand), a_min=0, a_max=None))),
        "time_windows": np.concatenate((np.array([[0, MAX_TW]]), np.stack((np.array(tw_start), np.array(tw_end)), axis=-1)), axis=0) if tw_start is not None else \
        np.stack((np.zeros(len(loc)+1), np.ones(len(loc)+1) * float("inf")), axis=1),
        "cost_matrix": cdist(np.concatenate((np.array(depot)[None, :], np.array(loc)), axis=0),
                                   np.concatenate((np.array(depot)[None, :], np.array(loc)), axis=0)),

    })
    if problem in ["OVRP", "OVRPB", "OVRPL", "OVRPTW", "OVRPBL", "OVRPLTW", "OVRPBTW", "OVRPBLTW"]:
        instance["open_route"] = True
    else:
        instance["open_route"] = False

    if problem in ["VRPB", "OVRPB", "VRPBL", "VRPBTW", "VRPBLTW", "OVRPBL", "OVRPBTW", "OVRPBLTW"]:
        instance["backhaul_class"] = 1
    else:
        instance["backhaul_class"] = 0
    instance = process_instance(instance)
    start = time.time()
    route, cost = solve(instance, max_runtime=timelimit ,scaling_factor=PYVRP_SCALING_FACTOR)
    duration = time.time() - start
    # cost, route = print_solution(data, manager, routing, assignment, problem=problem, log_file=open(log_filename, 'w'))  # route does not include the first and last node (i.e., depot)
    # print("\n".join(["{}".format(r) for r in ([data['depot']] + route + [data['depot']])]), file=open(tour_filename, 'w'))
    save_dataset((route, duration), output_filename, disable_print=True)

    return cost, route, duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyVRP baseline")
    parser.add_argument('--problem', type=str, default="CVRP", choices=["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                                                        "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                                                        "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument("--datasets", nargs='+', default=["../data/CVRP/cvrp50_uniform.pkl", ], help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_false', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, default=1000, help="Number of instances to process")
    parser.add_argument('-timelimit', type=int, default=20, help="timelimit (seconds) for OR-Tools to solve an instance")
    parser.add_argument('-seed', type=int, default=1234, help="random seed")
    parser.add_argument('--offset', type=int, default=0, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='baseline_results', help="Name of results directory")

    opts = parser.parse_args()
    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:
        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"
        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "{}_pyvrp".format(opts.problem))
            os.makedirs(results_dir, exist_ok=True)
            dir, filename = os.path.split(dataset_path)
            out_file = os.path.join(dir, "pyvrp_{}s_{}".format(opts.timelimit, filename))
        else:
            results_dir = os.path.join(opts.results_dir, "{}_pyvrp".format(opts.problem))
            os.makedirs(results_dir, exist_ok=True)
            out_file = opts.o
        assert opts.f or not os.path.isfile(out_file), "File already exists! Try running with -f option to overwrite."
        start_t = time.time()
        use_multiprocessing = True

        def run_func(args):
            directory, name, *args = args
            depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end = None, None, None, None, None, None, None, None
            if opts.problem in ["CVRP", "OVRP", "VRPB", "OVRPB"]:
                depot, loc, demand, capacity, *args = args
            elif opts.problem in ["VRPTW", "OVRPTW", "VRPBTW", "OVRPBTW"]:
                depot, loc, demand, capacity, service_time, tw_start, tw_end, *args = args
            elif opts.problem in ["VRPL", "VRPBL", "OVRPL", "OVRPBL"]:
                depot, loc, demand, capacity, route_limit, *args = args
            elif opts.problem in ["VRPLTW", "VRPBLTW", "OVRPLTW", "OVRPBLTW"]:
                depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end, *args = args
            else:
                raise NotImplementedError

            depot = depot[0] if len(depot) == 1 else depot  # if depot: [[x, y]] -> [x, y]
            grid_size = 1

            return solve_pyvrp_log(
                directory, name,
                depot=depot, loc=loc, demand=demand, capacity=capacity, route_limit=route_limit, service_time=service_time, tw_start=tw_start, tw_end=tw_end,
                timelimit=opts.timelimit, grid_size=grid_size, seed=opts.seed, problem=opts.problem
            )

        target_dir = os.path.join(results_dir, "{}_pyvrp_tl{}s".format(dataset_basename, opts.timelimit))
        print(">> Target dir: {}".format(target_dir))
        assert opts.f or not os.path.isdir(target_dir), "Target dir already exists! Try running with -f option to overwrite."
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        dataset = load_dataset(dataset_path)
        # Note: only processing n items is handled by run_all_in_pool
        results, parallelism = run_all_in_pool(
            run_func,
            target_dir, dataset, opts, use_multiprocessing=use_multiprocessing
        )

        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        print(">> Solving {} instances within {:.2f}s using PyVRP".format(opts.n, time.time() - start_t))
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        results = [(i[0], i[1]) for i in results]
        save_dataset(results, out_file)  # [(obj, route), ...]

        os.system("rm -rf {}".format(target_dir))