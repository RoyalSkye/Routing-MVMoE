from dataclasses import dataclass
from typing import Optional

import numpy as np
# import routefinder.baselines.pyvrp as pyvrp

from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from tensordict import TensorDict
from torch import Tensor
from pyvrp import Client, Depot, ProblemData, VehicleType, solve as _solve

import os, sys
import time
import argparse
from datetime import timedelta
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
from utils import check_extension, load_dataset, save_dataset, run_all_in_pool
import torch
from scipy.spatial.distance import cdist


# from .constants import ORTOOLS_SCALING_FACTOR
ORTOOLS_SCALING_FACTOR = 100_000
MAX_TW = 3.0

def solve(instance: TensorDict, max_runtime: float, **kwargs) -> tuple[Tensor, Tensor]:
    """
    Solves an MTVRP instance with OR-Tools.

    Args:
        instance: The MTVRP instance to solve.
        max_runtime: The maximum runtime for the solver.

    Returns:
        A tuple containing the action and the cost, respectively.

    Note:
        This function depends on PyVRP's data converter to convert the MTVRP
        instance to an OR-Tools compatible format. Future versions should
        implement a direct conversion.
    """
    data = instance2data(instance)
    action, cost = _solve(data, max_runtime)
    cost /= ORTOOLS_SCALING_FACTOR
    # cost *= -1

    return action, cost


@dataclass
class ORToolsData:
    """
    Convenient dataclass for instance data when using OR-Tools as solver.

    Args:
        depot: The depot index.
        distance_matrix: The distance matrix between locations.
        duration_matrix: The duration matrix between locations. This includes service times.
        num_vehicles: The number of vehicles.
        vehicle_capacities: The capacity of each vehicle.
        max_distance: The maximum distance a vehicle can travel.
        demands: The demands of each location.
        time_windows: The time windows for each location. Optional.
        backhauls: The pickup quantity for backhaul at each location.
    """

    depot: int
    distance_matrix: list[list[int]]
    duration_matrix: list[list[int]]
    num_vehicles: int
    vehicle_capacities: list[int]
    max_distance: int
    demands: list[int]
    time_windows: Optional[list[list[int]]]
    backhauls: Optional[list[int]]

    @property
    def num_locations(self) -> int:
        return len(self.distance_matrix)

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

def pyvrp_instance2data(instance: TensorDict, scaling_factor: int) -> ProblemData:
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
        # In VRPB, linehauls must be served before backhauls. This can be
        # enforced by setting a high value for the distance/duration from depot
        # to backhaul (forcing linehaul to be served first) and a large value
        # from backhaul to linehaul (avoiding linehaul after backhaul clients).
        # linehaul = np.flatnonzero(delivery > 0)
        # backhaul = np.flatnonzero(pickup > 0)
        # Note: we remove the constraint that we cannot visit backhauls *only* in a
        # a single route as per Slack discussion
        # matrix[0, backhaul] = MAX_VALUE
        # matrix[np.ix_(backhaul, linehaul)] = MAX_VALUE

    return ProblemData(clients, [depot], [vehicle_type], matrix, matrix)

def instance2data(instance: TensorDict) -> ORToolsData:
    """
    Converts an AnyVRP instance to an ORToolsData instance.
    """
    # TODO: Do not use PyVRP's data converter.
    data = pyvrp_instance2data(instance, ORTOOLS_SCALING_FACTOR)

    capacities = [
        veh_type.capacity
        for veh_type in data.vehicle_types()
        for _ in range(veh_type.num_available)
    ]
    max_distance = data.vehicle_type(0).max_distance

    demands = [0] + [client.delivery for client in data.clients()]
    backhauls = [0] + [client.pickup for client in data.clients()]
    service = [0] + [client.service_duration for client in data.clients()]

    tws = [[data.location(0).tw_early, data.location(0).tw_late]]
    tws += [[client.tw_early, client.tw_late] for client in data.clients()]

    # Set data to None if instance does not contain explicit values.
    default_tw = [0, np.iinfo(np.int64).max]
    if all(tw == default_tw for tw in tws):
        tws = None  # type: ignore

    if all(val == 0 for val in backhauls):
        backhauls = None  # type: ignore

    distances = data.distance_matrix().copy()
    durations = np.array(distances) + np.array(service)[:, np.newaxis]

    # if backhauls is not None:
    #     # Serve linehauls before backhauls.
    #     linehaul = np.flatnonzero(np.array(demands) > 0)
    #     backhaul = np.flatnonzero(np.array(backhauls) > 0)
    #     distances[np.ix_(backhaul, linehaul)] = max_distance

    return ORToolsData(
        depot=0,
        distance_matrix=distances.tolist(),
        duration_matrix=durations.tolist(),
        num_vehicles=data.num_vehicles,
        vehicle_capacities=capacities,
        demands=demands,
        time_windows=tws,
        max_distance=max_distance,
        backhauls=backhauls,
    )


def _solve(data: ORToolsData, max_runtime: float, log: bool = False):
    """
    Solves an instance with OR-Tools.

    Args:
        data: The instance data.
        max_runtime: The maximum runtime in seconds.
        log: Whether to log the search.

    Returns:
        A tuple containing the action and the cost, respectively.
    """
    # Manager for converting between nodes (location indices) and index
    # (internal CP variable indices).
    manager = pywrapcp.RoutingIndexManager(
        data.num_locations, data.num_vehicles, data.depot
    )
    routing = pywrapcp.RoutingModel(manager)

    # Set arc costs equal to distances.
    distance_transit_idx = routing.RegisterTransitMatrix(data.distance_matrix)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_transit_idx)

    # Max distance constraint.
    routing.AddDimension(
        distance_transit_idx,
        0,  # null distance slack
        data.max_distance,  # maximum distance per vehicle
        True,  # start cumul at zero
        "Distance",
    )

    # Vehicle capacity constraint.
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitVector(data.demands),
        0,  # null capacity slack
        data.vehicle_capacities,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Demand",
    )

    # Backhauls: this assumes that VRPB is implemented by forbidding arcs
    # that go from backhauls to linehauls.
    # if data.backhauls is not None:
    #     routing.AddDimensionWithVehicleCapacity(
    #         routing.RegisterUnaryTransitVector(data.backhauls),
    #         0,  # null capacity slack
    #         data.vehicle_capacities,  # vehicle maximum capacities
    #         True,  # start cumul to zero
    #         "Backhaul",
    #     )

    # Time window constraints.
    if data.time_windows is not None:
        depot_tw_early = data.time_windows[data.depot][0]
        depot_tw_late = data.time_windows[data.depot][1]

        # The depot's late time window is a valid upper bound for the waiting
        # time and maximum duration per vehicle.
        routing.AddDimension(
            routing.RegisterTransitMatrix(data.duration_matrix),
            depot_tw_late,  # waiting time upper bound
            depot_tw_late,  # maximum duration per vehicle
            False,  # don't force start cumul to zero
            "Time",
        )
        time_dim = routing.GetDimensionOrDie("Time")

        for node, (tw_early, tw_late) in enumerate(data.time_windows):
            if node == data.depot:  # skip depot
                continue

            index = manager.NodeToIndex(node)
            time_dim.CumulVar(index).SetRange(tw_early, tw_late)

        # Add time window constraints for each vehicle start node.
        for node in range(data.num_vehicles):
            start = routing.Start(node)
            time_dim.CumulVar(start).SetRange(depot_tw_early, depot_tw_late)

        for node in range(data.num_vehicles):
            cumul_start = time_dim.CumulVar(routing.Start(node))
            routing.AddVariableMinimizedByFinalizer(cumul_start)

            cumul_end = time_dim.CumulVar(routing.End(node))
            routing.AddVariableMinimizedByFinalizer(cumul_end)

    # Setup search parameters.
    params = pywrapcp.DefaultRoutingSearchParameters()

    gls = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.local_search_metaheuristic = gls

    params.time_limit.FromSeconds(int(max_runtime))  # only accepts int
    params.log_search = log

    solution = routing.SolveWithParameters(params)
    action = solution2action(data, manager, routing, solution)
    objective = solution.ObjectiveValue()

    return action, objective
def print_solution(data, manager, routing, assignment, problem="VRPB", log_file=None):
    """
        Only print route, and calculate cost (total distance).
    """

    def calc_vrp_cost(depot, loc, tour, problem):
        assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
        loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
        sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
        if problem in ["CVRP", "VRPB", "VRPL", "VRPTW", "VRPBL", "VRPLTW", "VRPBTW", "VRPBLTW"]:
            return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
        elif problem in ["OVRP", "OVRPB", "OVRPL", "OVRPTW", "OVRPBL", "OVRPLTW", "OVRPBTW", "OVRPBLTW"]:  # no need to return to depot
            full_tour = [0] + tour + [0]
            not_to_depot = np.array(full_tour)[1:] != 0
            return (np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1) * not_to_depot).sum()
        else:
            raise NotImplementedError

    route = []
    total_distance, total_load = 0, 0
    # distance_dimension = routing.GetDimensionOrDie('Distance')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    for vehicle_id in xrange(data['num_vehicles']):
        if not routing.IsVehicleUsed(vehicle=vehicle_id, assignment=assignment):
            continue
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        distance = 0
        while not routing.IsEnd(index):
            load_var = capacity_dimension.CumulVar(index)
            plan_output += ' {0} Load({1}) ->'.format(
                manager.IndexToNode(index),
                assignment.Value(load_var))
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            # distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)  # Bugs: always output 0 if given variable index, don't know why
            from_node, to_node = manager.IndexToNode(previous_index), manager.IndexToNode(index)
            to_node = to_node if to_node != data['dummy_depot'] else data['depot']
            distance += data['distance_matrix'][from_node][to_node]  # use distance matrix instead

        load_var = capacity_dimension.CumulVar(index)
        # dist_var = distance_dimension.CumulVar(index)
        plan_output += ' {0} Load({1})\n'.format(
            manager.IndexToNode(index),
            assignment.Value(load_var))
        # assert distance == assignment.Value(dist_var), ">> Distance not match!"
        plan_output += 'Distance of the route: {}\n'.format(distance)
        plan_output += 'Load of the route: {}\n'.format(assignment.Value(load_var))
        if log_file:
            print(plan_output, file=log_file)
        total_distance += distance
        total_load += assignment.Value(load_var)

    # double check
    cost = calc_vrp_cost(data['real_locations'][0], data['real_locations'][1:], route[1:], problem)
    if log_file:
        print('Route: {}'.format(route + [data['depot']]), file=log_file)
        print('Total Load of all routes: {}'.format(total_load), file=log_file)
        print('Total Distance of all routes: {} (Routing Error may exist)'.format(total_distance / SCALE), file=log_file)
        print('Final Result - Cost of the obtained solution: {}'.format(cost), file=log_file)

    return cost, route[1:]


def solution2action(data, manager, routing, solution) -> list[list[int]]:
    """
    Converts an OR-Tools solution to routes.
    """
    routes = []
    distance = 0  # for debugging

    for vehicle_idx in range(data.num_vehicles):
        index = routing.Start(vehicle_idx)
        route = []
        route_cost = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)

            prev_index = index
            index = solution.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(prev_index, index, vehicle_idx)

        if clients := route[1:]:  # ignore depot
            routes.append(clients)
            distance += route_cost

    return [visit for route in routes for visit in route + [0]]


def solve_ortools_log(directory, name, depot, loc, demand, capacity, route_limit=None, service_time=None, tw_start=None, tw_end=None,
                       timelimit=3600, grid_size=1, seed=1234, problem="CVRP", scaling_factor=1):
    """
        OR-Tools to solve VRP variants, Ref to:
            https://developers.google.com/optimization/routing/vrptw
            https://developers.google.com/optimization/routing/routing_options
            https://github.com/google/or-tools/issues/1051
            https://github.com/google/or-tools/issues/750
    """

    tour_filename = os.path.join(directory, "{}.ortoolsrl4co.tour".format(name))
    output_filename = os.path.join(directory, "{}.ortoolsrl4co.pkl".format(name))
    log_filename = os.path.join(directory, "{}.ortoolsrl4co.log".format(name))

    instance = TensorDict({
        "locs": np.concatenate((np.array(depot)[None, :], np.array(loc)), axis=0),
        "vehicle_capacity": np.array([capacity]),
        "distance_limit": np.array(route_limit) if route_limit is not None else np.array([float("inf")]),
        "service_time": np.concatenate((np.array([0]), np.array(service_time)), axis=0) if service_time is not None else np.zeros(len(loc)+1),
        "demand_linehaul": np.concatenate((np.array([0.0]), np.clip(np.array(demand), a_min=0, a_max=None))),
        "demand_backhaul": np.concatenate((np.array([0.0]), np.clip(-1 * np.array(demand), a_min=0, a_max=None))),
        "time_windows": np.concatenate((np.array([[0, MAX_TW]]), np.stack((np.array(tw_start), np.array(tw_end)), axis=-1)), axis=0) if tw_start is not None else \
        np.stack((np.zeros(len(loc)+1), np.ones(len(loc)+1) * float("inf")), axis=1),
        "cost_matrix": cdist(np.concatenate((np.array(depot)[None, :], np.array(loc)), axis=0),
                                   np.concatenate((np.array(depot)[None, :], np.array(loc)), axis=0)),

    })
    # print(instance.keys())
    # raise Exception
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
    route, cost = solve(instance, max_runtime=timelimit ,scaling_factor=ORTOOLS_SCALING_FACTOR)
    duration = time.time() - start
    # cost, route = print_solution(data, manager, routing, assignment, problem=problem, log_file=open(log_filename, 'w'))  # route does not include the first and last node (i.e., depot)
    # print("\n".join(["{}".format(r) for r in ([data['depot']] + route + [data['depot']])]), file=open(tour_filename, 'w'))
    save_dataset((route, duration), output_filename, disable_print=True)

    return cost, route, duration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ORTools RL4CO baseline")
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
            results_dir = os.path.join(opts.results_dir, "{}_ortools_rl4co".format(opts.problem))
            os.makedirs(results_dir, exist_ok=True)
            dir, filename = os.path.split(dataset_path)
            out_file = os.path.join(dir, "ortools_rl4co_{}s_{}".format(opts.timelimit, filename))
        else:
            results_dir = os.path.join(opts.results_dir, "{}_ortools_rl4co".format(opts.problem))
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

            return solve_ortools_log(
                directory, name,
                depot=depot, loc=loc, demand=demand, capacity=capacity, route_limit=route_limit, service_time=service_time, tw_start=tw_start, tw_end=tw_end,
                timelimit=opts.timelimit, grid_size=grid_size, seed=opts.seed, problem=opts.problem
            )

        target_dir = os.path.join(results_dir, "{}_ortools-rl4co_tl{}s".format(dataset_basename, opts.timelimit))
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
        print(">> Solving {} instances within {:.2f}s using OR Tools RL4CO".format(opts.n, time.time() - start_t))
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        results = [(i[0], i[1]) for i in results]
        save_dataset(results, out_file)  # [(obj, route), ...]

        os.system("rm -rf {}".format(target_dir))