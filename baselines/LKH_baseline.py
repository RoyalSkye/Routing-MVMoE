import os, sys
import time
import argparse
import numpy as np
from subprocess import check_call
from urllib.parse import urlparse
from datetime import timedelta
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
from utils import check_extension, load_dataset, save_dataset, run_all_in_pool


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz"):
    """
        Note: the version of LKH matters.
        Based on the experiments, for CVRPTW:
            if version >= 3.0.7: the number of 'VEHICLES' needs to be specified clearly (tuned), otherwise, the output solutions vary a lot for different 'VEHICLES';
            if version <= 3.0.6, we could simply set 'VEHICLES' = len(loc) for CVRPTW.
        For DCVPR (VRPL):
            For all versions, the number of 'VEHICLES' needs to be specified clearly (tuned).
    """

    cwd = os.path.abspath("lkh")
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))

        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), "Download failed, {} does not exist".format(file)
        check_call(["tar", "xvfz", file], cwd=cwd)

        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make", cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable)
    return os.path.abspath(executable)


def solve_lkh_log(executable, directory, name, depot, loc, demand, capacity, route_limit=None, service_time=None, tw_start=None, tw_end=None,
                  runs=1, MAX_TRIALS=10000, grid_size=1, scale=100000, seed=1234, disable_cache=True, problem="CVRP"):

    problem_filename = os.path.join(directory, "{}.lkh{}.vrp".format(name, runs))
    tour_filename = os.path.join(directory, "{}.lkh{}.tour".format(name, runs))
    output_filename = os.path.join(directory, "{}.lkh{}.pkl".format(name, runs))
    param_filename = os.path.join(directory, "{}.lkh{}.par".format(name, runs))
    log_filename = os.path.join(directory, "{}.lkh{}.log".format(name, runs))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_vrplib(problem_filename, depot, loc, demand, capacity, route_limit=route_limit, service_time=service_time,
                         tw_start=tw_start, tw_end=tw_end, grid_size=grid_size, scale=scale, name=name, problem=problem)

            params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": tour_filename, "RUNS": runs, "SEED": seed, "MAX_TRIALS": MAX_TRIALS}
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_lkh_vrplib(tour_filename, n=len(demand))

            save_dataset((tour, duration), output_filename, disable_print=True)

        return calc_vrp_cost(depot, loc, tour, problem), tour, duration

    except Exception as e:
        raise
        print("Exception occured: {}".format(e))
        return None


def calc_vrp_cost(depot, loc, tour, problem):
    assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    if problem in ["CVRP", "VRPB", "CVRPTW", "DCVRP", "VRPBTW"]:
        return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
    elif problem in ["OVRP"]:  # no need to return to depot
        full_tour = [0] + tour + [0]
        not_to_depot = np.array(full_tour)[1:] != 0
        return (np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1) * not_to_depot).sum()
    else:
        raise NotImplementedError


def write_lkh_par(filename, parameters):
    default_parameters = {  # Use none to include as flag instead of kv
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_lkh_vrplib(filename, n):
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])

            if line.startswith("TOUR_SECTION"):
                started = True

    # print(tour, len(tour))
    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    # remove the first, last and redundant zeros
    # print(tour)
    final_tour, non_zero = [], True
    for e in tour:
        if non_zero or e != 0:
            final_tour.append(e)
        if e == 0:
            non_zero = False
        else:
            non_zero = True
    tour = np.array(final_tour).astype(int)

    if tour[0] != 0:
        tour = [0] + tour
    if tour[-1] == 0:
        tour = tour[:-1]
    # print(tour)

    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


def write_vrplib(filename, depot, loc, demand, capacity, route_limit=None, service_time=None, tw_start=None, tw_end=None,
                 grid_size=1, scale=100000, name="Instance", problem="CVRP"):

    # scale = 100000  # EAS uses 1000, while AM uses 100000
    to_int = lambda x: int(x / grid_size * scale + 0.5)
    size_vehicle_dict = {50: 8, 100: 12}  # hardcoded, only for DCVRP

    with open(filename, 'w') as f:
        # 1. file head
        # Note: 'VEHICLES' cannot >= 'DIMENSION'
        #   a. for CVRP, no need to specifiy VEHICLES
        #   b. for other problems, need to specifiy VEHICLES, otherwise, VEHICLES=1 -> cannot find feasible solutions
        #      for DCVRP, the performance heavily depend on the number of VEHICLES
        if problem in ["CVRP"]:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("COMMENT", "{} Instance".format(problem)),
                    ("TYPE", problem),
                    ("DIMENSION", len(loc) + 1),
                    ("CAPACITY", int(capacity)),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D")
                )
            ]))
        elif problem in ["OVRP", "VRPB"]:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("COMMENT", "{} Instance".format(problem)),
                    ("TYPE", problem),
                    ("DIMENSION", len(loc) + 1),
                    ("VEHICLES", len(loc)),
                    ("CAPACITY", int(capacity)),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D")
                )
            ]))
        elif problem in ["CVRPTW", "VRPBTW"]:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("COMMENT", "{} Instance".format(problem)),
                    ("TYPE", problem),
                    ("DIMENSION", len(loc) + 1),
                    ("VEHICLES", len(loc)),
                    ("CAPACITY", int(capacity)),
                    ("SERVICE_TIME", to_int(service_time[0])),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D")
                )
            ]))
        elif problem in ["DCVRP"]:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("COMMENT", "{} Instance".format(problem)),
                    ("TYPE", problem),
                    ("DIMENSION", len(loc) + 1),
                    ("CAPACITY", int(capacity)),
                    ("DISTANCE", to_int(route_limit)),
                    ("SERVICE_TIME", 0),
                    ("VEHICLES", size_vehicle_dict[len(loc)]),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D")
                )
            ]))
        else:
            raise NotImplementedError

        # 2. coordinates
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, to_int(x), to_int(y))  # VRPlib does not take floats
            for i, (x, y) in enumerate([depot] + loc)
        ]))

        # 3. demand
        f.write("\n")
        backhauls = [i+1 for i, d in enumerate(demand) if d < 0]
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, abs(int(d)))  # convert to int for lkh3, otherwise "DEMAND_SECTION: Node number out of range: 0"
            for i, d in enumerate([0] + demand)
        ]))

        # 4. optional: time window
        if problem in ["CVRPTW", "VRPBTW"]:
            f.write("\n")
            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, to_int(e), to_int(l))
                for i, (e, l) in enumerate(zip([0]+tw_start, [3]+tw_end))  # hardcoded: tw for depot: [0., 3.]
            ]))

        # 5. optional: backhauls
        if len(backhauls) > 0:
            f.write("\n")
            f.write("BACKHAUL_SECTION\n")
            f.write("\t".join(["{}".format(b) for b in backhauls]))
            f.write("\t-1")

        # 6. file tail
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LKH baseline, due to different problem settings, not recommend to use LKH3 to solve VRPB and VRPBTW")
    parser.add_argument('--problem', type=str, default="CVRP", choices=["CVRP", "OVRP", "VRPL", "VRPTW", "VRPB", "VRPBTW"])
    parser.add_argument("--datasets", nargs='+', default=["../data/CVRP/cvrp50_uniform.pkl", ], help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_false', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--disable_cache', action='store_false', help='Disable caching')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, default=1000, help="Number of instances to process")
    parser.add_argument('-runs', type=int, default=1, help="hyperparameters for LKH3")
    parser.add_argument('-max_trials', type=int, default=10000, help="hyperparameters for LKH3")
    parser.add_argument('-scale', type=int, default=100000, help="coefficient for float -> int")
    parser.add_argument('-seed', type=int, default=1234, help="random seed")
    parser.add_argument('--offset', type=int, default=0, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='baseline_results', help="Name of results directory")

    opts = parser.parse_args()
    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"
    problem_dict = {"CVRP": "CVRP", "OVRP": "OVRP", "VRPB": "VRPB", "VRPTW": "CVRPTW", "VRPL": "DCVRP", "VRPBTW": "VRPBTW"}
    opts.problem = problem_dict[opts.problem]
    if opts.problem in ["VRPB", "VRPBTW"]:
        print(">> Warnings: Due to different problem settings, not recommend to use LKH3 to solve VRPB and VRPBTW!")

    for dataset_path in opts.datasets:
        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"
        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "{}_lkh".format(opts.problem))
            os.makedirs(results_dir, exist_ok=True)
            dir, filename = os.path.split(dataset_path)
            out_file = os.path.join(dir, "lkh_{}".format(filename))
        else:
            out_file = opts.o
        assert opts.f or not os.path.isfile(out_file), "File already exists! Try running with -f option to overwrite."
        start_t = time.time()
        use_multiprocessing = True
        executable = get_lkh_executable()

        def run_func(args):
            directory, name, *args = args
            depot, loc, demand, capacity, route_limit, service_time, tw_start, tw_end = None, None, None, None, None, None, None, None
            if opts.problem in ["CVRP", "OVRP", "VRPB"]:
                depot, loc, demand, capacity, *args = args
            elif opts.problem in ["CVRPTW", "VRPBTW"]:
                depot, loc, demand, capacity, service_time, tw_start, tw_end, *args = args
            elif opts.problem in ["DCVRP"]:
                depot, loc, demand, capacity, route_limit, *args = args
            else:
                raise NotImplementedError

            depot = depot[0] if len(depot) == 1 else depot  # if depot: [[x, y]] -> [x, y]
            grid_size = 1
            if len(args) > 0:
                depot_types, customer_types, grid_size = args

            return solve_lkh_log(
                executable,
                directory, name,
                depot=depot, loc=loc, demand=demand, capacity=capacity, route_limit=route_limit, service_time=service_time, tw_start=tw_start, tw_end=tw_end,
                runs=opts.runs, MAX_TRIALS=opts.max_trials, grid_size=grid_size, scale=opts.scale, seed=opts.seed, disable_cache=opts.disable_cache, problem=opts.problem
            )

        target_dir = os.path.join(results_dir, "{}_lkh_run{}_trial{}".format(dataset_basename, opts.runs, opts.max_trials))
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
        print(">> Solving {} instances within {:.2f}s using LKH3 - {} Runs".format(opts.n, time.time()-start_t, opts.runs))
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        results = [(i[0], i[1]) for i in results]
        save_dataset(results, out_file)  # [(obj, route), ...]

        os.system("rm -rf {}".format(target_dir))
