import argparse
import os, sys
import numpy as np
from subprocess import check_call, check_output
from urllib.parse import urlparse
import time
from datetime import timedelta
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
from utils import check_extension, load_dataset, save_dataset, run_all_in_pool


def get_hgs_cvrp_executable(url="https://github.com/vidalt/HGS-CVRP.git"):

    cwd = os.path.abspath('hgs')
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]
    cwd_build = os.path.join(filedir, "build")

    if not os.path.isdir(filedir):
        print("{} not found, downloading and compiling".format(filedir))
        check_call(f"git clone {url}", cwd=cwd, shell=True)
        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        os.makedirs(cwd_build, exist_ok=True)
        check_call("cmake .. -DCMAKE_BUILD_TYPE=Release", cwd=cwd_build, shell=True)
        check_call("make bin", cwd=cwd_build, shell=True)

    executable = os.path.join(cwd_build, "hgs")
    assert os.path.isfile(executable), f'Cannot find HGS executable file at {executable}'
    return os.path.abspath(executable)


def get_hgs_vrptw_executable(url="https://github.com/ortec/euro-neurips-vrp-2022-quickstart.git"):

    cwd = os.path.abspath('hgs')
    os.makedirs(cwd, exist_ok=True)

    file = os.path.join(cwd, os.path.split(urlparse(url).path)[-1])
    filedir = os.path.splitext(file)[0]
    cwd_build = os.path.join(filedir, "baselines/hgs_vrptw")
    mv_path = os.path.join(cwd, "HGS-VRPTW")

    if not os.path.isdir(mv_path):
        print("{} not found, downloading and compiling".format(filedir))
        check_call(f"git clone {url}", cwd=cwd, shell=True)
        assert os.path.isdir(filedir), "Extracting failed, dir {} does not exist".format(filedir)
        check_call("make all", cwd=cwd_build, shell=True)
        os.system("mv {} {}".format(cwd_build, mv_path))
        os.system("rm -rf {}".format(filedir))

    executable = os.path.join(mv_path, "genvrp")
    assert os.path.isfile(executable), f'Cannot find HGS executable file at {executable}'
    return os.path.abspath(executable)


def solve_hgs_log(executable, directory, name, depot, loc, demand, capacity, service_time=None, tw_start=None, tw_end=None,
                  max_iteration=20000, grid_size=1, scale=100000, seed=1234, disable_cache=True, problem="CVRP"):

    problem_filename = os.path.join(directory, "{}.hgs.vrp".format(name))
    tour_filename = os.path.join(directory, "{}.hgs.tour".format(name))
    output_filename = os.path.join(directory, "{}.hgs.pkl".format(name))
    log_filename = os.path.join(directory, "{}.hgs.log".format(name))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_vrplib(problem_filename, depot, loc, demand, capacity, service_time=service_time, tw_start=tw_start, tw_end=tw_end,
                         grid_size=grid_size, scale=scale, name=name, problem=problem)

            with open(log_filename, 'w') as f:
                start = time.time()
                # we call hgs with its default setting (i.e., -it=20000)
                if problem == "CVRP":
                    check_call("{} {} {} -it {} -seed {} -round {}".format(executable, problem_filename, tour_filename, max_iteration, seed, 0), shell=True, stdout=f, stderr=f)
                elif problem == "VRPTW":
                    check_call("{} {} {} -it {} -seed {}".format(executable, problem_filename, tour_filename, max_iteration, seed), shell=True, stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_hgs_vrplib(tour_filename, n=len(demand))

            save_dataset((tour, duration), output_filename, disable_print=True)

        return calc_vrp_cost(depot, loc, tour, problem), tour, duration

    except Exception as e:
        raise
        print("Exception occured")
        print(e)
        return None


def calc_vrp_cost(depot, loc, tour, problem):
    assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    if problem in ["CVRP", "VRPTW"]:
        return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
    else:
        raise NotImplementedError


def read_hgs_vrplib(filename, n):
    tour = []
    num_routes = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("Route"):
                l = line.strip().split(":")
                tour.append(0)
                tour.extend(map(int, l[1].split()))
                num_routes += 1

    tour = np.array(tour).astype(int)  # depot is 0 and other nodes start with 1 in HGS format
    assert len(tour) - num_routes == np.max(tour) == n

    return tour[1:].tolist()


def write_vrplib(filename, depot, loc, demand, capacity, service_time=None, tw_start=None, tw_end=None,
                 grid_size=1, scale=10000, name="Instance", problem="CVRP"):

    # scale = 10000  # EAS uses 1000, while AM uses 100000 (too large for HGS)
    if scale != 1:
        to_int = lambda x: int(x / grid_size * scale + 0.5)
    else:
        to_int = lambda x: x

    with open(filename, 'w') as f:
        # 1. file head
        # Note: 'VEHICLES' cannot >= 'DIMENSION', simply set VEHICLES = len(loc) is fine in HGS.
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
        elif problem in ["VRPTW"]:
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
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, int(d))
            for i, d in enumerate([0] + demand)
        ]))

        # 4. optional: time window
        if problem in ["VRPTW"]:
            f.write("\n")
            f.write("TIME_WINDOW_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, to_int(e), to_int(l))
                for i, (e, l) in enumerate(zip([0]+tw_start, [3]+tw_end))  # hardcoded: tw for depot: [0., 3.]
            ]))

        # 6. file tail
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="HGS baseline")
    parser.add_argument('--problem', type=str, default="CVRP", choices=["CVRP", "VRPTW"])
    parser.add_argument("--datasets", nargs='+', default=["../data/CVRP/cvrp50_uniform.pkl", ], help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_false', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--disable_cache', action='store_false', help='Disable caching')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, default=1000, help="Number of instances to process")
    parser.add_argument('-max_iteration', type=int, default=20000, help="hyperparameters for HGS")
    parser.add_argument('-scale', type=int, default=10000, help="coefficient for float -> int")
    parser.add_argument('-seed', type=int, default=1234, help="random seed")
    parser.add_argument('--offset', type=int, default=0, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='baseline_results', help="Name of results directory")

    opts = parser.parse_args()
    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:
        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"
        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "{}_hgs".format(opts.problem))
            os.makedirs(results_dir, exist_ok=True)
            dir, filename = os.path.split(dataset_path)
            out_file = os.path.join(dir, "hgs_{}".format(filename))
        else:
            out_file = opts.o
        assert opts.f or not os.path.isfile(out_file), "File already exists! Try running with -f option to overwrite."
        start_t = time.time()
        use_multiprocessing = True
        executable = get_hgs_cvrp_executable() if opts.problem == "CVRP" else get_hgs_vrptw_executable()

        def run_func(args):
            directory, name, *args = args
            depot, loc, demand, capacity, service_time, tw_start, tw_end = None, None, None, None, None, None, None
            if opts.problem in ["CVRP"]:
                depot, loc, demand, capacity, *args = args
            elif opts.problem in ["VRPTW"]:
                depot, loc, demand, capacity, service_time, tw_start, tw_end, *args = args
            else:
                raise NotImplementedError

            depot = depot[0] if len(depot) == 1 else depot  # if depot: [[x, y]] -> [x, y]
            grid_size = 1
            if len(args) > 0:
                depot_types, customer_types, grid_size = args

            return solve_hgs_log(
                executable,
                directory, name,
                depot=depot, loc=loc, demand=demand, capacity=capacity, service_time=service_time, tw_start=tw_start, tw_end=tw_end,
                max_iteration=opts.max_iteration, grid_size=grid_size, scale=opts.scale, seed=opts.seed, disable_cache=opts.disable_cache, problem=opts.problem
            )

        target_dir = os.path.join(results_dir, "{}_hgs_iter{}".format(dataset_basename, opts.max_iteration))
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
        print(">> Solving {} instances within {:.2f}s using HGS".format(opts.n, time.time()-start_t))
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        results = [(i[0], i[1]) for i in results]
        save_dataset(results, out_file)  # [(obj, route), ...]

        os.system("rm -rf {}".format(target_dir))
