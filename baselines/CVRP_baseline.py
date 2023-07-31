import argparse
import os, sys
import numpy as np
import re
from subprocess import check_call, check_output
from urllib.parse import urlparse
import tempfile
import time
from datetime import timedelta
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
from utils import check_extension, load_dataset, save_dataset, run_all_in_pool, move_to


def get_lkh_executable(url="http://www.akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.8.tgz"):

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


def get_hgs_executable(url="https://github.com/vidalt/HGS-CVRP.git"):

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


def solve_lkh(executable, depot, loc, demand, capacity):

    with tempfile.TemporaryDirectory() as tempdir:
        problem_filename = os.path.join(tempdir, "problem.vrp")
        output_filename = os.path.join(tempdir, "output.tour")
        param_filename = os.path.join(tempdir, "params.par")

        starttime = time.time()
        write_vrplib(problem_filename, depot, loc, demand, capacity, method="lkh")
        params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": output_filename}
        write_lkh_par(param_filename, params)
        output = check_output([executable, param_filename])
        result = read_lkh_vrplib(output_filename, n=len(demand))
        duration = time.time() - starttime
        return result, output, duration


def solve_lkh_log(executable, directory, name, depot, loc, demand, capacity, grid_size=1, runs=1, disable_cache=False, MAX_TRIALS=10000):

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
            write_vrplib(problem_filename, depot, loc, demand, capacity, grid_size, name=name, method="lkh")

            params = {"PROBLEM_FILE": problem_filename, "OUTPUT_TOUR_FILE": tour_filename, "RUNS": runs, "SEED": 1234, "MAX_TRIALS": MAX_TRIALS}
            write_lkh_par(param_filename, params)

            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([executable, param_filename], stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_lkh_vrplib(tour_filename, n=len(demand))

            save_dataset((tour, duration), output_filename)

        return calc_vrp_cost(depot, loc, tour), tour, duration

    except Exception as e:
        raise
        print("Exception occured")
        print(e)
        return None


def solve_hgs_log(executable, directory, name, depot, loc, demand, capacity, grid_size=1, runs=1, disable_cache=False):

    problem_filename = os.path.join(directory, "{}.hgs{}.vrp".format(name, runs))
    tour_filename = os.path.join(directory, "{}.hgs{}.tour".format(name, runs))
    output_filename = os.path.join(directory, "{}.hgs{}.pkl".format(name, runs))
    # param_filename = os.path.join(directory, "{}.hgs{}.par".format(name, runs))
    log_filename = os.path.join(directory, "{}.hgs{}.log".format(name, runs))

    try:
        # May have already been run
        if os.path.isfile(output_filename) and not disable_cache:
            tour, duration = load_dataset(output_filename)
        else:
            write_vrplib(problem_filename, depot, loc, demand, capacity, grid_size, name=name, method="hgs")

            with open(log_filename, 'w') as f:
                start = time.time()
                # we call hgs with its default setting (i.e., -it=20000)
                check_call("{} {} {} -it {} -seed {} -round {}".format(executable, problem_filename, tour_filename, 20000, 1234, 0), shell=True, stdout=f, stderr=f)
                duration = time.time() - start

            tour = read_hgs_vrplib(tour_filename, n=len(demand))

            save_dataset((tour, duration), output_filename)

        return calc_vrp_cost(depot, loc, tour), tour, duration

    except Exception as e:
        raise
        print("Exception occured")
        print(e)
        return None


def calc_vrp_cost(depot, loc, tour):
    assert (np.sort(tour)[-len(loc):] == np.arange(len(loc)) + 1).all(), "All nodes must be visited once!"
    # TODO validate capacity constraints
    loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
    sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


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

    assert len(tour) == dimension
    tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
    tour[tour > n] = 0  # Any nodes above the number of nodes there are is also depot
    assert tour[0] == 0  # Tour should start with depot
    assert tour[-1] != 0  # Tour should not end with depot
    return tour[1:].tolist()


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


def write_vrplib(filename, depot, loc, demand, capacity, grid_size, name="problem", method="lkh"):
    # default scale value is 100000 from "https://github.com/wouterkool/attention-learn-to-route"

    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("COMMENT", "CVRP Instance"),
                ("TYPE", "CVRP"),
                ("DIMENSION", len(loc) + 1),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                ("CAPACITY", capacity)
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, x, y) if method == "hgs" else  # we use -round=0 for hgs
            "{}\t{}\t{}".format(i + 1, int(x / grid_size * 100000 + 0.5), int(y / grid_size * 100000 + 0.5))  # VRPlib does not take floats
            for i, (x, y) in enumerate([depot] + loc)
        ]))
        f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("\n".join([
            "{}\t{}".format(i + 1, int(d))  # convert to int for lkh3, otherwise "DEMAND_SECTION: Node number out of range: 0"
            for i, d in enumerate([0] + demand)
        ]))
        f.write("\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='hgs', choices=["hgs", "lkh"])
    parser.add_argument("--datasets", nargs='+', default=["../data/CVRP/cvrp100_uniform.pkl", ], help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_false', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--disable_cache', action='store_false', help='Disable caching')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, default=1000, help="Number of instances to process")
    parser.add_argument('--offset', type=int, default=0, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='baseline_results', help="Name of results directory")

    opts = parser.parse_args()

    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    for dataset_path in opts.datasets:

        assert os.path.isfile(check_extension(dataset_path)), "File does not exist!"

        dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])

        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, "cvrp_{}".format(opts.method), dataset_basename)
            os.makedirs(results_dir, exist_ok=True)

            # out_file = os.path.join(results_dir, "{}{}{}-{}{}".format(
            #     dataset_basename,
            #     "offset{}".format(opts.offset) if opts.offset is not None else "",
            #     "n{}".format(opts.n) if opts.n is not None else "",
            #     opts.method, ext
            # ))
            dir, filename = os.path.split(dataset_path)
            out_file = os.path.join(dir, "{}_{}".format(opts.method, filename))
        else:
            out_file = opts.o

        assert opts.f or not os.path.isfile(
            out_file), "File already exists! Try running with -f option to overwrite."

        match = re.match(r'^([a-z_]+)(\d*)$', opts.method)
        assert match
        method = match[1]
        runs = 1 if match[2] == '' else int(match[2])

        start_t = time.time()

        if method == "lkh":
            use_multiprocessing = False
            executable = get_lkh_executable()
            def run_func(args):
                directory, name, *args = args
                depot, loc, demand, capacity, *args = args
                depot = depot[0] if len(depot) == 1 else depot  # if depot: [[x, y]] -> [x, y]
                grid_size = 1
                if len(args) > 0:
                    depot_types, customer_types, grid_size = args

                return solve_lkh_log(
                    executable,
                    directory, name,
                    depot, loc, demand, capacity, grid_size,
                    runs=runs, disable_cache=opts.disable_cache
                )
        elif method == "hgs":
            use_multiprocessing = False
            executable = get_hgs_executable()
            def run_func(args):
                directory, name, *args = args
                depot, loc, demand, capacity, *args = args
                depot = depot[0] if len(depot) == 1 else depot  # if depot: [[x, y]] -> [x, y]
                grid_size = 1
                if len(args) > 0:
                    depot_types, customer_types, grid_size = args

                return solve_hgs_log(
                    executable,
                    directory, name,
                    depot, loc, demand, capacity, grid_size,
                    runs=runs, disable_cache=opts.disable_cache
                )
        else:
            assert False, "Unknown method: {}".format(opts.method)

        target_dir = os.path.join(results_dir, "{}-{}".format(
            dataset_basename,
            opts.method
        ))
        assert opts.f or not os.path.isdir(target_dir), \
            "Target dir already exists! Try running with -f option to overwrite."

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        # CVRP contains tuple rather than single loc array
        dataset = load_dataset(dataset_path)

        # Note: only processing n items is handled by run_all_in_pool
        results, parallelism = run_all_in_pool(
            run_func,
            target_dir, dataset, opts, use_multiprocessing=use_multiprocessing
        )

        costs, tours, durations = zip(*results)  # Not really costs since they should be negative
        print(">> Solving {} instances within {:.2f}s using {}".format(opts.n, time.time()-start_t, opts.method))
        print("Average cost: {} +- {}".format(np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))))
        print("Average serial duration: {} +- {}".format(np.mean(durations), 2 * np.std(durations) / np.sqrt(len(durations))))
        print("Average parallel duration: {}".format(np.mean(durations) / parallelism))
        print("Calculated total duration: {}".format(timedelta(seconds=int(np.sum(durations) / parallelism))))

        results = [(i[0], i[1]) for i in results]
        save_dataset(results, out_file)  # [(obj, route), ...]
