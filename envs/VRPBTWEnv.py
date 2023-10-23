from dataclasses import dataclass
import torch
import os, pickle
import numpy as np

__all__ = ['VRPBTWEnv']


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_service_time: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_start: torch.Tensor = None
    # shape: (batch, problem)
    node_tw_end: torch.Tensor = None
    # shape: (batch, problem)
    prob_emb: torch.Tensor = None
    # shape: (num_training_prob)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    START_NODE: torch.Tensor = None
    PROBLEM: str = None
    # shape: (batch, pomo)
    selected_count: int = None
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_time: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    open: torch.Tensor = None
    # shape: (batch, pomo)
    current_coord: torch.Tensor = None
    # shape: (batch, pomo, 2)


class VRPBTWEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.problem = "VRPBTW"
        self.env_params = env_params
        self.backhaul_ratio = 0.2
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else env_params['device']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.START_NODE = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_service_time = None
        # shape: (batch, problem+1)
        self.depot_node_tw_start = None
        # shape: (batch, problem+1)
        self.depot_node_tw_end = None
        # shape: (batch, problem+1)
        self.speed = 1.0
        self.depot_start, self.depot_end = 0., 3.  # tw for depot [0, 3]

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        self.current_time = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.open = None
        # shape: (batch, pomo)
        self.current_coord = None
        # shape: (batch, pomo, 2)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, problems=None, aug_factor=1):
        if problems is not None:
            depot_xy, node_xy, node_demand, service_time, tw_start, tw_end = problems
        else:
            depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = self.get_random_problems(batch_size, self.problem_size, normalized=True)
            node_demand = node_demand / capacity.view(-1, 1)
        self.batch_size = depot_xy.size(0)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = self.augment_xy_data_by_8_fold(depot_xy)
                node_xy = self.augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                service_time = service_time.repeat(8, 1)
                tw_start = tw_start.repeat(8, 1)
                tw_end = tw_end.repeat(8, 1)
            else:
                raise NotImplementedError

        # reset pomo_size
        self.pomo_size = min(int(self.problem_size * (1 - self.backhaul_ratio)), self.pomo_size)
        self.START_NODE = torch.arange(start=1, end=self.problem_size+1)[None, :].expand(self.batch_size, -1).to(self.device)
        self.START_NODE = self.START_NODE[node_demand > 0].reshape(self.batch_size, -1)[:, :self.pomo_size]

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        depot_service_time = torch.zeros(size=(self.batch_size, 1)).to(self.device)
        depot_tw_start = torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_start
        depot_tw_end = torch.ones(size=(self.batch_size, 1)).to(self.device) * self.depot_end
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_service_time = torch.cat((depot_service_time, service_time), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_start = torch.cat((depot_tw_start, tw_start), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_tw_end = torch.cat((depot_tw_end, tw_end), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size).to(self.device)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size).to(self.device)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_service_time = service_time
        self.reset_state.node_tw_start = tw_start
        self.reset_state.node_tw_end = tw_end
        self.reset_state.prob_emb = torch.FloatTensor([1, 0, 1, 0, 1]).unsqueeze(0).to(self.device)  # bit vector for [C, O, B, L, TW]

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.open = torch.zeros(self.batch_size, self.pomo_size).to(self.device)
        self.step_state.START_NODE = self.START_NODE
        self.step_state.PROBLEM = self.problem

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1)).to(self.device)
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool).to(self.device)
        # shape: (batch, pomo)
        self.current_time = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size)).to(self.device)
        # shape: (batch, pomo)
        self.current_coord = self.depot_node_xy[:, :1, :]  # depot
        # shape: (batch, pomo, 2)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        current_coord = self.depot_node_xy[torch.arange(self.batch_size)[:, None], selected]
        # shape: (batch, pomo, 2)
        new_length = (current_coord - self.current_coord).norm(p=2, dim=-1)
        # shape: (batch, pomo)
        self.length = self.length + new_length
        self.length[self.at_the_depot] = 0  # reset the length of route at the depot
        self.current_coord = current_coord

        # Mask
        ####################################
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        # Only for VRPB: reset load to 0.
        # >> Old implementation
        #   a. if visit backhaul nodes in the first two POMO moves (i.e., depot -> backhaul, the route is mixed with backhauls and linehauls alternatively);
        #   b. if only backhaul nodes unserved, we relax the load to be 0 (i.e., the vehicle only visit backhauls nodes in the last few routes).
        # if self.selected_node_list.size(-1) == 1:  # POMO first move
        #     depot_backhaul = self.at_the_depot & (self.depot_node_demand[:, 1:self.pomo_size+1] < 0.)
        #     # shape: (batch, pomo)
        #     self.load[depot_backhaul] = 0.
        # else:
        # >> New implementation - Remove constraint a, the POMO start node should be a linehaul.
        unvisited_demand = demand_list + self.visited_ninf_flag
        # shape: (batch, pomo, problem+1)
        linehauls_unserved = torch.where(unvisited_demand > 0., True, False)
        reset_index = self.at_the_depot & (~linehauls_unserved.any(dim=-1))
        # shape: (batch, pomo)
        self.load[reset_index] = 0.

        # capacity constraint
        #   a. the remaining vehicle capacity >= the customer demands
        #   b. the remaining vehicle capacity <= the vehicle capacity (i.e., 1.0)
        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        exceed_capacity = self.load[:, :, None] - demand_list > 1.0 + round_error_epsilon
        self.ninf_mask[exceed_capacity] = float('-inf')

        # time window constraint
        #   current_time: the end time of serving the current node
        #   a. max(current_time + travel_time, tw_start) or current_time + travel_time <= tw_end
        #   b. vehicle should return to the depot: max(current_time + travel_time, tw_start) + service_time + dist(node, depot)/speed <= self.depot_end
        self.current_time = torch.max(self.current_time + new_length / self.speed, self.depot_node_tw_start[torch.arange(self.batch_size)[:, None], selected]) + self.depot_node_service_time[torch.arange(self.batch_size)[:, None], selected]
        self.current_time[self.at_the_depot] = 0
        # shape: (batch, pomo)
        arrival_time = torch.max(self.current_time[:, :, None] + (self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed, self.depot_node_tw_start[:, None, :].expand(-1, self.pomo_size, -1))
        out_of_tw = arrival_time > self.depot_node_tw_end[:, None, :].expand(-1, self.pomo_size, -1) + round_error_epsilon
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[out_of_tw] = float('-inf')
        fail_return_depot = arrival_time + self.depot_node_service_time[:, None, :].expand(-1, self.pomo_size, -1) + (self.depot_node_xy[:, None, :1, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)).norm(p=2, dim=-1) / self.speed > self.depot_end + round_error_epsilon
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[fail_return_depot] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.current_time = self.current_time
        self.step_state.length = self.length
        self.step_state.current_coord = self.current_coord

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        if self.loc_scaler:
            segment_lengths = torch.round(segment_lengths * self.loc_scaler) / self.loc_scaler

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def generate_dataset(self, num_samples, problem_size, path):
        data = self.get_random_problems(num_samples, problem_size, normalized=False)
        dataset = [attr.cpu().tolist() for attr in data]
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(path, 'wb') as f:
            pickle.dump(list(zip(*dataset)), f, pickle.HIGHEST_PROTOCOL)
        print("Save VRPBTW dataset to {}".format(path))

    def load_dataset(self, path, offset=0, num_samples=1000, disable_print=True):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
            if not disable_print:
                print(">> Load {} data ({}) from {}".format(len(data), type(data), path))
        depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data], [i[4] for i in data], [i[5] for i in data], [i[6] for i in data]
        depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(node_demand), torch.Tensor(capacity), torch.Tensor(service_time), torch.Tensor(tw_start), torch.Tensor(tw_end)
        node_demand = node_demand / capacity.view(-1, 1)
        data = (depot_xy, node_xy, node_demand, service_time, tw_start, tw_end)
        return data

    def get_random_problems(self, batch_size, problem_size, normalized=True):
        depot_xy = torch.rand(size=(batch_size, 1, 2))  # (batch, 1, 2)
        node_xy = torch.rand(size=(batch_size, problem_size, 2))  # (batch, problem, 2)

        if problem_size == 20:
            demand_scaler = 30
        elif problem_size == 50:
            demand_scaler = 40
        elif problem_size == 100:
            demand_scaler = 50
        elif problem_size == 200:
            demand_scaler = 70
        else:
            raise NotImplementedError

        # time windows (vehicle speed = 1.):
        #   1. The setting of "MTL for Routing Problem with Zero-Shot Generalization".
        """
        self.depot_start, self.depot_end = 0., 4.6.
        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, problem_size)
        tw_length = b + (c - b) * torch.rand(batch_size, problem_size)
        c = (node_xy - depot_xy).norm(p=2, dim=-1)
        h_max = (self.depot_end - service_time - tw_length) / c * self.speed - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, problem_size)) * c / self.speed
        tw_end = tw_start + tw_length
        """
        #   2. See "Learning to Delegate for Large-scale Vehicle Routing" in NeurIPS 2021.
        #   Note: this setting follows a similar procedure as in Solomon, and therefore is more realistic and harder.
        service_time = torch.ones(batch_size, problem_size) * 0.2
        travel_time = (node_xy - depot_xy).norm(p=2, dim=-1) / self.speed
        a, b = self.depot_start + travel_time, self.depot_end - travel_time - service_time
        time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
        time_half_width = (service_time / 2 - self.depot_end / 3) * torch.rand(batch_size, problem_size) + self.depot_end / 3
        tw_start = torch.clamp(time_centers - time_half_width, min=self.depot_start, max=self.depot_end)
        tw_end = torch.clamp(time_centers + time_half_width, min=self.depot_start, max=self.depot_end)
        # shape: (batch, problem)

        # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
        instance_invalid, round_error_epsilon = False, 0.00001
        total_time = torch.max(0 + (depot_xy - node_xy).norm(p=2, dim=-1) / self.speed, tw_start) + service_time + (node_xy - depot_xy).norm(p=2, dim=-1) / self.speed > self.depot_end + round_error_epsilon
        # (batch, problem)
        instance_invalid = total_time.any()

        if instance_invalid:
            print(">> Invalid instances, Re-generating ...")
            return self.get_random_problems(batch_size, problem_size, normalized=normalized)
        elif normalized:
            node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)  # (batch, problem)
            backhauls_index = torch.randperm(problem_size)[:int(problem_size * self.backhaul_ratio)]  # randomly select 20% customers as backhaul ones
            node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]
            return depot_xy, node_xy, node_demand, service_time, tw_start, tw_end
        else:
            node_demand = torch.Tensor(np.random.randint(1, 10, size=(batch_size, problem_size)))  # (unnormalized) shape: (batch, problem)
            backhauls_index = torch.randperm(problem_size)[:int(problem_size * self.backhaul_ratio)]
            node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]
            capacity = torch.Tensor(np.full(batch_size, demand_scaler))
            return depot_xy, node_xy, node_demand, capacity, service_time, tw_start, tw_end

    def augment_xy_data_by_8_fold(self, xy_data):
        # xy_data.shape: (batch, N, 2)

        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        # x,y shape: (batch, N, 1)

        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, N, 2)

        return aug_xy_data
