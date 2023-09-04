from dataclasses import dataclass
import torch
import os, random, pickle
import numpy as np

__all__ = ['TSPEnv']


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.loc_scaler = env_params['loc_scaler'] if 'loc_scaler' in env_params.keys() else None
        self.device = torch.device('cuda', torch.cuda.current_device()) if 'device' not in env_params.keys() else env_params['device']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size, problems=None, aug_factor=1):
        if problems is not None:
            self.batch_size = problems.size(0)
            self.problems = problems
        else:
            self.batch_size = batch_size
            self.problems = self.get_random_problems(batch_size, self.problem_size)
            # problems.shape: (batch, problem, 2)

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.problems = self.augment_xy_data_by_8_fold(self.problems)
                # shape: (8*batch, problem, 2)
            else:
                raise NotImplementedError

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)  # (batch_size, pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long).to(self.device)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size)).to(self.device)
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        return Reset_State(self.problems), reward, done

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

        # returning values
        done = (self.selected_count == self.problem_size)
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 2)
        # shape: (batch, pomo, problem, 2)
        seq_expanded = self.problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 2)

        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, problem, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()
        # shape: (batch, pomo, problem)

        if self.loc_scaler:
            segment_lengths = torch.round(segment_lengths * self.loc_scaler) / self.loc_scaler

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def load_dataset(self, path, offset=0, num_samples=1000):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset+num_samples]
        return torch.Tensor(data)

    def get_random_problems(self, batch_size, problem_size):
        problems = torch.rand(size=(batch_size, problem_size, 2))
        # problems.shape: (batch, problem, 2)
        return problems

    def get_mix_problems(self, batch_size, problem_size):
        """
            Generate instances following Gaussian Mixture.
        """
        def generate_gm_tsp(dataset_size, graph_size, num_modes=0, cdist=0):
            from sklearn.preprocessing import MinMaxScaler
            data = []
            for i in range(dataset_size):
                nums = np.random.multinomial(graph_size, np.ones(num_modes) / num_modes)
                xy = []
                for num in nums:
                    center = np.random.uniform(0, cdist, size=(1, 2))
                    nxy = np.random.multivariate_normal(mean=center.squeeze(), cov=np.eye(2, 2), size=(num,))
                    xy.extend(nxy)
                xy = np.array(xy)
                xy = MinMaxScaler().fit_transform(xy)
                data.append(xy)
            return torch.Tensor(np.array(data))

        task_set = [(0, 0), (1, 1)] + [(m, c) for m in [i for i in range(1, 10)] for c in [10, 20, 30, 40, 50]]
        task = random.sample(task_set, 1)[0]
        if task == (0, 0):
            return torch.rand(size=(batch_size, problem_size, 2))
        else:
            return generate_gm_tsp(batch_size, problem_size, task[0], task[1])

    def augment_xy_data_by_8_fold(self, problems):
        # problems.shape: (batch, problem, 2)

        x = problems[:, :, [0]]
        y = problems[:, :, [1]]
        # x,y shape: (batch, problem, 1)

        dat1 = torch.cat((x, y), dim=2)
        dat2 = torch.cat((1 - x, y), dim=2)
        dat3 = torch.cat((x, 1 - y), dim=2)
        dat4 = torch.cat((1 - x, 1 - y), dim=2)
        dat5 = torch.cat((y, x), dim=2)
        dat6 = torch.cat((1 - y, x), dim=2)
        dat7 = torch.cat((y, 1 - x), dim=2)
        dat8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
        # shape: (8*batch, problem, 2)

        return aug_problems
