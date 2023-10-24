import os, re, time

from utils import *


class Tester:
    def __init__(self, args, env_params, model_params, tester_params):

        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # ENV, MODEL, & Load checkpoint
        self.envs = get_env(self.args.problem)  # Env Class
        self.device = args.device
        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        self.model_params['problem'] = checkpoint['problem']  # training problem of the checkpoint
        self.model = get_model(self.args.model_type)(**self.model_params)
        num_param(self.model)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))

        # load dataset
        if tester_params['test_set_path'] is None or tester_params['test_set_path'].endswith(".pkl"):
            self.data_dir = "./data"
        else:
            # for solving instances with CVRPLIB format
            self.path_list = [os.path.join(tester_params['test_set_path'], f) for f in sorted(os.listdir(tester_params['test_set_path']))] \
                if os.path.isdir(tester_params['test_set_path']) else [tester_params['test_set_path']]
            assert self.path_list[-1].endswith(".vrp"), "Unsupported file types."

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        start_time = time.time()
        scores, aug_scores = torch.zeros(0), torch.zeros(0)

        for env_class in self.envs:
            if self.tester_params['test_set_path'] is None or self.tester_params['test_set_path'].endswith(".pkl"):
                compute_gap = not (self.tester_params['test_set_path'] is not None and self.tester_params['test_set_opt_sol_path'] is None)
                scores, aug_scores = self._test(env_class, compute_gap=compute_gap)
            else:
                for path in self.path_list:
                    score, aug_score = self._solve_cvrplib(path, env_class)
                    scores = torch.cat((scores, score), dim=0)
                    aug_scores = torch.cat((aug_scores, aug_score), dim=0)

            print(">> Evaluation finished within {:.2f}s\n".format(time.time() - start_time))

    def _test(self, env_class, compute_gap=False):
        self.time_estimator.reset()
        env = env_class(**self.env_params)
        score_AM, gap_AM = AverageMeter(), AverageMeter()
        aug_score_AM, aug_gap_AM = AverageMeter(), AverageMeter()
        scores, aug_scores = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        episode, test_num_episode = 0, self.tester_params['test_episodes']

        data_path = self.tester_params['test_set_path'] if self.tester_params['test_set_path'] \
            else os.path.join(self.data_dir, env.problem, "{}{}_uniform.pkl".format(env.problem.lower(), env.problem_size))

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            data = env.load_dataset(data_path, offset=episode, num_samples=batch_size)

            score, aug_score, all_score, all_aug_score = self._test_one_batch(data, env)
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            scores = torch.cat((scores, all_score), dim=0)
            aug_scores = torch.cat((aug_scores, all_aug_score), dim=0)

            if compute_gap:
                opt_sol_path = self.tester_params['test_set_opt_sol_path'] if self.tester_params['test_set_opt_sol_path'] \
                    else get_opt_sol_path(os.path.join(self.data_dir, env.problem), env.problem, env.problem_size)
                opt_sol = load_dataset(opt_sol_path, disable_print=True)[episode: episode + batch_size]  # [(obj, route), ...]
                opt_sol = [i[0] for i in opt_sol]
                gap = [(all_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
                aug_gap = [(all_aug_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
                gap_AM.update(sum(gap)/batch_size, batch_size)
                aug_gap_AM.update(sum(aug_gap)/batch_size, batch_size)

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            print("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                print(" \n*** Test Done on {} *** ".format(env.problem))
                print(" NO-AUG SCORE: {:.4f}, Gap: {:.4f} ".format(score_AM.avg, gap_AM.avg))
                print(" AUGMENTATION SCORE: {:.4f}, Gap: {:.4f} ".format(aug_score_AM.avg, aug_gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(score_AM.avg, gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(aug_score_AM.avg, aug_gap_AM.avg))

        return scores, aug_scores

    def _test_one_batch(self, test_data, env):
        aug_factor = self.tester_params['aug_factor']
        batch_size = test_data.size(0) if isinstance(test_data, torch.Tensor) else test_data[-1].size(0)

        # Ready
        self.model.eval()
        with torch.no_grad():
            env.load_problems(batch_size, problems=test_data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        no_aug_score_mean = no_aug_score.mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
        aug_score_mean = aug_score.mean()

        return no_aug_score_mean.item(), aug_score_mean.item(), no_aug_score, aug_score

    def _solve_cvrplib(self, path, env_class):
        """
            Solving one instance with CVRPLIB format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            elif line.startswith('DEMAND_SECTION'):
                demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            i += 1
        original_locations = locations[:, 1:]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
        loc_scaler = 1000
        locations = original_locations / loc_scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
        depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        node_demand = torch.Tensor(demand[1:, 1:].reshape((1, -1))) / capacity  # [1, n]

        env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': loc_scaler, 'device': self.device}
        env = env_class(**env_params)
        data = (depot_xy, node_xy, node_demand)
        _, _, no_aug_score, aug_score = self._test_one_batch(data, env)
        no_aug_score = torch.round(no_aug_score * loc_scaler).long()
        aug_score = torch.round(aug_score * loc_scaler).long()
        print(">> Finish solving {} -> no_aug: {} aug: {}".format(path, no_aug_score, aug_score))

        return no_aug_score, aug_score
