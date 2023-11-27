import re
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils import *


class Trainer:
    def __init__(self, args, env_params, model_params, optimizer_params, trainer_params):
        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.device = args.device
        self.log_path = args.log_path
        self.result_log = {"val_score": [], "val_gap": []}

        # Main Components
        self.envs = get_env(self.args.problem)  # a list of envs classes (different problems), remember to initialize it!
        self.model = get_model(self.args.model_type)(**self.model_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        num_param(self.model)

        # Restore
        self.start_epoch = 1
        if args.checkpoint is not None:
            checkpoint_fullname = args.checkpoint
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.start_epoch = 1 + checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = checkpoint['epoch'] - 1
            print(">> Checkpoint (Epoch: {}) Loaded!".format(checkpoint['epoch']))

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            print('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            print("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['model_save_interval']
            validation_interval = self.trainer_params['validation_interval']

            # MTL Validation & save latest images
            if epoch == 1 or (epoch % validation_interval == 0):
                val_problems = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]
                val_episodes, problem_size = 1000, self.env_params['problem_size']
                dir = [os.path.join("./data", prob) for prob in val_problems]
                paths = ["{}{}_uniform.pkl".format(prob.lower(), problem_size) for prob in val_problems]
                val_envs = [get_env(prob)[0] for prob in val_problems]
                for i, path in enumerate(paths):
                    # if no optimal solution provided, set compute_gap to False
                    score, gap = self._val_and_stat(dir[i], path, val_envs[i](**{"problem_size": problem_size, "pomo_size": problem_size}), batch_size=500, val_episodes=val_episodes, compute_gap=True)
                    self.result_log["val_score"].append(score)
                    self.result_log["val_gap"].append(gap)

                score_image_prefix = '{}/latest_val_score'.format(self.log_path)
                gap_image_prefix = '{}/latest_val_gap'.format(self.log_path)
                x, y1, y2, label = [], [], [], []
                for i, path in enumerate(paths):
                    y1.append([r for j, r in enumerate(self.result_log["val_score"]) if j % len(paths) == i])
                    y2.append([r for j, r in enumerate(self.result_log["val_gap"]) if j % len(paths) == i])
                    x.append([j * validation_interval for j in range(len(y1[-1]))])
                    label.append(val_problems[i])
                show(x, y1, label, title="Validation", xdes="Epoch", ydes="Score", path="{}.pdf".format(score_image_prefix))
                show(x, y2, label, title="Validation", xdes="Epoch", ydes="Opt. Gap (%)", path="{}.pdf".format(gap_image_prefix))

            if all_done or (epoch % model_save_interval == 0):
                print("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'problem': self.args.problem,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log
                }
                torch.save(checkpoint_dict, '{}/epoch-{}.pt'.format(self.log_path, epoch))

    def _train_one_epoch(self, epoch):
        episode = 0
        score_AM, loss_AM = AverageMeter(), AverageMeter()
        train_num_episode = self.trainer_params['train_episodes']

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            env = random.sample(self.envs, 1)[0](**self.env_params)
            data = env.get_random_problems(batch_size, self.env_params["problem_size"])
            avg_score, avg_loss = self._train_one_batch(data, env)
            # print(avg_score, avg_loss)

            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            episode += batch_size

        # Log Once, for each epoch
        print('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'.format(epoch, 100. * episode / train_num_episode, score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, data, env):
        self.model.train()
        self.model.set_eval_type(self.model_params["eval_type"])
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        state, reward, done = env.pre_step()
        # print("{}\n".format(state.PROBLEM))
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        loss_mean = loss.mean()
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        if hasattr(self.model, "aux_loss"):
            loss_mean = loss_mean + self.model.aux_loss  # add aux(moe)_loss for load balancing (default coefficient: 1e-2)

        # Step & Return
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        return score_mean.item(), loss_mean.item()

    def _val_one_batch(self, data, env, aug_factor=1, eval_type="argmax"):
        self.model.eval()
        self.model.set_eval_type(eval_type)
        batch_size = data.size(0) if isinstance(data, torch.Tensor) else data[-1].size(0)
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value

        return no_aug_score, aug_score

    def _val_and_stat(self, dir, val_path, env, batch_size=500, val_episodes=1000, compute_gap=False):
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list = [], [], [], []
        episode, no_aug_score, aug_score = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)

        while episode < val_episodes:
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            data = env.load_dataset(os.path.join(dir, val_path), offset=episode, num_samples=bs)
            no_aug, aug = self._val_one_batch(data, env, aug_factor=8, eval_type="argmax")
            no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
            aug_score = torch.cat((aug_score, aug), dim=0)
            episode += bs

        no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
        aug_score_list.append(round(aug_score.mean().item(), 4))

        if compute_gap:
            opt_sol = load_dataset(get_opt_sol_path(dir, env.problem, data[1].size(1)), disable_print=True)[: val_episodes]
            opt_sol = [i[0] for i in opt_sol]
            gap = [(no_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            no_aug_gap_list.append(round(sum(gap) / len(gap), 4))
            gap = [(aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            aug_gap_list.append(round(sum(gap) / len(gap), 4))
            print(">> Val Score on {}: NO_AUG_Score: {}, NO_AUG_Gap: {}% --> AUG_Score: {}, AUG_Gap: {}%".format(val_path, no_aug_score_list, no_aug_gap_list, aug_score_list, aug_gap_list))
            return aug_score_list[0], aug_gap_list[0]
        else:
            print(">> Val Score on {}: NO_AUG_Score: {}, --> AUG_Score: {}".format(val_path, no_aug_score_list, aug_score_list))
            return aug_score_list[0], 0
