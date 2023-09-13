import os, random, math, time
import pytz
import argparse
import pprint as pp
from datetime import datetime

from Tester import Tester
from utils import *


def args2dict(args):
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size}
    model_params = {"embedding_dim": args.embedding_dim, "sqrt_embedding_dim": args.sqrt_embedding_dim,
                    "encoder_layer_num": args.encoder_layer_num, "decoder_layer_num": args.decoder_layer_num,
                    "qkv_dim": args.qkv_dim, "head_num": args.head_num, "logit_clipping": args.logit_clipping,
                    "ff_hidden_dim": args.ff_hidden_dim, "num_experts": args.num_experts, "eval_type": args.eval_type,
                    "norm": args.norm, "norm_loc": args.norm_loc, "expert_loc": args.expert_loc, "problem": None}
    tester_params = {"checkpoint": args.checkpoint, "test_episodes": args.test_episodes, "test_batch_size": args.test_batch_size,
                     "aug_factor": args.aug_factor, "aug_batch_size": args.aug_batch_size,
                     "test_set_path": args.test_set_path, "test_set_opt_sol_path": args.test_set_opt_sol_path}

    return env_params, model_params, tester_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Towards Unified Models for Routing Problems")
    # env_params
    parser.add_argument('--problem', type=str, default="ALL", choices=["CVRP", "OVRP", "VRPB", "VRPTW", "VRPL",
                                                                       "VRPBL", "OVRPL", "VRPBTW", "OVRPLTW", "OVRPBTW", "OVRPBLTW"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, help="the number of start node, should <= problem size")

    # model_params
    parser.add_argument('--model_type', type=str, default="Single", choices=["Single", "MTL", "MOE"])
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=6, help="the number of MHA in encoder")
    parser.add_argument('--decoder_layer_num', type=int, default=1, help="the number of MHA in decoder")
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--num_experts', type=int, default=8, help="the number of FFN in a MOE layer")
    parser.add_argument('--eval_type', type=str, default="argmax", choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--expert_loc', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5], help="where to use MOE layer")
    parser.add_argument('--norm_loc', type=str, default="norm_last", choices=["norm_last", "norm_last"], help="whether conduct normalization before MHA/FFN/MOE")

    # tester_params
    parser.add_argument('--checkpoint', type=str, default="./checkpoint/epoch-10000.pt", help="load pretrained model to evaluate")
    parser.add_argument('--test_episodes', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--aug_factor', type=int, default=8, choices=[1, 8], help="whether to use instance augmentation during evaluation")
    parser.add_argument('--aug_batch_size', type=int, default=100)
    parser.add_argument('--test_set_path', type=str, default="./data/CVRP/cvrp50_uniform.pkl")
    parser.add_argument('--test_set_opt_sol_path', type=str, default="./data/CVRP/hgs_cvrp50_uniform.pkl")

    # settings (e.g., GPU)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--occ_gpu', type=float, default=0., help="occumpy (X)% GPU memory in advance, please use sparingly.")

    args = parser.parse_args()
    pp.pprint(vars(args))
    env_params, model_params, tester_params = args2dict(args)
    seed_everything(args.seed)

    if args.aug_factor != 1:
        args.test_batch_size = args.aug_batch_size
        tester_params['test_batch_size'] = tester_params['aug_batch_size']

    # set log & gpu
    # torch.set_printoptions(threshold=1000000)
    # process_start_time = datetime.now(pytz.timezone("Asia/Singapore"))
    # args.log_path = os.path.join(args.log_dir, "Test", process_start_time.strftime("%Y%m%d_%H%M%S"))
    # if not os.path.exists(args.log_path):
    #     os.makedirs(args.log_path)
    if not args.no_cuda and torch.cuda.is_available():
        occumpy_mem(args) if args.occ_gpu != 0. else print(">> No occupation needed")
        args.device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    print(">> USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(not args.no_cuda, args.gpu_id))

    # start training
    print(">> Start {} Testing ...".format(args.problem))
    tester = Tester(args=args, env_params=env_params, model_params=model_params, tester_params=tester_params)
    tester.run()
    print(">> Finish {} Testing ...".format(args.problem))
