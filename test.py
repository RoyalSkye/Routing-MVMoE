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
                    "norm": args.norm, "norm_loc": args.norm_loc, "expert_loc": args.expert_loc, "problem": None,
                    "topk": args.topk, "routing_level": args.routing_level, "routing_method": args.routing_method}
    tester_params = {"checkpoint": args.checkpoint, "test_episodes": args.test_episodes, "test_batch_size": args.test_batch_size,
                     "sample_size": args.sample_size, "aug_factor": args.aug_factor, "aug_batch_size": args.aug_batch_size,
                     "test_set_path": args.test_set_path, "test_set_opt_sol_path": args.test_set_opt_sol_path,
                     "fine_tune_episodes": args.fine_tune_episodes, "fine_tune_epochs": args.fine_tune_epochs,
                     "fine_tune_batch_size": args.fine_tune_batch_size, "fine_tune_aug_factor": args.fine_tune_aug_factor,
                     "lr": args.lr, "weight_decay": args.weight_decay}

    return env_params, model_params, tester_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts")
    # env_params
    parser.add_argument('--problem', type=str, default="ALL", choices=["ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                                                       "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                                                       "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument('--problem_size', type=int, default=100)
    parser.add_argument('--pomo_size', type=int, default=100, help="the number of start node, should <= problem size")

    # model_params
    parser.add_argument('--model_type', type=str, default="MOE_LIGHT", choices=["SINGLE", "MTL", "MOE", "MOE_LIGHT"])
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--sqrt_embedding_dim', type=float, default=128**(1/2))
    parser.add_argument('--encoder_layer_num', type=int, default=6, help="the number of MHA in encoder")
    parser.add_argument('--decoder_layer_num', type=int, default=1, help="the number of MHA in decoder")
    parser.add_argument('--qkv_dim', type=int, default=16)
    parser.add_argument('--head_num', type=int, default=8)
    parser.add_argument('--logit_clipping', type=float, default=10)
    parser.add_argument('--ff_hidden_dim', type=int, default=512)
    parser.add_argument('--num_experts', type=int, default=4, help="the number of FFN in a MOE layer")
    parser.add_argument('--eval_type', type=str, default="argmax", choices=["argmax", "softmax"])
    parser.add_argument('--norm', type=str, default="instance", choices=["batch", "batch_no_track", "instance", "layer", "rezero", "none"])
    parser.add_argument('--norm_loc', type=str, default="norm_last", choices=["norm_last", "norm_last"], help="whether conduct normalization before MHA/FFN/MOE")
    parser.add_argument('--topk', type=int, default=2, help="how many ffn(s) to route for each input")
    parser.add_argument('--expert_loc', type=str, nargs='+', default=['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'], help="where to use MOE layer")
    parser.add_argument('--routing_level', type=str, default="node", choices=["node", "instance", "problem"], help="routing level for MOE")
    parser.add_argument('--routing_method', type=str, default="input_choice", choices=["input_choice", "expert_choice", "soft_moe", "random"], help="only activate for instance-level and token-level routing")

    # tester_params
    parser.add_argument('--checkpoint', type=str, default="./checkpoint/epoch-5000.pt", help="load pretrained model to evaluate")
    parser.add_argument('--test_episodes', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--sample_size', type=int, default=10, help="only activate if eval_type is softmax")
    parser.add_argument('--aug_factor', type=int, default=8, choices=[1, 8], help="whether to use instance augmentation during evaluation")
    parser.add_argument('--aug_batch_size', type=int, default=100)
    parser.add_argument('--test_set_path', type=str, default=None, help="evaluate on default test dataset if None")
    parser.add_argument('--test_set_opt_sol_path', type=str, default=None, help="evaluate on default test dataset if None")

    parser.add_argument('--fine_tune_epochs', type=int, default=0, help="fine tune the pretrained model if > 0")
    parser.add_argument('--fine_tune_episodes', type=int, default=10000)
    parser.add_argument('--fine_tune_batch_size', type=int, default=64 * 2)
    parser.add_argument('--fine_tune_aug_factor', type=int, default=1, choices=[1, 8], help="whether to use instance augmentation during fine tuning")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

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
