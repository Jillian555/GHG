import argparse
from distutils.util import strtobool
from pipeline import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GHG')
    parser.add_argument("--n_agents", type=int, default=0, help="number of agents")
    parser.add_argument("--dataset", type=str, default='Reddit-CL',
                        help='CoraFull-CL, Arxiv-CL, Reddit-CL, Cora-CL, Citeseer-CL, SLAP-CL, Computers-CL')
    parser.add_argument('--par', type=str, default='noniid0.1', help="dataset partition, [noniid0.1, noniid0.5]")
    parser.add_argument('--n_cls', type=int, default=None, help='will be assigned during running')
    parser.add_argument("--n_cls_task", type=int, default=10, help='number of classes per task')
    parser.add_argument('--n-task', default=0, help='will be assigned during running')
    parser.add_argument('--n_rnds', type=int, default=0, help='number of interaction rounds')
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use.")
    parser.add_argument("--seed", type=int, default=1, help="seed for exp")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs, default = 200")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="weight decay")


    parser.add_argument('--w_c', type=float, default=1.0, help="weight of complementarity")
    parser.add_argument('--w_s', type=float, default=1.0, help="weight of similarity")
    parser.add_argument('--w_kl', type=float, default=0, help="weight decay of KL divergence loss")
    parser.add_argument('--w_sigma', type=float, default=0.1, help="weight of standard deviation loss")
    parser.add_argument('--w_col', type=float, default=1.0, help="weight of fine-grained collaboration")
    parser.add_argument('--compression_ratio', type=float, default=0.1, help="compression ratio")
    parser.add_argument('--lr_feat', type=float, default=0.005, help="learning rate of synthesis")
    parser.add_argument('--syn_epochs', type=int, default=50, help="number of synthesis epochs")
    parser.add_argument('--pca', type=int, default=3, help="n_components of PCA")


    parser.add_argument('--backbone', type=str, default='GCN', help="backbone GNN, [GAT, GCN, GIN]")
    parser.add_argument('--method', type=str,
                        choices=["bare", 'lwf', 'gem', 'ewc', 'mas', 'twp', 'ergnn', 'tpp', 'ghg', 'jointtrain', 'joint',
                                 'Joint'], default="bare", help="baseline continual learning method")
    parser.add_argument('--share-labels', type=strtobool, default=False,
                        help='task-IL specific, whether to share output label space for different tasks')
    parser.add_argument('--inter-task-edges', type=strtobool, default=False,
                        help='whether to keep the edges connecting nodes from different tasks')
    parser.add_argument('--classifier-increase', type=strtobool, default=True,
                        help='(deprecated) class-IL specific, whether to enlarge the label space with the coming of new classes, unrealistic to be set as False')
    # extra parameters
    parser.add_argument('--refresh_data', type=strtobool, default=False,
                        help='whether to load existing splitting or regenerate')
    parser.add_argument('--d_dtat', default=None, help='will be assigned during running')
    parser.add_argument('--ratio_valid_test', nargs='+', default=[0.2, 0.2],
                        help='ratio of nodes used for valid and test') #cora0.4, 0.4   corafull0.2, 0.2
    parser.add_argument('--transductive', type=strtobool, default=True, help='using transductive or inductive')
    parser.add_argument('--default_split', type=strtobool, default=False,
                        help='whether to  use the data split provided by the dataset')
    # parameters of backbone
    parser.add_argument('--GAT-args',
                        default={'num_layers': 1, 'num_hidden': 32, 'heads': 8, 'out_heads': 1, 'feat_drop': .6,
                                 'attn_drop': .6, 'negative_slope': 0.2, 'residual': False})
    parser.add_argument('--GCN-args', default={'h_dims': [256], 'dropout': 0.0, 'batch_norm': False})
    parser.add_argument('--SGC_args', default={'h_dims': [256], 'dropout': 0.0, 'bias': False, 'k': 2, 'alpha': 0.05,
                                               'batch_norm': False, 'linear_bias': False, 'linear': 'nn.Linear'})
    parser.add_argument('--GIN-args', default={'h_dims': [256], 'dropout': 0.0})
    parser.add_argument('--hidden', default=128, help='the hidden units of GNN')
    # parameters of continual learning methods
    parser.add_argument('--ergnn_args', type=str2dict, default={'budget': [100], 'd': [0.5], 'sampler': ['CM']},
                        help='sampler options: CM, CM_plus, MF, MF_plus')
    parser.add_argument('--tpp_args', type=str2dict, default={'prompts': [3], 'pe': [0.2], 'pf': [0.3]})
    parser.add_argument('--ghg_args', type=str2dict, default={'prompts': [3], 'pe': [0.2], 'pf': [0.3]})
    parser.add_argument('--lwf_args', type=str2dict, default={'lambda_dist': [1.0, 10.0], 'T': [2.0, 20.0]})
    parser.add_argument('--twp_args', type=str2dict, default={'lambda_l': 10000., 'lambda_t': 10000., 'beta': 0.01})
    parser.add_argument('--ewc_args', type=str2dict, default={'memory_strength': 10000.})
    parser.add_argument('--mas_args', type=str2dict, default={'memory_strength': 10000.})
    parser.add_argument('--gem_args', type=str2dict, default={'memory_strength': 0.5, 'n_memories': 100})
    parser.add_argument('--bare_args', type=str2dict, default={'Na': None})
    parser.add_argument('--joint_args', type=str2dict, default={'Na': None})
    # other parameters
    parser.add_argument('--cls-balance', type=strtobool, default=True,
                        help='whether to balance the cls when training and testing')
    parser.add_argument('--repeats', type=int, default=1,
                        help='how many times to repeat the experiments for the mean and std')
    parser.add_argument('--ILmode', default='classIL', choices=['taskIL', 'classIL'])
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--bs', type=int, default=5000, help="batch size of pre-training")
    parser.add_argument('--minibatch', type=strtobool, default=False, help='whether to use the mini-batch training')
    parser.add_argument('--batch_shuffle', type=strtobool, default=True,
                        help='whether to shuffle the data when constructing the dataloader')
    parser.add_argument('--sample_nbs', type=strtobool, default=True,
                        help='whether to sample neighbors instead of using all')
    parser.add_argument('--n_nbs_sample', type=lambda x: [int(i) for i in x.replace(' ', '').split(',')],
                        default=[10, 25],
                        help='number of neighbors to sample per hop, use comma to separate the numbers when using the command line, e.g. 10,25 or 10, 25')
    parser.add_argument('--nb_sampler', default=None)
    parser.add_argument('--replace_illegal_char', type=strtobool, default=False)
    parser.add_argument('--ori_data_path', type=str, default='/root/code/cglb/ncgl/store/data/',
                        help='the root path to raw data')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='the path to processed data (splitted into tasks)')
    parser.add_argument('--result_path', type=str, default='./results', help='the path for saving results')
    parser.add_argument('--load_check', type=strtobool, default=False,
                        help='whether to check the existence of processed data by loading')
    parser.add_argument('--perform_testing', type=strtobool, default=True,
                        help='whether to check the existence of processed data by loading')

    args = parser.parse_args()
    args.ratio_valid_test = [float(i) for i in args.ratio_valid_test]
    set_seed(args)
    mkdir_if_missing(f'{args.data_path}')

    method_args = {'lwf': args.lwf_args, 'twp': args.twp_args, 'ewc': args.ewc_args,
                   'bare': args.bare_args, 'gem': args.gem_args, 'mas': args.mas_args,
                   'ergnn': args.ergnn_args, 'tpp': args.tpp_args, 'ghg': args.ghg_args,
                   'joint': args.joint_args}
    backbone_args = {'GCN': args.GCN_args, 'GAT': args.GAT_args, 'GIN': args.GIN_args, 'SGC': args.SGC_args}
    hyp_param_list = compose_hyper_params(method_args[args.method])

    AP_best, name_best = -0.1, None
    AP_dict = {str(hyp_params).replace("'", '').replace(' ', '').replace(',', '_').replace(':', '_'): [] for hyp_params
               in hyp_param_list}
    AF_dict = {str(hyp_params).replace("'", '').replace(' ', '').replace(',', '_').replace(':', '_'): [] for hyp_params
               in hyp_param_list}
    PM_dict = {str(hyp_params).replace("'", '').replace(' ', '').replace(',', '_').replace(':', '_'): [] for hyp_params
               in hyp_param_list}

    for hyp_params in hyp_param_list:
        hyp_params_str = str(hyp_params).replace("'", '').replace(' ', '').replace(',', '_').replace(':', '_')
        print(hyp_params_str)
        assign_hyp_param(args, hyp_params)
        args.nb_sampler = dgl.dataloading.MultiLayerNeighborSampler(
            args.n_nbs_sample) if args.sample_nbs else dgl.dataloading.MultiLayerFullNeighborSampler(2)

        main = get_pipeline(args)
        from memory import test_mem
        test_mem(args)
        train_ratio = round(1 - args.ratio_valid_test[0] - args.ratio_valid_test[1], 2)
        if args.ILmode == 'classIL':
            subfolder = f'inter_task_edges/cls_IL/train_ratio_{train_ratio}/' if args.inter_task_edges else f'no_inter_task_edges/cls_IL/train_ratio_{train_ratio}/'
        elif args.ILmode == 'taskIL':
            subfolder = f'inter_task_edges/tsk_IL/train_ratio_{train_ratio}/' if args.inter_task_edges else f'no_inter_task_edges/tsk_IL/train_ratio_{train_ratio}/'
        name = f'{subfolder}val_{args.dataset}_{args.method}_{args.n_agents}_{args.n_rnds}_{args.par}_{args.syn_epochs}_{args.reduction_rate}_{args.wc}_{args.ws}_{args.wc_cls}_{args.ws_cls}_{args.w_sigma}_{args.w_fea}_{args.repeats}'
        # if args.minibatch:
        #     name = name + f'_bs{args.batch_size}'
        mkdir_if_missing(f'{args.result_path}/' + subfolder)
        if args.replace_illegal_char:
            name = remove_illegal_characters(name)

        print('method args are', hyp_params)
        As, APs, AFs = [], [], []
        for ite in range(args.repeats):
            print(name, ite)
            args.current_model_save_path = [name, ite]
            final_A_, final_AP_, final_AF_, final_A, final_AP, final_AF = main(args, valid=True)
            As.append(final_A_)
            APs.append(final_AP_)
            AFs.append(final_AF_)
            torch.cuda.empty_cache()
            AP_dict[hyp_params_str].append(final_AP_)
        import numpy as np
        #print(f"A: {np.mean(As):.2f}±{np.std(As, ddof=1):.2f}", flush=True)
        print(f"AP: {np.mean(APs):.2f}±{np.std(APs, ddof=1):.2f}", flush=True)
        print(f"AF: {np.mean(AFs):.2f}±{np.std(AFs, ddof=1):.2f}", flush=True)

        if np.mean(AP_dict[hyp_params_str]) > AP_best:
            AP_best = np.mean(AP_dict[hyp_params_str])
            hyp_best_str = hyp_params_str
            name_best = name



    # config_name = name_best.split('/')[-1]
    # subfolder_c = name_best.split(config_name)[-2]
    # if args.perform_testing:
    #     print('----------Now in testing--------')
    #
    #     for ite in range(args.repeats):
    #         args.current_model_save_path = [name_best, ite]
    #         final_AP_, final_AF_, final_AP, final_AF = main(args, valid=False)






