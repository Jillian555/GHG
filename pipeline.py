import pickle
import torch
import dgl
from Backbones.model_factory import get_model
from Backbones.utils import evaluatewp, NodeLevelDataset
from training.utils import *
import importlib
import copy
import warnings

warnings.filterwarnings('ignore')


def get_pipeline(args):
    return pipeline_class_IL_no_inter_edge


def data_prepare(args, dataset):
    torch.cuda.set_device(args.gpu)
    str_int_tsk = 'inter_tsk_edge' if args.inter_task_edges else 'no_inter_tsk_edge'
    n_task = args.n_cls // (args.n_cls_task)
    for task in range(n_task):
        task_cls = [cls for cls in range(task * args.n_cls_task, (task + 1) * args.n_cls_task)]
        try:
            subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(
                open(f'{args.data_path}/{str_int_tsk}/{args.dataset}_as{args.n_agents}_t{task}_p{args.par}.pkl', 'rb'))
        except:
            print(f'preparing data for task {task}')
            if args.inter_task_edges:  # prepare
                mkdir_if_missing(f'{args.data_path}/inter_tsk_edge')
                cls_retain = []
                for clss in args.task_seq[0:task + 1]:
                    cls_retain.extend(clss)
                subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids] = dataset.get_graph(
                    tasks_to_retain=cls_retain, n_agents=args.n_agents, partition=args.par)
                with open(f'{args.data_path}/inter_tsk_edge/{args.dataset}_{task_cls}.pkl', 'wb') as f:
                    pickle.dump([subgraph, ids_per_cls_all, [train_ids, valid_ids, test_ids]], f)
            else:
                mkdir_if_missing(f'{args.data_path}/no_inter_tsk_edge')
                subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = dataset.get_graph(tasks_to_retain=task_cls,
                                                                                            n_agents=args.n_agents,
                                                                                            partition=args.par)
                with open(
                        f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_as{args.n_agents}_t{task}_p{args.par}.pkl',
                        'wb') as f:
                    pickle.dump([subgraph, ids_per_cls, [train_ids, valid_ids, test_ids]], f)


def pipeline_class_IL_no_inter_edge(args, valid=False):
    epochs = args.epochs if valid else 0
    torch.cuda.set_device(args.gpu)
    dataset = NodeLevelDataset(args.dataset, ratio_valid_test=args.ratio_valid_test, args=args)
    args.d_data, _ = dataset.d_data, dataset.n_cls
    n_agents_cls_per_task = args.n_cls_task
    args.n_tasks = args.n_cls // n_agents_cls_per_task
    data_prepare(args, dataset)

    model = get_model(args).cuda(args.gpu) if valid else None
    life_model = importlib.import_module(f'Baselines.{args.method}_model')
    life_model_ins = life_model.NET(model, args) if valid else None

    acc_matrix_as, acc_mean_as, lms, principal_list, principal_list_as, similarity_as = [], [], [], [], [], []
    labels_syn_as, fea_syn_as, g_syn_as, labels_syn_as_as, fea_syn_as_as, g_syn_as_as = [], [], [], [], [], []
    local_train_sizes, local_train_sizes_cls, topo_fea_as, topo_fea_cls_as = [], [], [], []
    acc_matrix = np.zeros([args.n_tasks, args.n_tasks])
    labels2 = [copy.deepcopy([]) for _ in range(args.n_tasks)]
    g2 = [copy.deepcopy([]) for _ in range(args.n_tasks)]
    fea2 = [copy.deepcopy([]) for _ in range(args.n_tasks)]
    prototypes_ = torch.zeros(args.n_tasks, args.d_data)
    graph_matrix = torch.ones(args.n_agents, args.n_agents)
    for a_id in range(args.n_agents):
        acc_matrix_as.append(copy.deepcopy(acc_matrix))
        acc_mean_as.append(copy.deepcopy([]))
        topo_fea_as.append(copy.deepcopy([]))
        topo_fea_cls_as.append(copy.deepcopy([]))
        local_train_sizes.append(copy.deepcopy([]))
        local_train_sizes_cls.append(copy.deepcopy([]))
        principal_list.append(copy.deepcopy([]))
        principal_list_as.append(copy.deepcopy([]))
        similarity_as.append(copy.deepcopy([]))
        lms.append(copy.deepcopy(life_model_ins))
        labels_syn_as.append(copy.deepcopy(labels2))
        g_syn_as.append(copy.deepcopy(g2))
        fea_syn_as.append(copy.deepcopy(fea2))
    for a_id in range(args.n_agents):
        labels_syn_as_as.append(copy.deepcopy(labels_syn_as))
        g_syn_as_as.append(copy.deepcopy(g_syn_as))
        fea_syn_as_as.append(copy.deepcopy(fea_syn_as))
    print(args)

    name, ite = args.current_model_save_path
    config_name = name.split('/')[-1]
    subfolder_c = name.split(config_name)[-2]
    save_model_name = f'{config_name}_{ite}'
    save_model_path = f'{args.result_path}/{subfolder_c}val_models/{save_model_name}.pkl'
    save_proto_name = save_model_name + '_prototypes'
    save_proto_path = f'{args.result_path}/{subfolder_c}val_models/{save_proto_name}.pkl'
    if not valid:
        lms = pickle.load(open(save_model_path, 'rb'))
        prototypes_ = pickle.load(open(save_proto_path, 'rb'))

    for task in range(args.n_tasks):
        pro_cls = {}
        subgraph, ids_per_cls, [train_ids, valid_ids, test_ids] = pickle.load(
            open(f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_as{args.n_agents}_t{task}_p{args.par}.pkl', 'rb'))
        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()

        for rnd in range(args.n_rnds):
            for a_id in range(args.n_agents):
                if task == 0 and rnd == 0 and valid:
                    if args.bs == -1:
                        bs = None
                    else:
                        bs = args.bs
                    lms[a_id].pretrain(args, subgraph, features, bs)

                if valid:
                    for epoch in range(epochs):
                        principal_list[a_id], principal_list_as[a_id], similarity_as[a_id], local_train_sizes[a_id], \
                        local_train_sizes_cls[a_id] = lms[a_id].observe_il(subgraph, features,
                                                                           labels - n_agents_cls_per_task * task, task,
                                                                           train_ids[a_id], n_agents_cls_per_task, lms,
                                                                           graph_matrix[a_id], a_id,
                                                                           fea_syn_as_as[a_id], g_syn_as_as[a_id],
                                                                           labels_syn_as_as[a_id], rnd, args)
                    torch.cuda.empty_cache()

                if rnd == args.n_rnds - 1:
                    if valid:
                        prototypes_task = []
                        for aid in range(args.n_agents):
                            prototypes_task.append(
                                lms[a_id].getprototype(g_syn_as_as[a_id][aid][task], fea_syn_as_as[a_id][aid][task]))
                        prototypes_[task] = torch.nn.functional.normalize(torch.cat(prototypes_task, dim=0)).mean(0)

                    acc_mean = []
                    for t in range(task + 1):
                        subgraph, ids_per_cls, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
                            f'{args.data_path}/no_inter_tsk_edge/{args.dataset}_as{args.n_agents}_t{t}_p{args.par}.pkl',
                            'rb'))
                        subgraph = subgraph.to(device='cuda:{}'.format(args.gpu))
                        features, labels = subgraph.srcdata['feat'], subgraph.dstdata['label'].squeeze()
                        # test_ids = valid_ids_ if valid else test_ids_
                        test_ids = test_ids_
                        print('test val', len(test_ids_), len(valid_ids_))
                        ids_per_cls_test = [list(set(ids).intersection(set(test_ids))) for ids in ids_per_cls]
                        if task > 0:
                            taskid = lms[a_id].gettaskid(prototypes_, subgraph, features, task + 1, test_ids)
                        else:
                            taskid = 0
                        print('task id (predict real)', taskid, t)
                        output = lms[a_id].getpred(subgraph, features, taskid)
                        acc = evaluatewp(output, labels - args.n_cls_task * t, test_ids, cls_balance=args.cls_balance,
                                         ids_per_cls=ids_per_cls_test)
                        acc_matrix_as[a_id][task][t] = round(acc * 100, 2)
                        acc_mean.append(acc)
                        print(f"a{a_id} c{rnd} T{t:02d} {acc * 100:.2f}|", end="")
                    acc_mean_as[a_id] = round(np.mean(acc_mean) * 100, 2)
                    print()
                pro_cls[a_id] = {
                    'prompt': get_state_dict(lms[a_id].prompts[task - 1]),
                    'classification': get_state_dict(lms[a_id].classifications[task])
                }
            if rnd == 0 and valid:
                topo_fea_emb = lms[0].getprototype(subgraph, features)
                for a_id in range(args.n_agents):
                    topo_fea_as[a_id] = topo_fea_emb[train_ids[a_id]].mean(0)
                    topofeas = []
                    for cls in range(n_agents_cls_per_task):
                        index = list(
                            set(((labels - n_agents_cls_per_task * task) == cls).nonzero().view(-1).tolist()) & set(
                                train_ids[a_id]))
                        if len(index) > 0:
                            topofeas.append(topo_fea_emb[index].mean(0))
                        else:
                            topofeas.append([])
                    topo_fea_cls_as[a_id] = topofeas

            if rnd < args.n_rnds - 1 and valid:
                model_complementary_matrix = cal_complementary(args.n_agents, principal_list)
                print('complementary***********************')
                print(model_complementary_matrix)
                model_difference_matrix = cal_model_cosine_difference(args.n_agents, task, pro_cls, topo_fea_as)
                print('difference ***********************')
                print(model_difference_matrix)
                ratio = (np.array(local_train_sizes) / np.sum(local_train_sizes)).tolist()
                graph_matrix = optimizing_graph_matrix_neighbor(args.n_agents, graph_matrix,
                                                                model_complementary_matrix, model_difference_matrix,
                                                                args.w_c, args.w_s, ratio)
                print('all***********************')
                print(graph_matrix)
                graph_matrix = torch.tensor(graph_matrix)
                graph_matrix_cls_cs = []
                for cls in range(n_agents_cls_per_task):
                    model_complementary_matrix_cls = cal_complementary_cls(args.n_agents,
                                                                           np.array(principal_list_as)[:, cls])
                    print('complementary***********************', cls)
                    print(model_complementary_matrix_cls)
                    model_difference_matrix_cls = cal_model_cosine_difference_cls(args.n_agents,
                                                                                  np.array(similarity_as)[:, cls],
                                                                                  np.array(topo_fea_cls_as)[:, cls])
                    print('difference ***********************', cls)
                    print(model_difference_matrix_cls)
                    ratio_cls_ = (np.array(np.array(local_train_sizes_cls)[:, cls].squeeze()) / np.sum(
                        np.array(local_train_sizes_cls)[:, cls].squeeze())).tolist()
                    graph_matrix_cls_ = torch.ones(args.n_agents, args.n_agents)
                    graph_matrix_cls_ = optimizing_graph_matrix_neighbor(args.n_agents, graph_matrix_cls_,
                                                                         model_complementary_matrix_cls,
                                                                         model_difference_matrix_cls,
                                                                         args.w_c, args.w_s, ratio_cls_)
                    print('all***********************', cls)
                    print(graph_matrix_cls_)
                    graph_matrix_cls_cs.append(graph_matrix_cls_)
                for i in range(args.n_agents):
                    for j in range(args.n_agents):
                        print(f'agent{j} for agent{i} syn data ......................................')
                        ratio_cls_orig = (
                                    np.array(local_train_sizes_cls[j]) / np.sum(local_train_sizes_cls[j])).tolist()
                        mij = [args.w_col * graph_matrix_cls_cs[cls][i][j] + ratio_cls_orig[cls] for cls in
                               range(n_agents_cls_per_task)]
                        for c in range(n_agents_cls_per_task):
                            if ratio_cls_orig[c] == 0:
                                mij[c] = 0
                        mij = [item / sum(mij) for item in mij]

                        fea_syn_as_as[i][j][task], g_syn_as_as[i][j][task], labels_syn_as_as[i][j][task] = lms[
                            j].syn_init_cls(copy.deepcopy(fea_syn_as_as[i][j][task]), labels_syn_as_as[i][j][task], rnd,
                                            mij, labels - n_agents_cls_per_task * task, train_ids[j], features,
                                            n_agents_cls_per_task, args)
                        fea_syn_as_as[i][j][task], g_syn_as_as[i][j][task] = lms[j].getsyn(subgraph, features,
                                                                                           train_ids[j],
                                                                                           labels - n_agents_cls_per_task * task,
                                                                                           n_agents_cls_per_task,
                                                                                           args)

    if valid:
        mkdir_if_missing(f'{args.result_path}/{subfolder_c}/val_models')
        with open(save_model_path, 'wb') as f:
            pickle.dump(lms, f)
        with open(save_proto_path, 'wb') as f:
            pickle.dump(prototypes_, f)

    final_A, final_AP, final_AF = [], [], []
    for a_id in range(args.n_agents):
        backward = []
        for t in range(args.n_tasks - 1):
            b = acc_matrix_as[a_id][args.n_tasks - 1][t] - acc_matrix_as[a_id][t][t]
            backward.append(round(b, 2))
        mean_backward = round(np.mean(backward), 2)
        final_AP.append(acc_mean_as[a_id])
        final_AF.append(mean_backward)
        tri_acc = []
        for t in range(args.n_tasks):
            tri_acc.append(acc_matrix_as[a_id][t][t])
        final_A.append(np.mean(tri_acc))
        print('AP AF: ', acc_mean_as[a_id], mean_backward)
    final_A_ = np.mean(final_A)
    final_AP_ = np.mean(final_AP)
    final_AF_ = np.mean(final_AF)
    max_A_ = np.max(final_A)
    max_AP_ = np.max(final_AP)
    max_AF_ = np.max(final_AF)
    min_A_ = np.min(final_A)
    min_AP_ = np.min(final_AP)
    min_AF_ = np.min(final_AF)
    print('final AP AF: ', final_AP_, final_AF_)
    print('max AP AF: ', max_AP_, max_AF_)
    print('min AP AF: ', min_AP_, min_AF_)

    return final_A_, final_AP_, final_AF_, final_A, final_AP, final_AF
