import random
import numpy as np
import torch
import dgl
import os
import copy
import errno
import cvxpy as cp


def assign_hyp_param(args, params):
    if args.method == 'lwf':
        args.lwf_args = params
    if args.method == 'bare':
        args.bare_args = params
    if args.method == 'gem':
        args.gem_args = params
    if args.method == 'ewc':
        args.ewc_args = params
    if args.method == 'mas':
        args.mas_args = params
    if args.method == 'twp':
        args.twp_args = params
    if args.method in ['jointtrain', 'joint', 'Joint']:
        args.joint_args = params
    if args.method == 'ergnn':
        args.ergnn_args = params
    if args.method == 'tpp':
        args.tpp_args = params


def str2dict(s):
    # accepts a str like " 'k1':v1; ...; 'km':vm ", values (v1,...,vm) can be single values or lists (for hyperparameter tuning)
    output = dict()
    kv_pairs = s.replace(' ', '').replace("'", '').split(';')
    for kv in kv_pairs:
        key = kv.split(':')[0]
        v_ = kv.split(':')[1]
        if '[' in v_:
            # transform list of values
            v_list = v_.replace('[', '').replace(']', '').split(',')
            vs = []
            for v__ in v_list:
                try:
                    # if the parameter is float
                    vs.append(float(v__))
                except:
                    # if the parameter is str
                    vs.append(str(v__))
            output.update({key: vs})
        else:
            try:
                output.update({key: float(v_)})
            except:
                output.update({key: str(v_)})
    return output


def compose_hyper_params(hyp_params):
    hyp_param_list = [{}]
    for hk in hyp_params:
        hyp_param_list_ = []
        hyp_p_current = hyp_params[hk] if isinstance(hyp_params[hk], list) else [hyp_params[hk]]
        for v in hyp_p_current:
            for hk_ in hyp_param_list:
                hk__ = copy.deepcopy(hk_)
                hk__.update({hk: v})
                hyp_param_list_.append(hk__)
        hyp_param_list = hyp_param_list_
    return hyp_param_list


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def set_seed(args=None):
    seed = 1 if not args else args.seed

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)


def remove_illegal_characters(name, replacement='_'):
    # replace any potential illegal characters with 'replacement'
    for c in ['-', '[', ']', '{', '}', "'", ',', ':', ' ']:
        name = name.replace(c, replacement)
    return name


def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)
    torch.save(data, fpath)


def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)
    return torch.load(fpath, map_location=torch.device('cpu'))


from collections import defaultdict, OrderedDict


def convert_np_to_tensor(state_dict, gpu_id, skip_stat=False, skip_mask=False, model=None):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        if skip_stat:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue
        if skip_mask:
            if 'mask' in k or 'pre' in k or 'pos' in k:
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = torch.tensor(v).cuda(gpu_id)
        else:
            _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id)
    return _state_dict


def get_state_dict(model):
    # state_dict = model.state_dict()
    state_dict = convert_tensor_to_np(model.state_dict())
    return state_dict


def set_state_dict(model, state_dict, gpu_id, skip_stat=False, skip_mask=False):
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip_stat=skip_stat, skip_mask=skip_mask,
                                      model=model.state_dict())
    model.load_state_dict(state_dict)


def convert_tensor_to_np(state_dict):
    return OrderedDict([(k, v.clone().detach().cpu()) for k, v in state_dict.items()])
    # return OrderedDict([(k, v.clone().detach().cpu().numpy()) for k, v in state_dict.items()])


def weight_flatten_all_sep(model1, model2):
    params = []
    for k in model1:
        params.append(model1[k].reshape(-1))
    if model2 is not None:
        for k in model2:
            params.append(model2[k].reshape(-1))
    params = torch.cat(params)
    return params






def compute_principal_angles(A, B):
    assert A.shape[0] == B.shape[0], "A and B must have the same number of vectors"

    k = A.shape[0]
    norm_A = np.linalg.norm(A, axis=1)[:, np.newaxis]
    norm_B = np.linalg.norm(B, axis=1)
    dot_product = np.dot(A, B.T)
    cosine_matrix = dot_product / (norm_A * norm_B)
    cos_phi_values = []

    for _ in range(k):
        i, j = np.unravel_index(np.argmax(cosine_matrix, axis=None), cosine_matrix.shape)
        cos_phi_values.append(cosine_matrix[i, j])
        cosine_matrix[i, :] = -np.inf
        cosine_matrix[:, j] = -np.inf
    phi = np.arccos(np.clip(cos_phi_values, -1, 1))

    return phi


def cal_complementary(n_agents, principal_list):
    model_complementary_matrix = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i, n_agents):
            k = principal_list[i].shape[0]
            phi = compute_principal_angles(principal_list[i], principal_list[j])
            principal_angle = np.cos((1 / k) * np.sum(phi))
            model_complementary_matrix[i][j] = principal_angle
            model_complementary_matrix[j][i] = principal_angle
    return model_complementary_matrix


def cal_complementary_cls(n_agents, principal_list):
    model_complementary_matrix = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i, n_agents):
            if i == j:
                model_complementary_matrix[i][j] = 1
                model_complementary_matrix[j][i] = 1
            elif principal_list[i] == [] and principal_list[j] != []:
                model_complementary_matrix[i][j] = 0
                model_complementary_matrix[j][i] = 1
            elif principal_list[i] != [] and principal_list[j] == []:
                model_complementary_matrix[i][j] = 1
                model_complementary_matrix[j][i] = 0
            elif principal_list[i] == [] and principal_list[j] == []:
                model_complementary_matrix[i][j] = 1
                model_complementary_matrix[j][i] = 1
            else:
                k = len(principal_list[i])
                phi = compute_principal_angles(principal_list[i], principal_list[j])
                principal_angle = np.cos((1 / k) * np.sum(phi))
                model_complementary_matrix[i][j] = principal_angle
                model_complementary_matrix[j][i] = principal_angle
    return model_complementary_matrix


def cal_model_cosine_difference(n_agents, task, pro_cls, topo_fea_as):
    model_similarity_matrix = torch.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i, n_agents):
            if task > 0:
                parai = weight_flatten_all_sep(pro_cls[i]['classification'].copy(), pro_cls[i]['prompt'].copy())
                paraj = weight_flatten_all_sep(pro_cls[j]['classification'].copy(), pro_cls[i]['prompt'].copy())
            else:
                parai = weight_flatten_all_sep(pro_cls[i]['classification'].copy(), None)
                paraj = weight_flatten_all_sep(pro_cls[j]['classification'].copy(), None)
            embi = torch.cat([parai, topo_fea_as[i].cpu()])
            embj = torch.cat([paraj, topo_fea_as[j].cpu()])
            diff = - torch.nn.functional.cosine_similarity(embi.unsqueeze(0), embj.unsqueeze(0))
            model_similarity_matrix[i, j] = diff
            model_similarity_matrix[j, i] = diff
    return model_similarity_matrix


def cal_model_cosine_difference_cls(n_agents, outputs_as, topo_fea_cls_as):
    model_similarity_matrix = torch.zeros((n_agents, n_agents))
    for i in range(n_agents):
        for j in range(i, n_agents):
            if i == j:
                model_similarity_matrix[i, j] = -1
                model_similarity_matrix[j, i] = -1
            elif outputs_as[i] == [] and outputs_as[j] != []:
                model_similarity_matrix[i, j] = 0
                model_similarity_matrix[j, i] = 0
            elif outputs_as[i] != [] and outputs_as[j] == []:
                model_similarity_matrix[i, j] = 0
                model_similarity_matrix[j, i] = 0
            elif outputs_as[i] == [] and outputs_as[j] == []:
                model_similarity_matrix[i, j] = 0#-1
                model_similarity_matrix[j, i] = 0#-1
            else:
                embi = torch.cat([outputs_as[i].cpu(), topo_fea_cls_as[i].cpu()])
                embj = torch.cat([outputs_as[j].cpu(), topo_fea_cls_as[j].cpu()])
                diff = - torch.nn.functional.cosine_similarity(embi.unsqueeze(0), embj.unsqueeze(0))
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
    return model_similarity_matrix


def optimizing_graph_matrix_neighbor(n_agents, graph_matrix, model_complementary_matrix, model_difference_matrix,
                                     w_c, w_s, ratio):
    n = model_difference_matrix.shape[0]
    p = np.array(ratio)
    P = np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_complementary_vector = model_complementary_matrix[i]
        model_difference_vector = model_difference_matrix[i]
        s = model_difference_vector.numpy()
        c = model_complementary_vector
        q = w_c * c + w_s * s - 2 * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                          [G @ x <= h,
                           A @ x == b]
                          )
        prob.solve()

        graph_matrix[i, list(range(n_agents))] = torch.Tensor(x.value)
    return graph_matrix



