import torch
import copy
from torch import Tensor
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn.inits import glorot
from Backbones.gnns import SGC_Agg
from Baselines.grace import ModelGrace, traingrace, LogReg
import dgl
from sklearn.decomposition import PCA


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        return x + p

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.fc(x)
        return x


class NET(torch.nn.Module):
    def __init__(self, model, args):
        super(NET, self).__init__()
        self.n_tasks = args.n_tasks
        self.model = model
        self.drop_edge = args.tpp_args['pe']
        self.drop_feature = args.tpp_args['pf']
        self.n_agents = args.n_agents
        num_promt = int(args.tpp_args['prompts'])
        if num_promt < 2:
            prompt = SimplePrompt(args.d_data).to(device='cuda:{}'.format(args.gpu))
        else:
            prompt = GPFplusAtt(args.d_data, num_promt).to(device='cuda:{}'.format(args.gpu))

        cls_head = LogReg(args.hidden, args.n_cls_task).to(device='cuda:{}'.format(args.gpu))
        self.classifications = ModuleList([copy.deepcopy(cls_head) for _ in range(args.n_tasks)])
        self.prompts = ModuleList([copy.deepcopy(prompt) for _ in range(args.n_tasks - 1)])

        self.optimizers = []
        for taskid in range(args.n_tasks):
            model_param_group = []
            if taskid == 0:
                model_param_group.append({"params": self.classifications[taskid].parameters()})
            else:
                model_param_group.append({"params": self.prompts[taskid - 1].parameters()})
                model_param_group.append({"params": self.classifications[taskid].parameters()})
            self.optimizers.append(torch.optim.Adam(model_param_group, lr=args.lr, weight_decay=args.weight_decay))
        self.ce = torch.nn.functional.cross_entropy
        self.mse = torch.nn.MSELoss()

    def getprototype(self, g, features, train_ids=None, k=3):
        neighbor_agg = SGC_Agg(k=k)
        features = neighbor_agg(g, features)
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(features.device).unsqueeze(1)
        features = features * norm
        if train_ids is None:
            prototype = features
        else:
            prototype = torch.mean(features[train_ids], dim=0)
        return prototype



    def syn_init_cls(self, fea_syn_old, labels_syn_old, rnd, mij, labels, train_ids, features, n_agents_cls_per_task, args):
        dnodes_syn = features.shape[1]
        import random
        random.seed(42)
        sampled_ids = []
        labels_syn = []
        self.coeff_syn = []
        self.coeff_cur = []
        nnodes_syn_cls_all = [0 for _ in range(n_agents_cls_per_task)]


        syn_nodes = max(1,int(len(train_ids) * args.compression_ratio))
        remain_syn_nodes = syn_nodes
        sort_mij_index = sorted(range(n_agents_cls_per_task), key=lambda k: mij[k], reverse=False)
        for cls in sort_mij_index:
            index = list(set((labels == cls).nonzero().view(-1).tolist()) & set(train_ids))
            if len(index) > 0 and mij[cls] > 0 and remain_syn_nodes > 0:
                nnodes_syn_cls = max(1, int(mij[cls] * len(train_ids) * args.compression_ratio))
                if nnodes_syn_cls >= len(index):
                    nnodes_syn_cls = len(index)
                    mij[cls] = 0
                if remain_syn_nodes < nnodes_syn_cls:
                    nnodes_syn_cls = remain_syn_nodes
                remain_syn_nodes -= nnodes_syn_cls
                nnodes_syn_cls_all[cls] = nnodes_syn_cls
            else:
                nnodes_syn_cls_all[cls] = 0
        remain = remain_syn_nodes
        while remain > 0:
            mij = [item / sum(mij) for item in mij]
            sort_mij_index = sorted(range(n_agents_cls_per_task), key=lambda k: mij[k], reverse=False)
            for cls in sort_mij_index:
                index = list(set((labels == cls).nonzero().view(-1).tolist()) & set(train_ids))
                if len(index) > 0 and mij[cls] > 0:
                    nnodes_syn_cls = max(1, int(mij[cls] * remain))
                    if nnodes_syn_cls + nnodes_syn_cls_all[cls] >= len(index):
                        nnodes_syn_cls = len(index) - nnodes_syn_cls_all[cls]
                        mij[cls] = 0
                    if remain < nnodes_syn_cls:
                        nnodes_syn_cls = remain
                    remain -= nnodes_syn_cls
                    nnodes_syn_cls_all[cls] += nnodes_syn_cls
                if remain <= 0:
                    break
        nnodes_syn = sum(nnodes_syn_cls_all)
        self.feat_syn = nn.Parameter(torch.FloatTensor(nnodes_syn, dnodes_syn).to(device='cuda:{}'.format(args.gpu)))
        if rnd==0:
            for cls in range(n_agents_cls_per_task):
                index = list(set((labels == cls).nonzero().view(-1).tolist()) & set(train_ids))
                if len(index) > 0:
                    sampled_ids.extend(random.sample(index, k=nnodes_syn_cls_all[cls]))
                    labels_syn += [cls] * nnodes_syn_cls_all[cls]
                    self.coeff_syn.append(nnodes_syn_cls_all[cls])
                    self.coeff_cur.append(len(index))
                else:
                    self.coeff_syn.append(0)
                    self.coeff_cur.append(0)
            self.feat_syn.data.copy_(features[sampled_ids])
        else:
            fea_syn_ini = []
            for cls in range(n_agents_cls_per_task):
                index = list(set((labels == cls).nonzero().view(-1).tolist()) & set(train_ids))
                index_syn_old = (labels_syn_old == cls).nonzero().view(-1).tolist()
                if len(index) > 0:
                    if len(index_syn_old)<nnodes_syn_cls_all[cls]:
                        sample_new = random.sample(index, k=nnodes_syn_cls_all[cls]-len(index_syn_old))
                        fea_syn_ini.append(fea_syn_old[index_syn_old])
                        fea_syn_ini.append(features[sample_new])
                    else:
                        sample_old = random.sample(index_syn_old, k=nnodes_syn_cls_all[cls])
                        fea_syn_ini.append(fea_syn_old[sample_old])
                    labels_syn += [cls] * nnodes_syn_cls_all[cls]
                    self.coeff_syn.append(nnodes_syn_cls_all[cls])
                    self.coeff_cur.append(len(index))
                else:
                    self.coeff_syn.append(0)
                    self.coeff_cur.append(0)
            self.feat_syn.data.copy_(torch.cat(fea_syn_ini, dim=0))

        self.labels_syn = torch.LongTensor(labels_syn).to(device='cuda:{}'.format(args.gpu))
        u_v = []
        for u, v in zip(list(range(nnodes_syn)), list(range(nnodes_syn))):
            u_v.append((u, v))
        self.g_syn = dgl.graph(u_v).to(device='cuda:{}'.format(args.gpu))
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        return self.feat_syn.detach(), self.g_syn, self.labels_syn


    def getsyn(self, g, features, train_ids, labels, n_agents_cls_per_task, args):
        for _ in range(args.syn_epochs):
            # other datasets
            # emb_syn = F.normalize(self.getprototype(self.g_syn.to(device='cuda:{}'.format(args.gpu)), self.feat_syn))
            # emb_real = F.normalize(self.getprototype(g, features))

            # corafull and arxiv
            emb_syns = [self.feat_syn]
            emb_reals = [features]
            for k_ in range(1, 4):
                emb_syns.append(self.getprototype(self.g_syn.to(device='cuda:{}'.format(args.gpu)), self.feat_syn, k=k_))
                emb_reals.append(self.getprototype(g, features, k=k_))
            emb_syn = F.normalize(torch.cat(emb_syns, dim=1))
            emb_real = F.normalize(torch.cat(emb_reals, dim=1))

            coeff_syn_cur = [self.coeff_cur[i] ** 2 / (self.coeff_syn[i] + 1e-10) for i in range(n_agents_cls_per_task)]
            coeff = [item / max(coeff_syn_cur) for item in coeff_syn_cur]
            loss = torch.tensor(0.).to(device='cuda:{}'.format(args.gpu))

            for cls in range(n_agents_cls_per_task):
                index = list(set((labels == cls).nonzero().view(-1).tolist()) & set(train_ids))
                index_syn = (self.labels_syn == cls).nonzero().view(-1).tolist()
                if len(index) > 0 and len(index_syn) > 0:
                    real_emb_at_class = emb_real[index].view(len(index), -1)
                    syn_emb_at_class = emb_syn[index_syn].view(len(index_syn), -1)
                    dist = torch.mean(real_emb_at_class, 0) - torch.mean(syn_emb_at_class, 0)
                    loss += torch.sum(dist ** 2) * coeff[cls]

                    if len(index_syn) > 1:
                        dist2 = torch.std(real_emb_at_class, 0) - torch.std(syn_emb_at_class, 0)
                        loss += torch.sum(dist2 ** 2) * coeff[cls] * args.w_sigma

            self.optimizer_feat.zero_grad()
            loss.backward()
            self.optimizer_feat.step()

        return self.feat_syn.detach(), self.g_syn

    def gettaskid(self, prototypes, g, features, task, test_ids, k=3):
        neighbor_agg = SGC_Agg(k=k)
        features = neighbor_agg(g, features)
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(features.device).unsqueeze(1)
        features = F.normalize(features * norm)
        testprototypes = torch.mean(features[test_ids], dim=0).cpu()
        dist = torch.norm(prototypes[0:task] - testprototypes, dim=1)
        _, taskid = torch.min(dist, dim=0)
        return taskid.numpy()

    def pretrain(self, args, g, features, batch_size=None):
        num_hidden = args.hidden
        num_proj_hidden = 2 * num_hidden
        gracemodel = ModelGrace(self.model, num_hidden, num_proj_hidden, tau=0.5).to(device='cuda:{}'.format(args.gpu))
        traingrace(gracemodel, g, features, batch_size, drop_edge_prob=self.drop_edge,
                   drop_feature_prob=self.drop_feature)

    def observe_il(self, g, features, labels, t, train_ids, n_agents_cls_per_task, lms, weights, a_self,
                   fea_syn_as, g_syn_as, labels_syn_as, rnd, args):
        label_frep, label_current = [], []
        label_freq_all = []
        local_train_size = len(train_ids)
        for cls in range(n_agents_cls_per_task):
            index_s = list(set((labels == cls).nonzero().view(-1).tolist()) & set(train_ids))
            label_current.append(len(index_s))
            index_syns = []
            for aid in range(self.n_agents):
                index_syn = torch.tensor(labels_syn_as[aid][t] == cls).nonzero().view(-1).tolist()
                index_syns.extend(index_syn)
            label_frep.append(len(index_syns))
            label_freq_all.append(len(index_s) + len(index_syns))

        loss_w_ = [1. / max(i, 1) for i in label_freq_all]
        loss_w_ = torch.tensor(loss_w_).to(device='cuda:{}'.format(args.gpu))


        self.model.eval()
        cls_head = self.classifications[t]
        cls_head.train()
        cls_head.zero_grad()
        optimizer_t = self.optimizers[t]
        if t > 0:
            prompt_t = self.prompts[t - 1]
            prompt_t.train()
            prompt_t.zero_grad()
            features = prompt_t.add(features)
        output = self.model(g, features)
        output = cls_head(output)


        loss2 = 0
        output_self_as = []
        labels_syn_as_ = []

        if rnd > 0:
            for aid in range(self.n_agents):
                if aid != a_self:
                    fea_syn_a = fea_syn_as[aid][t]
                    fea_syn_self = fea_syn_as[aid][t]
                    if t > 0:
                        fea_syn_a = lms[aid].prompts[t - 1].add(fea_syn_a)
                        fea_syn_self = self.prompts[t - 1].add(fea_syn_self)
                    lms[aid].model.eval()
                    output_a = lms[aid].classifications[t](lms[aid].model(g_syn_as[aid][t], fea_syn_a))
                    output_self = self.classifications[t](self.model(g_syn_as[aid][t], fea_syn_self))
                    klloss = weights[aid] * F.kl_div(F.log_softmax(output_self / 1.0, dim=-1),
                                                     F.softmax(output_a.detach() / 1.0, dim=-1), reduction='batchmean')
                    output_self_as.append(output_self)
                    labels_syn_as_.append(labels_syn_as[aid][t].long().to(device='cuda:{}'.format(args.gpu)))
                    loss2 += klloss
                else:
                    fea_syn_self = fea_syn_as[a_self][t]
                    if t > 0:
                        fea_syn_self = self.prompts[t - 1].add(fea_syn_self)
                    output_self = self.classifications[t](self.model(g_syn_as[a_self][t], fea_syn_self))
                    output_self_as.append(output_self)
                    labels_syn_as_.append(labels_syn_as[a_self][t].long().to(device='cuda:{}'.format(args.gpu)))


        if output_self_as != []:
            output_self_as.append(output[train_ids])
            labels_syn_as_.append(labels[train_ids])
            final_output = torch.cat(output_self_as, dim=0)
            final_label = torch.cat(labels_syn_as_)
        else:
            final_output = output[train_ids]
            final_label = labels[train_ids]



        loss = self.ce(final_output, final_label, weight=loss_w_)
        loss += args.w_kl * loss2
        loss.backward()
        optimizer_t.step()

        self.pca = PCA(n_components=args.pca)
        self.pca.fit_transform(copy.deepcopy(final_output.detach().cpu().numpy()))
        orthogonal_basis = self.pca.components_
        orthogonal_basis_clss = []
        output_clss = []
        for cls in range(n_agents_cls_per_task):
            pca = PCA(n_components=1)
            index_s = (final_label == cls).nonzero().view(-1).tolist()
            if len(index_s)>0:
                pca.fit_transform(copy.deepcopy(final_output[index_s].view(len(index_s),-1).detach().cpu().numpy()))
                orthogonal_basis_clss.append(pca.components_)
                output_clss.append(copy.deepcopy(final_output[index_s].view(len(index_s),-1).detach().mean(0)))
            else:
                orthogonal_basis_clss.append(copy.deepcopy([]))
                output_clss.append(copy.deepcopy([]))

        return orthogonal_basis, orthogonal_basis_clss, output_clss, local_train_size, label_current


    def getpred(self, g, features, taskid):
        self.model.eval()
        if taskid != 0:
            prompt_t = self.prompts[taskid - 1]
            prompt_t.eval()
            features = prompt_t.add(features)
        output = self.model(g, features)
        cls_head = self.classifications[taskid]
        cls_head.eval()
        output = cls_head(output)
        return output
