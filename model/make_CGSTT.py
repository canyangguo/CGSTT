import torch
import torch.nn as nn
from model.basic_module import weight_and_initial, glu, matrix_decomposition, CMD
import torch.nn.functional as F


class spatial_graph_construct(nn.Module):
    def __init__(self, source_node, d_model, times, num_of_patterns, drop_rate=0.1):
        super(spatial_graph_construct, self).__init__()
        self.d_model = d_model
        self.num_of_patterns = num_of_patterns
        self.weights = weight_and_initial(d_model, d_model * 2, times, bias=False)
        self.dr = nn.Dropout(drop_rate)

    def forward(self, x):
        weights = self.weights().reshape(-1, self.d_model, self.d_model * 2)
        qks = torch.einsum('kcd, nc -> knd', weights, torch.mean(x, dim=(0, 1)))  # knd

        q, k = torch.chunk(qks, 2, dim=-1)
        score = torch.einsum('kmd, knd -> kmn', q, k) / (self.d_model ** 0.5)
        score = torch.softmax(score, dim=-1)
        score = self.dr(score)

        return score


class dynamic_spatial_graph_convolution(nn.Module):
    def __init__(self, source_node, d_model, num_of_latents, num_of_times, num_of_days, drop_rate=0.1):  # 注意
        super(dynamic_spatial_graph_convolution, self).__init__()
        self.sta_adj = spatial_graph_construct(source_node, d_model, 1, num_of_latents, drop_rate)
        self.t_adjs = spatial_graph_construct(source_node, d_model, num_of_times, num_of_latents, drop_rate)
        self.glu = glu(d_model)

    def forward(self, x, ep, S2D, ti):
        B, T, N, C = x.shape
        sta_adj = self.sta_adj(x).reshape(N, N)
        if ep < S2D:
            x = torch.einsum('mn, btnc -> btmc', sta_adj, x)
        else:
            t_adj = self.t_adjs(x).reshape(-1, N, N)[ti]
            adj = t_adj / 2 + sta_adj / 2

            x = torch.einsum('bmn, btnc -> btmc', adj, x)
        x = self.glu(x)
        return x


class dynamic_temporal_graph_convolution(nn.Module):
    def __init__(self, d_model, input_length, output_length, num_of_times, num_of_days, source_node_num,
                 num_of_patterns):
        super(dynamic_temporal_graph_convolution, self).__init__()
        self.sta_adj = weight_and_initial(input_length, output_length, bias=None)
        self.t_adjs = weight_and_initial(input_length, output_length, num_of_times, bias=None)
        self.glu = glu(d_model)

    def forward(self, x, ep, tdx, S2D, ti):
        sta_adj = self.sta_adj()

        if ep < S2D:
            x = torch.einsum('pq, bqnc -> bpnc', sta_adj[: tdx], x)
        else:
            t_adj = self.t_adjs()[ti]
            adj = t_adj / 2 + sta_adj / 2
            x = torch.einsum('bpq, bqnc -> bpnc', adj[:, :tdx], x)
        x = self.glu(x)
        return x


class private_module(nn.Module):
    def __init__(self, d_model, input_length, output_length, num_of_latent, num_of_times, num_of_days, num_of_nodes,
                 drop_rate=0.8):
        super(private_module, self).__init__()

        self.d_model = d_model
        self.s_adjs = matrix_decomposition(num_of_nodes, num_of_nodes, d_model // 2, num_of_times + 1)
        self.drop = nn.Dropout(drop_rate)
        self.glu1 = glu(d_model)

        self.t_adjs = weight_and_initial(input_length, output_length, num_of_times + 1, bias=None)
        self.glu2 = glu(d_model)

    def forward(self, x, ti, ep, tdx, S2D):

        if ep < S2D:
            s_adj = self.s_adjs()[-1]
            x = self.glu1(torch.einsum('mn, btnc -> btmc', self.drop(s_adj), x)) + x
            t_adj = self.t_adjs()[-1, : tdx]
            x = torch.cat((self.glu2(torch.einsum('pq, bqnc -> bpnc', t_adj, x)) + x[:, :tdx], x[:, tdx:]), dim=1)
        else:
            s_adjs = (self.s_adjs()[ti] + self.s_adjs()[-1]) / 2
            x = self.glu1(torch.einsum('bmn, btnc -> btmc', self.drop(s_adjs), x)) + x

            t_adjs = (self.t_adjs()[ti] + self.t_adjs()[-1]) / 2
            x = torch.cat((self.glu2(torch.einsum('bpq, bqnc -> bpnc', t_adjs, x)) + x[:, :tdx], x[:, tdx:]), dim=1)

        return x


class st_module(nn.Module):
    def __init__(self, d_model, input_length, output_length, num_of_latents, num_of_times, num_of_days,
                 source_node_num):
        super(st_module, self).__init__()
        self.psgcn = dynamic_spatial_graph_convolution(source_node_num, d_model, num_of_latents, num_of_times,
                                                       num_of_days)
        self.ptgcn = dynamic_temporal_graph_convolution(d_model, input_length, output_length, num_of_times, num_of_days,
                                                        source_node_num, num_of_latents)

    def forward(self, x, ti, ni, ep, tdx, S2D):
        x = self.psgcn(x, ep, S2D, ti) + x
        x = torch.cat((self.ptgcn(x, ep, tdx, S2D, ti) + x[:, :tdx], x[:, tdx:]), dim=1)
        return x


class gated_layer(nn.Module):
    def __init__(self, d_model):
        super(gated_layer, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, x1, x2):
        g = torch.sigmoid(self.fc1(x1) + self.fc2(x2))
        return g * x1 + (1 - g) * x2


class mf_based_gnn(nn.Module):
    def __init__(self, d_model1, d_model2):
        super(mf_based_gnn, self).__init__()
        self.fc = nn.Linear(d_model1, d_model2)

    def forward(self, x, adj):
        x = torch.einsum('mn, btmc -> btnc', adj, x)
        x = self.fc(x)
        x = torch.relu(x)
        return x


class diff_pool(nn.Module):
    def __init__(self, d_model, num_of_node, idx):
        super(diff_pool, self).__init__()
        self.s = matrix_decomposition(num_of_node // (4 ** idx), num_of_node // (4 ** (idx + 1)), d_model // 2)
        self.glu = glu(d_model)
        # self.mf_gnn_emb = mf_based_gnn(d_model, d_model)
        # self.mf_gnn_pool = mf_based_gnn(d_model, num_of_node//(3**idx))
        # self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        s = torch.softmax(self.s(), dim=-1)

        x = torch.einsum('mn, btmc -> btnc', s, x)
        x = self.glu(x)
        ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()
        return x, s, ent_loss


class start_make_model(nn.Module):
    def __init__(self, config_args, args, mean_std):
        super(start_make_model, self).__init__()
        model_args, task_args, data_args = config_args['model'], config_args['task'], config_args['data']
        self.mean_std = mean_std
        self.emb = nn.Linear(model_args['num_of_features'], model_args['d_model'])
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2

        self.st_models = nn.ModuleList([
            st_module(model_args['d_model'], task_args['his_num'], task_args['pred_num'], args.num_of_latent,
                      data_args[args.source_dataset]['num_of_times'], data_args[args.source_dataset]['num_of_days'],
                      data_args[args.source_dataset]['node_num'])
            for _ in range(model_args['num_of_layers'])
        ])

        self.re_emb_s = nn.Linear(model_args['d_model'], model_args['num_of_features'])
        self.var_layer1 = private_module(model_args['d_model'], task_args['his_num'], task_args['pred_num'],
                                         args.num_of_latent,
                                         data_args[args.target_dataset]['num_of_times'],
                                         data_args[args.target_dataset]['num_of_days'],
                                         data_args[args.target_dataset]['node_num'])
        self.var_layer2 = private_module(model_args['d_model'], task_args['his_num'], task_args['pred_num'],
                                         args.num_of_latent,
                                         data_args[args.target_dataset]['num_of_times'],
                                         data_args[args.target_dataset]['num_of_days'],
                                         data_args[args.target_dataset]['node_num'])

    def index_selector(self, x1, x2, moments=2, element_wise=False):
        cmds = CMD(x1, x2, moments, element_wise)
        return cmds

    def forward(self, stage, x, ti, di, ep, tdx, S2D=-1, mbx=None, mby=None, tid=None, s2t_id=None):
        if stage == 'source':
            x = torch.relu(self.emb(x))
            for st_module in self.st_models:
                x = st_module(x, ti, ti, ep, tdx, S2D)
            x = self.re_emb_s(x[:, : tdx])
            return (x * self.mean_std[1] + self.mean_std[0]).squeeze(-1)

        if stage == 'transition':

            # source embedding layer
            x = torch.relu(self.emb(x))

            # embedding layer
            mbx = torch.relu(self.emb(mbx))

            # target distribution transformation
            mbx = self.var_layer1(mbx, tid, ep, tdx, S2D)

            # select index
            sim1, gv1, gi = self.index_selector(mbx, x)  # B

            gi = s2t_id[ti]

            for st_module in self.st_models:
                x = st_module(x, ti, ti, ep, tdx, S2D)
                mbx = st_module(mbx, gi, gi, ep, tdx, S2D)

            mbx = self.var_layer2(mbx, tid, ep, tdx, S2D) 

            x = self.re_emb_s(x)
            mbx = self.re_emb_s(mbx)

            mby = (mby - self.mean_std[0]) / self.mean_std[1]
            gv2 = self.index_selector(mbx, mby, element_wise=True)  # B

            loss = self.lambda1 * torch.mean(gv1) + self.lambda2 * torch.mean(gv2)  # 是否固定映射呢

            return (x * self.mean_std[1] + self.mean_std[0]).squeeze(-1), loss

        if stage == 'finetuning':
            x = torch.relu(self.emb(x))

            x = self.var_layer1(x, ti, ep, tdx, S2D)

            gi = s2t_id[ti]

            for st_module in self.st_models:
                x = st_module(x, gi, gi, ep, tdx, S2D)

            x = self.var_layer2(x, ti, ep, tdx, S2D)  
            x = self.re_emb_s(x)
            return (x * self.mean_std[1] + self.mean_std[0]).squeeze(-1)




