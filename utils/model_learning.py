import torch
import time
from utils.evaluation import masked_mae_np, masked_mape_np, masked_mse_np
from utils.logs import log_string
import os
from tqdm import tqdm


class model_learning:
    def __init__(self, config_args, args):
        self.model_args, self.task_args, self.data_args = config_args['model'], config_args['task'], config_args['data']
        self.args = args
        self.S2D = args.S2D
        self.EpT = args.EpT
        self.max_norm = self.data_args[args.source_dataset]['max_norm']
        self.criterion = torch.nn.HuberLoss(delta=1)
        self.mae_criterion = torch.nn.L1Loss(reduction='mean')

    def increase_step(self, net, ep, tdx, log):
        if ep % self.EpT == 0:
            log_string(log, 'step increase from {} to {}...'.format(tdx, tdx + 1))
            tdx = tdx + 1
            if tdx != 1:
                qs = []
                ks = []
                for layer in range(self.model_args['num_of_layers']):
                    qs.append('st_models.' + str(layer) + '.ptgcn.sta_adj.weight')
                    ks.append('st_models.' + str(layer) + '.ptgcn.sta_adj.weight')
                    for q, k in zip(qs, ks):
                        net.state_dict()[q][tdx - 1: tdx, :] = net.state_dict()[k][tdx - 2: tdx - 1, :]
        return net, tdx

    def switch_to_dynamic(self, net):
        qs = []
        ks = []
        for layer in range(self.model_args['num_of_layers']):
            qs.append('st_models.' + str(layer) + '.ptgcn.t_adjs.weight')
            ks.append('st_models.' + str(layer) + '.ptgcn.sta_adj.weight')
            qs.append('st_models.' + str(layer) + '.psgcn.t_adjs.weights.weight')
            ks.append('st_models.' + str(layer) + '.psgcn.sta_adj.weights.weight')
            for q, k in zip(qs, ks):
                net.state_dict()[q][:] = net.state_dict()[k]
        return net

    def get_in_out(self, data, label):
        data = data.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)
        ti, di, data = data[:, -1, -1, 1].type(torch.LongTensor), \
                       data[:, -1, -1, 2].type(torch.LongTensor), \
                       data[..., :1]
        return data, label, ti, di

    def get_bank_in_out(self, target_memory_bank):
        for _, (mbx, mby) in enumerate(target_memory_bank):
            tid, mbx = mbx[:, -1, -1, 1].type(torch.LongTensor), mbx[..., :1]
            mbx = mbx.cuda(non_blocking=True)
            mby = mby.cuda(non_blocking=True).unsqueeze(-1)
            break
        return mbx, mby, tid

    def start_training_stage(self, net, optimizer, loader, ep, epoch, tdx, samples, stage, log, target_memory_bank=None, s2t_id=None):
        train_mae = 0
        net.train()
        start_time = time.time()

        with torch.enable_grad():
            for idx, (data, label) in enumerate(tqdm(loader)):
                data, label, ti, di = self.get_in_out(data, label)
                if stage == 'source':
                    pre = net(stage, data, ti, di, ep, tdx, self.S2D)
                    train_mae = train_mae + self.mae_criterion(pre, label[:, : tdx]).item() * (pre.shape[0] / samples)
                    loss = self.criterion(pre, label[:, : tdx])

                elif stage == 'transition':
                    mbx, mby, tid = self.get_bank_in_out(target_memory_bank)
                    pre, domain_loss = net(stage, data, ti, di, ep, tdx, mbx=mbx, mby=mby, tid=tid)
                    train_mae = train_mae + domain_loss * (pre.shape[0] / samples)
                    loss = domain_loss + self.criterion(pre, label)

                elif stage == 'finetuning':
                    pre = net(stage, data, ti, di, ep, tdx, s2t_id=s2t_id)
                    train_mae = train_mae + self.mae_criterion(pre, label).item() * (pre.shape[0] / samples)
                    loss = self.criterion(pre, label)

                optimizer.zero_grad()
                loss.backward()
                if self.max_norm:
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm, norm_type=2)
                optimizer.step()

        train_time = time.time() - start_time
        log_string(log, 'training: epoch: {}/{},  mae: {:.3f}, time: {:.3f}s'.format(ep + 1, epoch, train_mae, train_time), use_info=False)
        return net, train_mae, train_time

    def start_validation_stage(self, net, loader, ep, tdx, samples, lowest_val_loss, stage, log, param_file, target_memory_bank=None, s2t_id=None):
        net.eval()
        val_mae = 0
        start_time = time.time()
        with torch.no_grad():
            for idx, (data, label) in enumerate(tqdm(loader)):
                data, label, ti, di = self.get_in_out(data, label)
                if stage == 'source':
                    pre = net(stage, data, ti, di, ep, tdx, self.S2D)
                    val_mae = val_mae + self.mae_criterion(pre, label[:, : tdx]).item() * (pre.shape[0] / samples)

                elif stage == 'transition':
                    mbx, mby, tid = self.get_bank_in_out(target_memory_bank)
                    pre, domain_loss = net(stage, data, ti, di, ep, tdx, mbx=mbx, mby=mby, tid=tid)
                    val_mae = val_mae + domain_loss * (pre.shape[0] / samples)
                else:
                    pre = net(stage, data, ti, di, ep, tdx, s2t_id=s2t_id)
                    val_mae = val_mae + self.mae_criterion(pre, label).item() * (pre.shape[0] / samples)
        torch.save(net.state_dict(), param_file + '--epoch=' + str(ep + 1))

        if stage == 'source':
            log_string(log, 'validation: mae: {:.3f}, time: {:.3f}s'.format(val_mae, time.time() - start_time), use_info=False)
            if val_mae < lowest_val_loss:
                log_string(log, 'update best_model...', use_info=False)
                torch.save(net.state_dict(), param_file + '--best_model')
                lowest_val_loss = val_mae
        else:
            log_string(log, 'testing: mae: {:.3f}, time: {:.3f}s'.format(val_mae, time.time() - start_time), use_info=False)
            log_string(log, 'update current_model...', use_info=False)
        log_string(log, '\n', use_info=False)
        return lowest_val_loss, val_mae

    def start_test_stage(self, net, loader, stage, s2t_id=None):
        net.eval()
        with torch.no_grad():
            pres = []
            labels = []
            for idx, (data, label) in enumerate(tqdm(loader)):
                data, label, ti, di = self.get_in_out(data, label)
                pre = net(stage, data, ti, di, ep=self.task_args['source_epoch'], tdx=self.task_args['pred_num'], s2t_id=s2t_id)
                pres.append(pre.to('cpu'))
                labels.append(label.to('cpu'))
            pres = torch.cat(pres, dim=0).detach().numpy()
            labels = torch.cat(labels, dim=0).detach().numpy()
            prediction_info = []
            for idx in range(self.task_args['pred_num']):
                y, x = labels[:, idx: idx + 1, :], pres[:, idx: idx + 1, :]
                prediction_info.append((masked_mae_np(y, x, 0), masked_mape_np(y, x, 0), masked_mse_np(y, x, 0) ** 0.5))

            prediction_info.append((
                masked_mae_np(labels, pres, 0),
                masked_mape_np(labels, pres, 0),
                masked_mse_np(labels, pres, 0) ** 0.5))
        return prediction_info


    def load_current_model(self, stage, net, epoch, tdx, param_file, log, loader, samples, optimizer, lowest_val_loss, target_memory_bank=None, s2t_id=None):
        for ep_id in range(epoch):
            if stage == 'source':
                if ep_id == 150:
                    optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['source_lr'] * 0.1)
                if tdx < self.task_args['pred_num']:
                    net, tdx = self.increase_step(net, ep_id, tdx, log)
                if ep_id == self.S2D:
                    net = self.switch_to_dynamic(net)
            if os.path.exists(param_file + '--epoch={}'.format(ep_id+1)):
                continue
            else:
                if stage == 'source':
                    net.load_state_dict(torch.load(param_file + '--best_model'))
                    lowest_val_loss, val_mae = self.start_validation_stage(net, loader, ep_id, tdx, samples,
                                                                       lowest_val_loss, stage, log, param_file)
                    net.load_state_dict(torch.load(param_file + '--epoch={}'.format(ep_id)))
                else:
                    net.load_state_dict(torch.load(param_file + '--epoch={}'.format(ep_id)))
                    lowest_val_loss, val_mae = self.start_validation_stage(net, loader, ep_id, tdx, samples,
                                                                       lowest_val_loss, stage, log, param_file, target_memory_bank, s2t_id)
                log_string(log, 'continue to training...\n')
                break
        return net, ep_id, lowest_val_loss, optimizer, tdx


    def source_learning(self, net, param_file, loader, samples, log):
        stage = 'source'
        epoch = self.task_args['source_epoch']
        lowest_val_loss = 1e6
        ep_id = 0
        train_maes, val_maes, train_times = [], [], []
        optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['source_lr'])  # weight_decay=wd
        if self.args.step_increase:
            tdx = 0
        else:
            tdx = self.task_args['pred_num']
        if os.path.exists(param_file + '--epoch={}'.format(ep_id+1)):
            net, ep_id, lowest_val_loss, optimizer, tdx = self.load_current_model(
            stage, net, epoch, tdx, param_file, log, loader[1], samples[1], optimizer, lowest_val_loss)

        for ep in range(ep_id, epoch):
            if ep == 150:
                optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['source_lr'] * 0.1)

            if tdx < self.task_args['pred_num']:
                lowest_val_loss = 1e6
                net, tdx = self.increase_step(net, ep, tdx, log)
            if ep == self.S2D:
                lowest_val_loss = 1e6
                net = self.switch_to_dynamic(net)

            net, train_mae, train_time = self.start_training_stage(net, optimizer, loader[0], ep, epoch, tdx, samples[0], stage, log)
            lowest_val_loss, val_mae = self.start_validation_stage(net, loader[1], ep, tdx, samples[1], lowest_val_loss, stage, log, param_file)

            train_maes.append(train_mae)
            val_maes.append(val_mae)
            train_times.append(train_time)

        return net

    def transition_learning(self, net, param_file, loader, target_memory_bank, samples, log):
        stage = 'transition'
        epoch = self.task_args['transition_epoch']
        lowest_val_loss = 1e6
        ep_id = 0
        train_maes, val_maes, train_times = [], [], []
        optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['transition_lr'])  # weight_decay=wd
        tdx = self.task_args['pred_num']
        if os.path.exists(param_file + '--epoch={}'.format(ep_id+1)):
            net, ep_id, lowest_val_loss, optimizer, tdx = self.load_current_model(
            stage, net, epoch, tdx, param_file, log, loader[0], samples[0], optimizer, lowest_val_loss, target_memory_bank)

        for ep in range(ep_id, epoch):
            net, train_mae, train_time = self.start_training_stage(net, optimizer, loader[0], ep, epoch, tdx, samples[0], stage, log, target_memory_bank)
            lowest_val_loss, val_mae = self.start_validation_stage(net, loader[2], ep, tdx, samples[2], lowest_val_loss, stage, log, param_file, target_memory_bank)

            train_maes.append(train_mae)
            val_maes.append(val_mae)
            train_times.append(train_time)
        return net

    def finetuning_learning(self, net, param_file, loader, samples, log, s2t_id):
        stage = 'finetuning'
        epoch = self.task_args['finetuning_epoch']
        lowest_val_loss = 1e6
        ep_id = 0
        train_maes, val_maes, train_times = [], [], []
        optimizer = torch.optim.Adam(net.parameters(), lr=self.model_args['finetuning_lr'])  # weight_decay=wd
        tdx = self.task_args['pred_num']

        if os.path.exists(param_file + '--epoch={}'.format(ep_id+1)):
            net, ep_id, lowest_val_loss, optimizer, tdx = self.load_current_model(
            stage, net, epoch, tdx, param_file, log, loader[0], samples[0], optimizer, lowest_val_loss, s2t_id=s2t_id)

        for ep in range(ep_id, epoch):
            net, train_mae, train_time = self.start_training_stage(net, optimizer, loader[0], ep, epoch, tdx, samples[0], stage, log, s2t_id=s2t_id)
            lowest_val_loss, val_mae = self.start_validation_stage(net, loader[2], ep, tdx, samples[2], lowest_val_loss, stage, log, param_file, s2t_id=s2t_id)
            train_maes.append(train_mae)
            val_maes.append(val_mae)
            train_times.append(train_time)

        return net



