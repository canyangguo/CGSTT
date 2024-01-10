import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from utils.logs import log_string

def softmax(x):
    x -= np.expand_dims(np.max(x, axis=-1), axis=-1)
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)  #


class data_loader:
    def __init__(self, config_args, mean_std=None):
        self.data_args, self.task_args = config_args['data'], config_args['task']
        self.mean_std = mean_std

    def load_raw_data(self, dataset_name):
        if 'npz' in self.data_args[dataset_name]['dataset_path']:  # pems datasets
            data = np.load(self.data_args[dataset_name]['dataset_path'])['data'][:, :, :1]
        else:  # didi datasets
            data = np.load(self.data_args[dataset_name]['dataset_path'])[:, :, :1]
        return data

    def add_index(self, data, dataset_name):
        index = np.arange(data.shape[0]).reshape(data.shape[0], 1, 1).repeat(data.shape[1], axis=1)
        time_index = index % self.data_args[dataset_name]['num_of_times']
        day_index = (index // self.data_args[dataset_name]['num_of_times']) % self.data_args[dataset_name]['num_of_days']
        data = np.concatenate([data, time_index, day_index], axis=-1)
        return data

    def get_centers(self, x, log, stage, dataset_name):
        centers = []
        tc = np.concatenate([
            np.mean((x[..., : 1])[np.where(x[:, 0, 0, 1] == i)], axis=0, keepdims=True)
            for i in range(self.data_args[dataset_name]['num_of_times'])], axis=0)
        tc = torch.as_tensor(tc, dtype=torch.float32).cuda()
        log_string(log, 'times_centers shape:{}'.format(tc.shape))
        centers.append(tc)

        if stage == 'source':
            dc = np.concatenate([
                np.mean((x[..., : 1])[np.where(x[:, 0, 0, 2] == i)], axis=0, keepdims=True)
                for i in range(self.data_args[dataset_name]['num_of_days'])], axis=0)
            dc = torch.as_tensor(dc, dtype=torch.float32).cuda()
            log_string(log, 'days_centers shape:{}'.format(dc.shape))
            centers.append(dc)
        log_string(log, '', use_info=False)
        return centers

    def split_train_val_test(self, data, target_day, dataset_name):
        length = data.shape[0]
        if target_day is None:
            train_start, train_end = 0, int(length * self.task_args['train_rate'])
        else:
            train_start, train_end = 0, target_day * self.data_args[dataset_name]['num_of_times']

        val_start, val_end = int(length * self.task_args['train_rate']), int(length * (self.task_args['train_rate']+self.task_args['val_rate']))
        test_start, test_end = val_end, length

        for line1, line2 in ((train_start, train_end),
                             (val_start, val_end),
                             (test_start, test_end)):
            x, y = self.generate_seq(data[line1: line2])

            if self.mean_std is None:
                self.mean_std = self.calculate_mean_std(x)
            x = np.concatenate([(x[..., :-2] - self.mean_std[0]) / self.mean_std[1], x[..., -2:]], axis=-1)
            yield x, y

    def generate_seq(self, data):
        x = np.concatenate([np.expand_dims(
            data[i: i + self.task_args['his_num']], 0)
            for i in range(data.shape[0] - self.task_args['his_num'] - self.task_args['pred_num'] + 1)],axis=0)

        y = np.concatenate([np.expand_dims(
            data[i + self.task_args['his_num']: i + self.task_args['his_num'] + self.task_args['pred_num']], 0)
            for i in range(data.shape[0] - self.task_args['his_num'] - self.task_args['pred_num'] + 1)], axis=0)[..., 0]
        return x, y

    def calculate_mean_std(self, train_x):
        return [train_x[..., 0].mean(), train_x[..., 0].std()]


    def get_data_loader(self, x, y, idx, log):

        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        sample = x.shape[0]
        log_string(log, 'input shape:{}'.format(x.shape))
        log_string(log, 'output shape:{}\n'.format(y.shape))
        data = Data.TensorDataset(x, y)

        batch_size = self.task_args['batch_size']
        if idx != 0:
            batch_size = batch_size * 16  # speeding up val and test
        loader = Data.DataLoader(dataset=data, batch_size=batch_size,
                                # pin_memory=True, # num_workers=0,
                                drop_last=False, shuffle=(idx == 0))
        return loader, sample


    def starting(self, log, dataset_name, stage, target_day=None):
        # data L, N, C=1
        data = self.load_raw_data(dataset_name)

        # data L, N, C=3 (feature, time_index, day_index)
        data = self.add_index(data, dataset_name)

        loaders, samples = [], []
        for idx, (x, y) in enumerate(self.split_train_val_test(data, target_day, dataset_name)):
            if idx == 0:
                centers = self.get_centers(x, log, stage, dataset_name)  # 选择哪些时段合适呢？
                if stage == 'target':
                    target_bank, _ = self.get_data_loader(x, y, idx, log)
            loader, sample = self.get_data_loader(x, y, idx, log)
            loaders.append(loader)
            samples.append(sample)

        if stage == 'source':
            return loaders, samples, centers, self.mean_std
        else:
            return loaders, samples, centers






