import torch
import argparse
import yaml
import os
from utils.model_learning import model_learning
from model.make_CGSTT import start_make_model
from utils.data_process import data_loader
from utils.logs import log_string
from model.basic_module import CMD
from utils.random_seed import setup_seed
from thop import profile


parser = argparse.ArgumentParser(description='description')
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--source_dataset', type=str)
parser.add_argument('--target_dataset', type=str)
parser.add_argument('--target_days', default=3, type=int)
parser.add_argument('--num_of_layer', default=4, type=int)
parser.add_argument('--d_model', default=64, type=int)
parser.add_argument('--num_of_latent', default=32, type=int)
parser.add_argument('--S2D', default=80, type=int)
parser.add_argument('--EpT', default=5, type=int)
parser.add_argument("--gpu", type=str, default='1', help="gpu ID")
parser.add_argument('--lambda1', type=float)
parser.add_argument('--lambda2', type=float)
parser.add_argument('--seed', default=0, type=str)
parser.add_argument('--remark', type=str)
parser.add_argument('--step_increase', default=True, type=bool)
parser.add_argument('--transfer_learning', default=True, type=bool)
parser.add_argument("--stars", type=str, default='*************************/{}/*************************')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


class starting:
    def __init__(self, args, config_args):
        self.mean = None
        self.std = None
        self.args = args
        self.config_args = config_args
        self.data_args, self.task_args, self.model_args = config_args['data'], config_args['task'], config_args['model']
        self.data_loader = data_loader(config_args)
        self.model_learner = model_learning(self.config_args, args)

    def file_config(self, stage):

        net_config = '/num_of_layer={}--d_model={}--seed={}--num_of_latent={}'\
            .format(args.num_of_layer, args.d_model, args.seed, args.num_of_latent)
        if stage != 'source':
            net_config = net_config + '--target_day={}'.format(args.target_days)

        # generating log path
        if stage != 'source':
            log_dir = self.data_args[args.source_dataset]['log_path'] + args.remark + '/' + stage
        else:
            log_dir = self.data_args[args.source_dataset]['log_path'] + stage
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        id = 0
        log_file = log_dir + net_config + '_' + str(id)
        while os.path.exists(log_file + '.txt'):
            id = id + 1
            log_file = log_dir + net_config + '_' + str(id)
        log = open(log_file + '.txt', 'w')

        # generating param path
        if stage != 'source':
            param_dir = self.data_args[args.source_dataset]['param_path'] + args.remark + '/' + stage
        else:
            param_dir = self.data_args[args.source_dataset]['param_path'] + stage
        param_file = param_dir + net_config
        if not os.path.exists(param_dir):
            os.makedirs(param_dir)
        log_string(log, 'log_file: {}'.format(log_file))
        log_string(log, 'param_file: {}\n'.format(param_file))

        return log, param_file

    def print_model(self, net):
        param_num = 0
        for name, params in net.named_parameters():
            param_num = param_num + params.nelement()
            log_string(log, str(name) + '--' + str(params.shape), use_info=False)
        log_string(log, 'total num of parameters: {}'.format(param_num))

    def print_FLOPs(self, net, stage, loader):
        net.eval()
        with torch.no_grad():
            for _, (data, label) in enumerate(loader):
                data, label, ti, di = self.model_learner.get_in_out(data, label)
                break
            FLOPs, params = profile(net, inputs=(stage, data, ti, di, self.task_args['source_epoch'], self.task_args['pred_num']))  # macs
        log_string(log, 'total num of FLOPs: {:.4f}G'.format(FLOPs / 1000000000))


    def print_predction(self, info):
        horizon = 1
        log_string(log, args.stars.format('prediction result'), use_info=False)
        for i in info:
            if horizon <= self.config_args['task']['pred_num']:
                log_string(log, '@t' + str(horizon) + ' {:.3f} {:.3f} {:.3f}'.format(*i), use_info=False)
            else:
                log_string(log, 'avg {:.3f} {:.3f} {:.3f}\n'.format(*i), use_info=False)
            horizon = horizon + 1

    def start_load_data(self, stage, log):
        # load source data
        if stage == 'source':
            log_string(log, args.stars.format('loading source data: {}...'.format(args.source_dataset)), use_info=False)
            source_loaders, source_samples, source_centers, mean_std = self.data_loader.starting(log, args.source_dataset, stage)
            self.source_loaders = source_loaders
            self.source_samples = source_samples
            self.source_centers = source_centers
            self.mean_std = mean_std
        else:
            log_string(log, args.stars.format('loading target data: {}...'.format(args.target_dataset)), use_info=False)
            target_loaders, target_samples, target_centers = self.data_loader.starting(log, args.target_dataset, stage='target', target_day=3)
            _, _, s2t_id = CMD(target_centers[0], self.source_centers[0])
            self.target_loaders = target_loaders
            self.target_samples = target_samples
            self.target_memory_bank = target_loaders[0]
            self.s2t_id = s2t_id

    def start_construct_model(self):
        # initial model
        net = start_make_model(self.config_args, args, self.mean_std).cuda()
        return net

    def get_source_model(self, net, param_file, log):
        #starting_main.print_FLOPs(net, stage, self.source_loaders[0])
        if os.path.exists(param_file + '--epoch={}'.format(self.task_args['source_epoch'])):
            log_string(log, 'loading best source model...')
        else:
            log_string(log, 'starting training source model...')
            self.model_learner.source_learning(net, param_file, self.source_loaders, self.source_samples, log)
        net.load_state_dict(torch.load(param_file + '--best_model'))
        return net

    def get_transition_model(self, net, param_file, log):
        if os.path.exists(param_file + '--epoch={}'.format(self.task_args['transition_epoch'])):
            log_string(log, 'loading transition model...')
            net.load_state_dict(torch.load(param_file + '--epoch={}'.format(self.task_args['transition_epoch'])))
        else:
            log_string(log, 'starting training transition model...')
            net = self.model_learner.transition_learning(net, param_file, self.source_loaders, self.target_memory_bank, self.source_samples, log)
        return net

    def get_finetuning_model(self, net, param_file, log):
        if os.path.exists(param_file + '--epoch={}'.format(self.task_args['finetuning_epoch'])):
            log_string(log, 'loading finetuning model...')
            net.load_state_dict(torch.load(param_file + '--epoch={}'.format(self.task_args['finetuning_epoch'])))
        else:
            log_string(log, 'starting finetuning model...')
            net = self.model_learner.finetuning_learning(net, param_file, self.target_loaders, self.target_samples, log, self.s2t_id)
        return net

    def testing_source_model(self, net):
        log_string(log, 'starting source testing...')
        prediction_info = self.model_learner.start_test_stage(net, self.source_loaders[2], stage)
        self.print_predction(prediction_info)
        log.close()

    def testing_finetuning_model(self, net):
        log_string(log, 'starting finetuning test...')
        prediction_info = self.model_learner.start_test_stage(net, self.target_loaders[2], stage, s2t_id=self.s2t_id)
        self.print_predction(prediction_info)
        log.close()


if __name__ == '__main__':

    setup_seed(args.seed)
    with open(args.config_filename) as f:
        config_args = yaml.load(f, Loader=yaml.FullLoader)

    starting_main = starting(args, config_args)
    stage = 'source'
    log, param_file = starting_main.file_config(stage)
    starting_main.start_load_data(stage, log)
    net = starting_main.start_construct_model()
    starting_main.print_model(net)
    net = starting_main.get_source_model(net, param_file, log)
    starting_main.testing_source_model(net)

    ''' **********transfer learning********** '''
    if args.transfer_learning:
        stage = 'transition'
        log, param_file = starting_main.file_config(stage)
        starting_main.start_load_data(stage, log)
        net = starting_main.get_transition_model(net, param_file, log)

        stage = 'finetuning'
        log, param_file = starting_main.file_config(stage)
        net = starting_main.get_finetuning_model(net, param_file, log)
        starting_main.testing_finetuning_model(net)