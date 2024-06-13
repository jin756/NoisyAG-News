import yaml
import os
import torch.nn.functional as F
import logging
import json
import pickle
import datetime
from trainers.wn import WN_Trainer
from trainers.ct import CT_Trainer
from trainers.cm import CM_Trainer
from trainers.cmgt import CMGT_Trainer
from trainers.smoothing import Smoothing_Trainer
from trainers.NegSmoothing import NegSmoothing_Trainer
from trainers.btls import BTLS_Trainer


# 增加噪声处理方法


def create_trainer(args, logger, log_dir, model_config, full_dataset, random_state):
    # 在这里选择噪声处理方法
    if args.trainer_name == 'wn':
        trainer = WN_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'ct':
        trainer = CT_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'cm':
        trainer = CM_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'cmgt':
        trainer = CMGT_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'ls':
        trainer = Smoothing_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'btls':
        trainer = BTLS_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    elif args.trainer_name == 'nls':
        trainer = NegSmoothing_Trainer(args, logger, log_dir, model_config, full_dataset, random_state)
    else:
        raise NotImplementedError('Unknown Trainer Name')

    return trainer


def load_config(args):
    model_config = {}
    model_config['drop_rate'] = args.bert_dropout_rate
    return model_config

def save_config(save_dir, config_name, config_data):
    save_path = os.path.join(save_dir, f'{config_name}.yaml')

    with open(save_path, 'w') as file:
        yaml.dump(config_data, file)

def pickle_save(file_name, data):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_name):
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
    return b


def create_log_path(log_root, args, starting_time):
    staring_time_str = starting_time.strftime("%m_%d_%H_%M_%S")
    suffix = staring_time_str

    # 黄鸿飞修改
    suffix += f'_{args.trainer_name}_{args.model_name}_{args.dataset}_{args.noise_type}_nle{args.noise_level}_seed{args.manualSeed}'
    #suffix += f'_nlb{args.nl_batch_size}'

    log_dir = os.path.join(log_root, suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, 'log.txt')
    return log_path, log_dir


def create_logger(log_root, args):
    starting_time = datetime.datetime.now()

    log_path, log_dir = create_log_path(log_root, args, starting_time)

    # check if the file exist

    console_logging_format = "%(levelname)s %(message)s"
    file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure logger
    logging.basicConfig(level=logging.INFO, format=console_logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    handler = logging.FileHandler(log_path)

    # set the logging level for log file
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(file_logging_format)
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    return logger, log_dir


def save_args(log_dir, args):
    arg_save_path = os.path.join(log_dir, 'config.json')
    with open(arg_save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def create_folders(saveData, dataset, modelName, method):
    # 创建数据集文件夹
    dataset_folder = os.path.join(saveData, dataset)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # 创建模型文件夹
    model_folder = os.path.join(dataset_folder, modelName)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # 创建方法文件夹
    method_folder = os.path.join(model_folder, method)
    if not os.path.exists(method_folder):
        os.makedirs(method_folder)

    return method_folder