import numpy as np
import torch
import copy
import os


# code based on pytorchtools -> early stopping
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopper:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, delta=0, save_dir=None, verbose=False, large_is_better=False, trace_func=print):
        """
        :param patience:
        :param delta:
        :param save_dir:
        :param verbose:
        :param large_is_better: set large_is_better to false if we log loss, because smaller loss is better
        :param trace_func:
        """
        # 当验证集损失在连续patience次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.trace_func = trace_func
        self.large_is_better = large_is_better

        # buffer the best model and optimizer states
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None
        self.model_save_path = os.path.join(self.save_dir, 'model_dict.pt') if self.save_dir is not None else None
        self.optimizer_save_path = os.path.join(self.save_dir,
                                                'optimizer_dict.pt') if self.save_dir is not None else None

    def get_final_res(self):
        res = {'es_best_model': self.best_model_state_dict,
               'es_best_opt': self.best_optimizer_state_dict,
               'best_score': self.best_score}

        return res

    def register(self, current_score, model, optimizer):
        # assert not self.early_stop, "early_stop=True, you should not do more registration"
        if self.early_stop:
            self.trace_func('Actually you should stop registering scores, because early stop is already triggered')
            return

        if self.large_is_better:
            current_score = -1.0 * current_score

        if self.best_score is None:
            self.best_score = current_score
            self.buffer_checkpoint(model, optimizer)
        elif current_score > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.buffer_checkpoint(model, optimizer)
            self.counter = 0

    def buffer_checkpoint(self, model, optimizer):
        self.best_model_state_dict = copy.deepcopy(model.state_dict())
        self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None

        if self.save_dir is not None:
            self.save_checkpoint(model, optimizer)
            suffix = ' and saved on disk'
        else:
            suffix = ''

        if self.verbose:
            self.trace_func(f'Best Model/Optimizer buffered{suffix}')

    def save_checkpoint(self, model, optimizer):
        """
        Saves model to disc
        """

        torch.save(model.state_dict(), self.model_save_path)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), self.optimizer_save_path)
