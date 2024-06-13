import torch
import torch.nn as nn
from noise_functions import get_uniform_m_flip_mat, get_single_flip_mat


# 在cm里要载入一个获得实例噪声模拟噪声矩阵的函数

class CM(nn.Module):

    def __init__(self, model_config, args, base_model):
        super(CM, self).__init__()

        self.base_model = base_model
        num_classes = model_config['num_classes']
        # self.noise_matrix = nn.Parameter(torch.log(torch.eye(num_classes) + 1e-6), requires_grad=True)
        # 生成一个对角线为1，其他地方为0的4*4的二维数组
        self.noise_model = nn.Parameter(torch.eye(num_classes), requires_grad=True)
        self.mat_normalizer = nn.Softmax(dim=1)
        self.logits2dist = nn.Softmax(dim=1)

    def forward(self, x, x_length):
        # clean_logits为经过768*4得到的那个4维的向量
        baseRes = self.base_model(x, x_length)
        clean_logits = baseRes['logits']
        pooler_outputVec = baseRes['pooler_repr']
        

        # 做一下softmax
        clean_dist = self.logits2dist(clean_logits)
        trans_mat = self.mat_normalizer(self.noise_model)

        # 把它乘上估计的混淆矩阵
        logits = torch.matmul(clean_dist, trans_mat)

        return {'logits': logits,'pooler_repr':pooler_outputVec,'mat':trans_mat}


class CMGT(nn.Module):
    def __init__(self, model_config, args, base_model, noise_mat,logger):
        super(CMGT, self).__init__()

        self.base_model = base_model
        self.num_classes = model_config['num_classes']
        self.noise_type = args.noise_type
        self.noise_level = args.noise_level
        self.logger = logger
        self.noise_matrix = nn.Parameter(torch.tensor(noise_mat).float(), requires_grad=False)
        self.logits2dist = nn.Softmax(dim=1)

    def forward(self, x, x_length):
        baseRes = self.base_model(x, x_length)
        clean_logits = baseRes['logits']
        pooler_outputVec = baseRes['pooler_repr']

        clean_dist = self.logits2dist(clean_logits)
        noisy_prob = torch.matmul(clean_dist, self.noise_matrix)
        log_noisy_logits = torch.log(noisy_prob + 1e-6)

        return {'log_noisy_logits': log_noisy_logits,'pooler_repr':pooler_outputVec,'mat':self.noise_matrix}
    
