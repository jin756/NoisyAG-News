import numpy as np
import copy
import torch
from scipy import stats
from text2vec import SentenceModel
import os
import pickle
from numpy import inf
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel 
from sklearn.metrics import confusion_matrix  

def calculate_accuracy(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    correct_predictions = sum(1 for a, b in zip(array1, array2) if a == b)
    total_predictions = len(array1)

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

def make_noisy_general(clean_data, noise_matrix, random_state, num_classes):
    """ Perturbs the MNIST labels based on the probabilities of the given noise matrix

    Args:
        clean_data: list of instances
        noise_matrix: defines the noise process
        random_state: for reproducibility

    Returns:
        A perturbed copy of clean_data (the noisy_data)
    """
    for row in noise_matrix:
        assert np.isclose(np.sum(row), 1)

    assert len(noise_matrix) == num_classes

    noisy_data = copy.deepcopy(clean_data)
    for i in range(len(noisy_data)):
        probability_row = noise_matrix[noisy_data[i]]
        noisy_data[i] = random_state.choice(num_classes, p=probability_row)
    return noisy_data


def make_data_noisy(input_data, y, noise_level, logger,noise_type, r_state, num_classes):
    assert noise_type in ['sflip', 'uniform', 'uniform_m','Wmatrix_Ins','NoneNoise']

    if noise_type == 'sflip':
        _, noisy_data = make_noisy_single_flip(y, noise_level, r_state, num_classes)
    elif noise_type == 'uniform':
        _, noisy_data = make_noisy_uniform(y, noise_level, r_state, num_classes)
    elif noise_type == 'uniform_m':
        _, noisy_data = make_noisy_uniform_m(y, noise_level, r_state, num_classes)
    elif noise_type == 'Wmatrix_Ins':
        _, noisy_data = make_noisy_Wmatrix_Ins(input_data,y, noise_level, r_state, num_classes,logger)
    elif noise_type == 'NoneNoise':
        noisy_data = y 
    else:
        raise NotImplementedError('noise type not supported')

    logger.info(" make_data_noisy : get the rate ")
    logger.info(type(y))
    logger.info(type(noisy_data))
    logger.info("accuracy is :  ")
    logger.info(calculate_accuracy(y,noisy_data))

    return noisy_data

def make_noisy_Wmatrix_Ins(input_data,y, noise_level, r_state, num_classes,logger):
    # n -> noise_rate   0.2 已经有了，noise_level
    # dataset -> mnist, cifar10 # 有了，input_data
    # labels -> labels (targets)  60000 * 1 的tensor
    # 好像只起一个shape的作用，所以可以由input_data里面的代替
    # label_num -> class number   10  有num_classes
    # feature_size -> the size of input images (e.g. 28*28)  28 * 28 = 784
    # 这个也可以自己设置，或者由embedding向量得到
    # norm_std -> default 0.1  可以自己设置，不用传递进来
    # seed -> random_seed 这个有了r_state
    norm_std = 0.1

    n = noise_level

    print("building dataset...")
    label_num = num_classes
    # 设置随机数种子

    P = []
    # 以噪声率n生成一个截断正态分布的噪声率
    # 上限为 (1 - n) / norm_std 倍数大小，下限为(0 - n) / norm_std 倍数大小 ，均值为n ，方差为norm_std
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std,
                                        loc=n, scale=norm_std)
    # shape: 60000 * 1 相当于每一个数据是每一个样本的噪声率 意思是随机选取 len(y)个数据
    flip_rate = flip_distribution.rvs(len(y))

    # 加载模型以生成文本的embeddings，然后根据embedding生成相关的实例噪声。
    # 如果已经有了，请直接加载，但是注意这里是不是训练集和验证集难以区分，所以需要每次都加载。
    # embeddingsFilePath = "./data/AG_News/embeddings/AG_News_embeddings.pkl"
    # if os.path.exists(embeddingsFilePath):
    #     with open(embeddingsFilePath,"rb") as Fr:
    #         textVecList = pickle.load(Fr)
    # else:
    #     sbert_model = SentenceModel("./text2vec/text2vec-base-multilingual")
    #     textVecList = sbert_model.encode(input_data['text'])
    #     with open(embeddingsFilePath,"wb") as Fw:
    #         pickle.dump(textVecList,Fw)
    # 在这里加入判断是不是AG_News的全数据集，如果是的话，加载缓存的embeddings文件，要不然需要运行很久
    logger.info(" get the content embeddings ")

    # 利用text2vec来生成表示
    sbert_model = SentenceModel("./text2vec/text2vec-base-multilingual")
    textVecList = sbert_model.encode(input_data['text'])
    #     with open(embeddingsFilePath,"wb") as Fw:
    #         pickle.dump(textVecList,Fw)
    # 判断下textVecList的长度和现在的input_data长度是不是相等

    # textVecList = []
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = AutoModel.from_pretrained("bert-base-uncased")
    # for i in range(len(input_data['text'])):
    #     inputs = tokenizer(input_data['text'][i], return_tensors="pt")
    #     outputs = model(**inputs) 
    #     textVecList.append(outputs[1][0]) 
    #     if i % 1000 == 0:
    #         logger.info(f'vec per: {i}') 
    assert len(textVecList) == len(input_data['text'])
    feature_size = len(textVecList[0])  # 这个可以自动获取
    logger.info(" get the W  ")
    W = np.random.randn(label_num, feature_size, label_num)
    W = torch.FloatTensor(W).cuda()
    #  W 为 4 * 384 * 4
    y = y.tolist()

    logger.info(" get the probaily P ")
    for i,(vec,label) in enumerate( zip(textVecList,y )):
        # 取出一个文本的 向量表示 和 标签
        vec = torch.FloatTensor(vec).cuda()
        # x.view(1, -1)这里先把x展开为 1* 384
        # W[label]为 384 * 4 ，  mm表示tensor的相乘，相当于是 1 * 384 X 384 * 4  = 1 * 4
        # squeeze降维，把1*4变成4 ，去掉第一个维度
        A = vec.view(1, -1).mm(W[label]).squeeze(0)
        # 当表示为-inf时，表示当前的A[y]参与Softmax计算时，占据的权重极小，只计算其他的权重
        A[label] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        # 经过相当于是不发生噪声的概率
        A[label] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    # 根据计算的P[i],从l取标签，其中l = [0,1,2,3,4,5,6,7,8,9]
    new_label = [r_state.choice(l, p=P[i]) for i in range(len(y))]
    logger.info(" get the new_label ")

    # 实例噪声模拟方法和他们不一样，他们需要先获得噪声矩阵然后生成噪声标签
    # 我们通过向量投影计算出噪声标签，去生成混淆矩阵。
    cm = confusion_matrix(y, np.array(new_label)) 
    logger.info(f'confusion matrix: {cm}') 

    return cm,np.array(new_label)




def make_noisy_uniform(y, noise_level, r_state, num_classes):
    assert num_classes == len(set(y))
    clean_label_probability = 1 - noise_level
    uniform_noise_probability = noise_level / num_classes  # distribute noise_level across all other labels
    clean_label_probability += uniform_noise_probability

    true_noise_matrix = np.empty((num_classes, num_classes))
    true_noise_matrix.fill(uniform_noise_probability)
    for true_label in range(num_classes):
        true_noise_matrix[true_label][true_label] = clean_label_probability

    noisy_data = make_noisy_general(y, true_noise_matrix, r_state, num_classes)

    return true_noise_matrix, noisy_data

def get_uniform_m_flip_mat(noise_level, num_classes):
    clean_label_probability = 1 - noise_level
    uniform_noise_probability = noise_level / (num_classes - 1)  # distribute noise_level across all other labels

    true_noise_matrix = np.empty((num_classes, num_classes))
    true_noise_matrix.fill(uniform_noise_probability)
    for true_label in range(num_classes):
        true_noise_matrix[true_label][true_label] = clean_label_probability

    return true_noise_matrix


def make_noisy_uniform_m(y, noise_level, r_state, num_classes):
    assert num_classes == len(set(y))

    true_noise_matrix = get_uniform_m_flip_mat(noise_level, num_classes)

    noisy_data = make_noisy_general(y, true_noise_matrix, r_state, num_classes)

    return true_noise_matrix, noisy_data



def get_single_flip_mat(noise_level, num_classes):
    flips = np.arange(num_classes)
    flips = np.roll(flips, 1)

    true_noise_matrix = np.zeros((num_classes, num_classes))
    for true_label in range(num_classes):
        true_noise_matrix[true_label][true_label] = 1 - noise_level
        true_noise_matrix[true_label][flips[true_label]] = noise_level
    return true_noise_matrix



def make_noisy_single_flip(y, noise_level, r_state, num_classes):
    assert num_classes == len(set(y))
    true_noise_matrix = get_single_flip_mat(noise_level, num_classes)

    noisy_data = make_noisy_general(y, true_noise_matrix, r_state, num_classes)

    return true_noise_matrix, noisy_data
