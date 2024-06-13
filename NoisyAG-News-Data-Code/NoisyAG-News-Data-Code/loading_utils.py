import os
import copy
import numpy as np
import pickle
import torch
from tqdm import tqdm
from text_dataset import TextBertDataset, TextBertRealDataset
from transformers import BertTokenizer,XLNetTokenizer,RobertaTokenizer,BartTokenizer,AlbertTokenizer,T5Tokenizer,GPT2Tokenizer

from transformers import AutoTokenizer
import utils


def compute_transition_matrix(y, noisy):
    # 确定标签的数量
    num_labels = max(max(noisy), max(y)) + 1

    # 初始化转移矩阵
    T = np.zeros((num_labels, num_labels))

    # 统计每个标签对的出现次数
    for i in range(len(y)):
        T[y[i]][noisy[i]] += 1

    # 归一化每一行
    T = T / T.sum(axis=1, keepdims=True)
    
    return T

def prepare_data(args, logger, r_state, num_classes, has_val, has_ul):
    # used for experiments with injected noise
    # 在这里加载模型的tokenizer，后续如果更换模型，相应的tokenizer可能也需要修改
    tokenizer = load_tokenizer(args)
    tr_data, val_data = get_training_validation_set(args, logger, tokenizer, r_state, has_val, num_classes)
    test_data = load_and_cache_text(args, tokenizer, logger, tag='test')

    # 在构造数据集的时候看要不要加噪声
    n_set = TextBertDataset(args, tr_data, tokenizer, r_state, num_classes, logger,make_noisy=True)
    v_set = TextBertDataset(args, val_data, tokenizer, r_state, num_classes, logger,make_noisy=True)
    t_set = TextBertDataset(args, test_data, tokenizer, r_state, num_classes, logger, make_noisy=False)

    n_set_noisy_labels = copy.deepcopy(n_set.noisy_labels)
    with open("./data/gt/txt_data/train_labels.pickle","rb") as PF:
        gtLabel = pickle.load(PF)
    trainMat = compute_transition_matrix(gtLabel[:len(n_set.noisy_labels)],n_set.noisy_labels)
    v_set_noisy_labels = copy.deepcopy(v_set.noisy_labels)
    valMat = compute_transition_matrix(v_set.clean_labels,v_set.noisy_labels)
    n_set_noisy_labels_hash = hash(tuple(n_set_noisy_labels))
    v_set_noisy_labels_hash = hash(tuple(v_set_noisy_labels))
    # wandb.run.summary["train_n_hash"] = n_set_noisy_labels_hash
    # wandb.run.summary["val_n_hash"] = v_set_noisy_labels_hash

    u_set = None
    l2id = None
    id2l = None
    

    return n_set, u_set, v_set, t_set, l2id, id2l,trainMat,valMat


def prepare_af_data(args, logger, num_classes, has_ul):
    tokenizer = load_tokenizer(args)
    n_set = get_clean_and_noisy_data_by_tag(args, logger, tokenizer, num_classes, tag='train')
    v_set = get_clean_and_noisy_data_by_tag(args, logger, tokenizer, num_classes, tag='validation')
    t_set = get_clean_and_noisy_data_by_tag(args, logger, tokenizer, num_classes, tag='test')

    assert not has_ul  # we do not have unlabeled data in Yoruba and Hausa dataset
    u_set = None

    label_mapping_data_dir = os.path.join(args.data_root, args.dataset, 'txt_data')
    l2id = utils.pickle_load(os.path.join(label_mapping_data_dir, 'l2idx.pickle'))
    id2l = utils.pickle_load(os.path.join(label_mapping_data_dir, 'idx2l.pickle'))

    return n_set, u_set, v_set, t_set, l2id, id2l


def get_training_validation_set(args, logger, tokenizer, r_state, has_val, num_classes):
    # sanity check: args.gen_val is used when there is no validation set
    if has_val:
        assert not args.gen_val

    # 把数据集的文本经过预处理，得到tokens ids和mask等
    tr_data = load_and_cache_text(args, tokenizer, logger, tag='train')

    if has_val:  # original validation set available
        val_data = load_and_cache_text(args, tokenizer, logger, tag='validation')
    elif args.gen_val:  # create validation set using the training set
        # 利用训练集的数据，创建一个交叉验证集，索引文件的路径为val_indices下原始的train加上val_indices
        val_indices_path = os.path.join(args.data_root, args.dataset, 'val_indices', f'{args.dataset}_val_indices.pickle')
        with open(val_indices_path, 'rb') as handle:
            val_indices = pickle.load(handle)

        val_mask = np.zeros(len(tr_data['labels']), dtype=bool)
        val_mask[val_indices] = True

        val_features = {k: v[val_mask] for k,v in tr_data['features'].items()}
        val_labels = tr_data['labels'][val_mask]
        val_text  = np.array(tr_data['text'])[val_mask]
        val_index = tr_data['textIndex'][val_mask]

        train_features = {k: v[~val_mask] for k,v in tr_data['features'].items()}
        train_labels = tr_data['labels'][~val_mask]
        train_text  = np.array(tr_data['text'])[~val_mask]
        train_index = tr_data['textIndex'][~val_mask]

        val_data = {"features": val_features, "labels": val_labels, "text": val_text,"textIndex":val_index}
        tr_data = {"features": train_features, "labels": train_labels, "text": train_text,"textIndex":train_index}

    else:
        raise ValueError("we need a validation set, set gen_val to True to extract"
                         "a subset from the training data as validation data")

    return tr_data, val_data

def get_clean_and_noisy_data_by_tag(args, logger, tokenizer, num_classes, tag):
    noisy_data_tag = f'{tag}_clean'

    #get text data with noisy labels
    clean_noisy_data = load_and_cache_text(args, tokenizer, logger, tag=noisy_data_tag)

    #get the clean training and the clean validation sets
    txt_data_dir = os.path.join(args.data_root, args.dataset, 'txt_data')
    input_path = os.path.join(txt_data_dir, f'{tag}_clean_noisy_labels.pickle')
    noisy_labels = load_pickle_data(input_path)

    td = TextBertRealDataset(args, clean_noisy_data, noisy_labels, tokenizer, num_classes)
    return td



def load_and_cache_text(args, tokenizer, logger, tag):
    # 这里是bert_preprocessed
    cached_features_dir = os.path.join(args.data_root, args.dataset, 'bert_preprocessed') # cache dir (output dir)
    txt_data_dir = os.path.join(args.data_root, args.dataset, 'txt_data') # input file dir

    if not os.path.exists(cached_features_dir):
        os.makedirs(cached_features_dir)

    cached_features_path = os.path.join(cached_features_dir,
                                        f'{args.model_name}_{tag}_trun_{args.truncate_mode}_maxl_{args.max_sen_len}')
    input_path = os.path.join(txt_data_dir, f'{tag}.txt')
    logger.info(f'data path:  {input_path}')
    # 这里的input_path指的是到train.txt的路径
    docs = read_txt(input_path)

    # 如果存在预处理好的特征，直接加载
    # if os.path.exists(cached_features_path):
    #     logger.info(f'[Loading and Caching] loading from cache...')
    #     features = torch.load(cached_features_path)
    # else:
    # 直接每次都做tokenizer，不加载什么cache 
    # logger.info(f'[Loading and Caching] number of documents = {len(docs)}')
    # logger.info(f'[Loading and Caching] convert text to features...')
    features = get_input_features(docs, tokenizer, args)
    # logger.info("[Loading and Caching] saving/caching the features...")
    torch.save(features, cached_features_path)
    # logger.info("[Loading and Caching] saved")

    # logger.info(f'[Loading and Caching] loading labels...')
    input_path = os.path.join(txt_data_dir, f'{tag}_labels.pickle')
    with open(input_path, 'rb') as handle:
        labels = np.array(pickle.load(handle))

    # 返回每个样本的索引
    docsIndex = np.array([int(i) for i in range(len(docs))])

    # feature里面又套了一层
    # return {'input_ids': input_id_tensor, 'token_type_ids': token_type_tensor,
    #       'attention_mask': attention_mask_tensor, 'length': length_tensor}
    # 相当于是先找到features 再找到input_ids
    return {"features": features, "labels": labels, "text": docs,"textIndex":docsIndex}




def load_tokenizer(args):
    # 具体是这里加载模型的tokenizer
    if args.model_name == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained("./models/bert-base-uncased/")
    elif args.model_name == "xlnet-base-cased":
        tokenizer = XLNetTokenizer.from_pretrained("./models/xlnet-base-cased/")
    elif args.model_name == "roberta-base":
        tokenizer = RobertaTokenizer.from_pretrained("./models/roberta-base/")
    elif args.model_name == "bart-base":
        tokenizer = BartTokenizer.from_pretrained("./models/bart-base/")
    elif args.model_name == "albert-base-v2":
        tokenizer = AlbertTokenizer.from_pretrained("./models/albert-base-v2/")
    elif args.model_name == "t5-base":
        tokenizer = T5Tokenizer.from_pretrained("./models/t5-base/")
    elif args.model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("./models/GPT-2/")
    elif args.model_name == "deberta-v3-base":
        tokenizer = AutoTokenizer.from_pretrained("./models/deberta-v3-base/")
        
    return tokenizer


def load_pickle_data(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data



def read_txt(file_path):
    text_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        text_list = [line.rstrip('\n') for line in f]
    return text_list


def truncate_token_ids(token_ids, args, limit):
    if args.truncate_mode == 'last':
        return token_ids[-limit:]
    elif args.truncate_mode == 'hybrid':
        return token_ids[:128] + token_ids[-382:]
    else:
        raise ValueError('truncate model not supported')


def get_input_features(docs, tokenizer, args):
    # 把读入的文本内容，转化成可以输入的tokens和mask
    limit = args.max_sen_len - args.special_token_offsets
    # sanity check
    if args.truncate_mode == 'hybrid':
        assert args.max_sen_len == 512
    assert limit > 0
    num_docs = len(docs)

    input_id_tensor = torch.zeros((num_docs, args.max_sen_len)).long()
    length_tensor = torch.zeros(num_docs).long()
    token_type_tensor = torch.zeros((num_docs, args.max_sen_len)).long()
    attention_mask_tensor = torch.zeros((num_docs, args.max_sen_len)).long()

    for idx, doc in enumerate(tqdm(docs, desc='convert docs to tensors')):
        # 在这里进行tokenizer
        tokens = tokenizer.tokenize(doc)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids_trunc = truncate_token_ids(token_ids, args, limit)
        input_ids = torch.tensor([tokenizer.cls_token_id] + token_ids_trunc +
                                [tokenizer.sep_token_id]).long()
        input_ids_length = len(input_ids)
        # token_types = torch.zeros(len(input_ids)).long()
        attention_mask = torch.ones(len(input_ids)).long()
        input_id_tensor[idx, :input_ids_length] = input_ids
        length_tensor[idx] = input_ids_length
        attention_mask_tensor[idx, :input_ids_length] = attention_mask
        

    return {'input_ids': input_id_tensor, 'token_type_ids': token_type_tensor,
            'attention_mask': attention_mask_tensor, 'length': length_tensor}