import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from trainers.trainer import Trainer
from tqdm import tqdm
import time
import torch.nn.functional as F
import time 
import pickle 
# import wandb

from trainers.early_stopper import EarlyStopper
from trainers.loss_noise_tracker import LossNoiseTracker


def create_folders(saveData, dataset, modelName, method,noise_type,noise_level):
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

    # 创建方法文件夹
    noise_folder = os.path.join(method_folder, str(noise_type) + "_" + str(noise_level))
    if not os.path.exists(noise_folder):
        os.makedirs(noise_folder)

    train_dir = os.path.join(noise_folder, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print(f"Created directory: {train_dir}")
    else:
        print(f"Directory {train_dir} already exists.")

    train_dir = os.path.join(noise_folder, "trainEval")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        print(f"Created directory: {train_dir}")
    else:
        print(f"Directory {train_dir} already exists.")

    # 创建val文件夹
    val_dir = os.path.join(noise_folder, "val")
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        print(f"Created directory: {val_dir}")
    else:
        print(f"Directory {val_dir} already exists.")

    # 创建test文件夹
    test_dir = os.path.join(noise_folder, "test")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created directory: {test_dir}")
    else:
        print(f"Directory {test_dir} already exists.")

    return noise_folder


def log_t(u, t):
    if t==1.0:
        return torch.log(u)
    else:
        return (u**(1.0-t)-1.0)/(1.0-t)

def exp_t(u, t):
    if t==1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0+(1.0-t)*u)**(1.0/(1.0-t))

def compute_normalization_fixed_point(activations,t,num_iters=5):
    mu=torch.max(activations,dim=-1).values.view(-1, 1)
    normalized_activations_step_0=activations-mu
    normalized_activations = normalized_activations_step_0
    i=0
    while i<num_iters:
        i+=1
        logt_partition=torch.sum(exp_t(normalized_activations,t),dim=-1).view(-1,1)
        normalized_activations=normalized_activations_step_0*(logt_partition**(1.0-t))
    logt_partition=torch.sum(exp_t(normalized_activations,t),dim=-1).view(-1,1)
    return -log_t(1.0/logt_partition,t)+mu

def compute_normalization(activations,t,num_iters=5):
    if t<1.0:
        return None
    else:
        return compute_normalization_fixed_point(activations,t,num_iters)

# 是不用one-hot还是这里直接sum了，变成32*1
def tempered_softmax(activations,t,num_iters=5):
    if t==1.0:
        normalization_constants=torch.log(torch.sum(torch.exp(activations),dim=-1))
    else:
        normalization_constants=compute_normalization(activations,t,num_iters)
    return exp_t(activations-normalization_constants,t)

def bi_tempered_logistic_loss(activations,labels,t1,t2,label_smoothing=0.0,num_iters=5):
    # Args:
    #     activations: A multi-dimensional tensor with last dimension `num_classes`. 多维向量，最后一维为类别数目
    #     labels: A tensor with shape and dtype as activations. 真实标签
    #     t1: Temperature 1 (< 1.0 for boundedness).
    #     t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    #     label_smoothing: Label smoothing parameter between [0, 1).
    #     num_iters: Number of iterations to run the method. 运行多少次这个方法
    # Returns:
    #     A loss tensor.
    if label_smoothing>0.0:
        num_classes=labels.shape[-1]
        labels=(1-num_classes/(num_classes-1)*label_smoothing)*labels+label_smoothing/(num_classes-1)
    probabilities=tempered_softmax(activations,t2,num_iters)
    temp1=(log_t(labels+1e-10,t1)-log_t(probabilities,t1))*labels
    temp2=(1/(2-t1))*(torch.pow(labels,2-t1)-torch.pow(probabilities,2-t1))
    loss_values=temp1-temp2
    return torch.sum(loss_values)


class BTLS_Trainer(Trainer):
    def __init__(self, args, logger, log_dir, model_config, full_dataset, random_state):
        super(BTLS_Trainer, self).__init__(args, logger, log_dir, model_config, full_dataset, random_state)
        self.store_model_flag = True if args.store_model == 1 else False


    def train(self, args, logger, full_dataset):
        # wandb.init(project="bertlnl",name="Bert-wn-nl-" + str(args.noise_level) + "-" + str(args.noise_type),mode = "offline")
        
        logger.info('Bert BTLS Trainer: training started')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_time = time.time()
        logger.info(f"train start time: {time.ctime(start_time)}")
        savePath = create_folders("./saveData/",args.dataset,args.model_name,args.trainer_name,args.noise_type,args.noise_level)

        logger.info(f"savePath: {savePath}")
        saveDataDic = {}

        nl_set, ul_set, v_set, t_set, l2id, id2l,trainMat,valMat = full_dataset
        logger.info(f'training size: {len(nl_set)}',)
        logger.info(f'validation size: {len(v_set)}' )
        logger.info(f'test size: {len(t_set)}')

        model = self.create_model(args)
        model = model.to(device)

        assert args.nl_batch_size % args.gradient_accumulation_steps == 0
        nl_sub_batch_size = args.nl_batch_size // args.gradient_accumulation_steps
        nl_bucket = torch.utils.data.DataLoader(nl_set, batch_size=nl_sub_batch_size,
                                                shuffle=False,
                                                num_workers=0,drop_last = True)

        nl_iter = iter(nl_bucket)


        t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.eval_batch_size,
                                               shuffle=True,
                                               num_workers=0,drop_last = True)

        if v_set is None:
            logger.info('No validation set is used here')
            v_loader = None
        else:
            logger.info('Validation set is used here')
            v_loader = torch.utils.data.DataLoader(v_set, batch_size=args.eval_batch_size,
                                                   shuffle=False,
                                                   num_workers=0,drop_last = True)

        num_training_steps = args.num_training_steps

        optimizer_grouped_parameters = self.get_optimizer_grouped_parameters(args, model)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                              num_training_steps=num_training_steps)
        ce_loss_fn = nn.CrossEntropyLoss()

        if self.store_model_flag:
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        # We log the validation accuracy, so, large_is_better should set to True
        early_stopper = EarlyStopper(patience=args.patience, delta=0, save_dir=early_stopper_save_dir,
                                     large_is_better=True, verbose=False, trace_func=logger.info)

        noise_tracker_dir = os.path.join(self.log_dir, 'loss_noise_tracker')
        loss_noise_tracker = LossNoiseTracker(args, logger, nl_set, noise_tracker_dir)

        global_step = 0


        for idx in tqdm(range(num_training_steps),desc=f'{time.strftime("%H:%M:%S")} - [Vanilla Trainer] training - '):
            ce_loss_mean = 0.0
            # 得到当前的epoch
            nowEpoch = ((idx + 1) * args.nl_batch_size) // len(nl_set)

            for i in range(args.gradient_accumulation_steps):
                model.train()
                try:
                    nl_batch = next(nl_iter)
                except:
                    nl_iter = iter(nl_bucket)
                    nl_batch = next(nl_iter)

                # 在这里生成了loss
                nll_loss,sampleSet  = \
                    self.forward_backward_noisy_batch(model, {'nl_batch': nl_batch}, ce_loss_fn, args, device)
                ce_loss_mean += nll_loss
                saveKey = str(nowEpoch) + "-" + str(idx)
                saveDataDic[saveKey] = sampleSet
                # 多少个batch保存数据，并且希望同一个
                if idx %320  == 0 and idx != 0:
                
                    with open(savePath+"/train/train_" +args.dataset+"_" + args.model_name+ "_" + args.trainer_name +"_" + str(idx) + ".pkl","wb") as f:
                        pickle.dump(saveDataDic,f)
                    #np.save(savePath+"/" +args.dataset+"_" + args.model_name+ "_" + args.trainer_name +str(idx) + ".npy",saveDataDic)
                    # 把保存的数据置空
                    saveDataDic = {}

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            #wandb.log({'train/batch_loss': ce_loss_mean})
            logger.info(str(idx) + "-ce_loss_mean-: " + str(ce_loss_mean))

            if self.needs_eval(args, global_step):
                val_score = self.eval_model_with_both_labels(args,model, v_loader, device, logger,idx,savePath,fast_mode=args.fast_eval)
                
                test_score = self.eval_model(args, logger, t_loader, model, device, idx,savePath,fast_mode=args.fast_eval)
               
                early_stopper.register(val_score['score_dict_n']['accuracy'], model, optimizer)

                # wandb.log({'eval/loss/val_c_loss': val_score['val_c_loss'],
                #            'eval/loss/val_n_loss': val_score['val_n_loss'],
                #            'eval/score/val_c_acc': val_score['score_dict_c']['accuracy'],
                #            'eval/score/val_n_acc': val_score['score_dict_n']['accuracy'],
                #            'eval/score/test_acc': test_score['score_dict']['accuracy']}, step=global_step)
                logger.info(str(global_step// args.eval_freq) + "-eval/loss/val_c_loss-: " + str(val_score['val_c_loss']))
                logger.info(str(global_step// args.eval_freq) +"-eval/loss/val_n_loss-: " + str(val_score['val_n_loss']))
                logger.info(str(global_step// args.eval_freq) +"-eval/score/val_c_acc-: &&" + str(val_score['score_dict_c']['accuracy']) +"&&")
                logger.info(str(global_step// args.eval_freq) +"-eval/score/val_n_acc-: &&" + str(val_score['score_dict_n']['accuracy']) +"&&")
                logger.info(str(global_step// args.eval_freq) +"-eval/score/test_acc-: **" + str(test_score['score_dict']['accuracy']) +"**")

                loss_noise_tracker.log_loss(model, global_step, device)
                #loss_noise_tracker.log_last_histogram_to_wandb(step=global_step, normalize=True, tag='eval/loss')
            # 经过一些step对train数据集进行测试
            # if self.train_eval(args, global_step):
            #     train_score = self.eval_model_with_both_train_labels(args,model, nl_bucket, device, logger,idx,savePath,fast_mode=args.fast_eval)
            #     logger.info(str(global_step// args.eval_freq) + "-eval/loss/val_c_loss-: " + str(train_score['val_c_loss']))
            #     logger.info(str(global_step// args.eval_freq) +"-eval/loss/val_n_loss-: " + str(train_score['val_n_loss']))
            #     logger.info(str(global_step// args.eval_freq) +"-eval/score/val_c_acc-: ||" + str(train_score['score_dict_c']['accuracy']) +"||")
            #     logger.info(str(global_step// args.eval_freq) +"-eval/score/val_n_acc-: ||" + str(train_score['score_dict_n']['accuracy']) + "||")


            if early_stopper.early_stop:
                test_score = self.eval_model(args, logger, t_loader, model, device, idx,savePath,fast_mode=False)
                logger.info(str(global_step// args.eval_freq) +"-eval/score/test_acc-: " + str(test_score['score_dict']['accuracy']))
                break
        
        # 记录结束时间
        end_time = time.time()

        # 计算运行时间
        execution_time = end_time - start_time

        logger.info(f"end training time: {time.ctime(end_time)}")
        
        print(f" time cost : {execution_time:.2f} 秒")
        logger.info("--------------------------  end of training ----------------------------------")
       
    

    def forward_backward_noisy_batch(self, model, data_dict, loss_fn, args, device):

        nl_databatch = data_dict['nl_batch']
        input_ids = nl_databatch['input_ids']
        attention_mask = nl_databatch['attention_mask']
        n_labels = nl_databatch['n_labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        n_labels = n_labels.to(device)

        # 得到了输出，应该是用pooler_repr比较好
        # 在这里用pooler_repr作为激活，计算Bi tem loss即可
        res = model(input_ids, attention_mask)
        outputs = res['logits'] 

        #loss = loss_fn(outputs, n_labels)
        # bi_tempered_logistic_loss
        loss = bi_tempered_logistic_loss(outputs,F.one_hot(n_labels,self.num_classes),0.5,2)

        sampleIndex =  list(t.item() for t in nl_databatch['index'])
        sampleString = nl_databatch['content']
        sampleFeatures = res['pooler_repr'].tolist()
        # 样本特征向量经过全连接层的预测值,GT的标签，噪声标签，loss和所在的epoch ,step
        sampleLogits = res['logits'].tolist()
        sampleLabel =  list(t.item() for t in nl_databatch['c_labels'])
        sampleNoisyLabel =  list(t.item() for t in nl_databatch['n_labels'])

        sampleSet = [sampleIndex,sampleString,
                    sampleFeatures,sampleLogits,sampleLabel,sampleNoisyLabel]

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()


        return loss.item(),sampleSet


