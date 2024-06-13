import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm
from trainers.trainer import Trainer
#import wandb
from trainers.early_stopper import EarlyStopper
from trainers.loss_noise_tracker import LossNoiseTracker
import os
import time
import pickle 

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

# Reimplementation of the paper: Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels
# Check https://papers.nips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html
# Note that we use the ground truth noise level, to eliminate the influence of the estimation errors

class CT_Trainer(Trainer):
    def __init__(self, args, logger, log_dir, model_config, full_dataset, random_state):
        super(CT_Trainer, self).__init__(args, logger, log_dir, model_config, full_dataset, random_state)

    def train(self, args, logger, full_dataset):
        #wandb.init(project="bertlnl",name="Bert-wn-nl-" + str(args.noise_level) + "-" + str(args.noise_type),mode = "offline")
        
        logger.info('Bert CT Trainer: training started')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_time = time.time()
        logger.info(f"train start time: {time.ctime(start_time)}")
        savePath = create_folders("./saveData/",args.dataset,args.model_name,args.trainer_name,args.noise_type,args.noise_level)

        logger.info(f"savePath: {savePath}")
        saveDataDic = {}

        nl_set, ul_set, v_set, t_set, l2id, id2l,trainMat,valMat = full_dataset


        # 构造两个模型
        model1 = self.create_model(args)
        model1 = model1.to(device)
        model2 = self.create_model(args)
        model2 = model2.to(device)

        nl_bucket = torch.utils.data.DataLoader(nl_set, batch_size=args.nl_batch_size, shuffle=False, num_workers=0,drop_last = True)

        nl_iter = iter(nl_bucket)

        if v_set is None:
            logger.info('No validation set is used here')
            v_loader = None
        else:
            logger.info('Validation set is used here')
            v_loader = torch.utils.data.DataLoader(v_set, batch_size=args.eval_batch_size,
                                                   shuffle=False, 
                                                   num_workers=0,drop_last = True)

        t_loader = torch.utils.data.DataLoader(t_set, batch_size=args.eval_batch_size,
                                               shuffle=True,
                                               num_workers=0,drop_last = True)

        num_training_steps = args.num_training_steps
        optimizer1, optimizer_scheduler1 = self.get_optimizer(model1, args, num_training_steps)
        optimizer2, optimizer_scheduler2 = self.get_optimizer(model2, args, num_training_steps)




        noise_level = args.noise_level
        forget_rate = noise_level * args.forget_factor
        rate_schedule = np.ones(num_training_steps) * forget_rate
        rate_schedule[:args.T_k] = np.linspace(0, forget_rate ** args.c, args.T_k)
        logger.info(f"Total Steps: {num_training_steps} ,T_k: {args.T_k}")


        if self.store_model_flag:
            early_stopper_save_dir = os.path.join(self.log_dir, 'early_stopper_model')
            if not os.path.exists(early_stopper_save_dir):
                os.makedirs(early_stopper_save_dir)
        else:
            early_stopper_save_dir = None

        early_stopper = EarlyStopper(patience=args.patience, delta=0, save_dir=early_stopper_save_dir,
                                     large_is_better=True, verbose=False, trace_func=logger.info)

        noise_tracker_dir = os.path.join(self.log_dir, 'loss_noise_tracker')
        loss_noise_tracker = LossNoiseTracker(args, logger, nl_set, noise_tracker_dir)


        global_step = 0

        # train the network
        for idx in tqdm(range(num_training_steps), desc=f'training'):
            # 得到当前的epoch
            nowEpoch = ((idx + 1) * args.nl_batch_size) // len(nl_set)
            try:
                nl_batch = next(nl_iter)
            except:
                nl_iter = iter(nl_bucket)
                nl_batch = next(nl_iter)

            loss1, loss2,sampleSet = \
                self.forward_path_for_sorting_loss(model1, model2, nl_batch,
                                                   args, device)
            
            

            ce_loss1, ce_loss2, purity1, purity2,recordBatch1,recordBatch2 = \
                self.do_coteaching(nl_batch, (model1, model2),
                                   (optimizer1, optimizer2),
                                   (optimizer_scheduler1, optimizer_scheduler2),
                                   (loss1, loss2),
                                   rate_schedule[global_step],
                                   args, device)
            global_step += 1
            saveKey = str(nowEpoch) + "-" + str(idx)
            saveDataDic[saveKey] = [sampleSet,recordBatch1,recordBatch2]
            # 多少个batch保存数据，并且希望同一个

            # hhf:save Flag
            if idx %320  == 0 and idx != 0 :
            
                with open(savePath+"/train/train_" +args.dataset+"_" + args.model_name+ "_" + args.trainer_name +"_" + str(idx) + ".pkl","wb") as f:
                    pickle.dump(saveDataDic,f)
                #np.save(savePath+"/" +args.dataset+"_" + args.model_name+ "_" + args.trainer_name +str(idx) + ".npy",saveDataDic)
                # 把保存的数据置空
                saveDataDic = {}


            logger.info(str(idx) + "-ce_loss_mean-: " + str(ce_loss1) + "-" + str(ce_loss2))


            if self.needs_eval(args, global_step):
                #val_score = self.eval_model_with_both_labels(model1, v_loader, device, fast_mode=args.fast_eval)
                # val_score = self.eval_model_with_both_labels(args,model1, v_loader, device, logger,idx,savePath,fast_mode=args.fast_eval)

                # test_score = self.eval_model(args, logger, t_loader, model1, device, fast_mode=args.fast_eval)

                val_score = self.eval_model_with_both_labels(args,model1, v_loader, device, logger,idx,savePath,fast_mode=args.fast_eval)
                
                test_score = self.eval_model(args, logger, t_loader, model1, device, idx,savePath,fast_mode=args.fast_eval)

                early_stopper.register(val_score['score_dict_n']['accuracy'], model1, optimizer1)

               
                logger.info(
                    str(global_step // args.eval_freq) + "-eval/loss/val_c_loss-: " + str(val_score['val_c_loss']))
                logger.info(
                    str(global_step // args.eval_freq) + "-eval/loss/val_n_loss-: " + str(val_score['val_n_loss']))
                logger.info(str(global_step // args.eval_freq) + "-eval/score/val_c_acc-: &&" + str(
                    val_score['score_dict_c']['accuracy']) +'&&')
                logger.info(str(global_step // args.eval_freq) + "-eval/score/val_n_acc-: &&" + str(
                    val_score['score_dict_n']['accuracy']) + '&&')
                logger.info(str(global_step // args.eval_freq) + "-eval/score/test_acc-: **" + str(
                    test_score['score_dict']['accuracy']) + '**')

                loss_noise_tracker.log_loss(model1, global_step, device)
                #loss_noise_tracker.log_last_histogram_to_wandb(step=global_step, normalize=True, tag='eval/loss')
            # 经过一些step对train数据集进行测试
            # # hhf:save Flag
            if self.train_eval(args, global_step):
                train_score = self.eval_model_with_both_train_labels(args,model1, nl_bucket, device, logger,idx,savePath,fast_mode=args.fast_eval)
                logger.info(str(global_step// args.eval_freq) + "-eval/loss/val_c_loss-: " + str(train_score['val_c_loss']))
                logger.info(str(global_step// args.eval_freq) +"-eval/loss/val_n_loss-: " + str(train_score['val_n_loss']))
                logger.info(str(global_step// args.eval_freq) +"-eval/score/val_c_acc-: ||" + str(train_score['score_dict_c']['accuracy']) +"||")
                logger.info(str(global_step// args.eval_freq) +"-eval/score/val_n_acc-: ||" + str(train_score['score_dict_n']['accuracy']) +"||")

            if early_stopper.early_stop:
                test_score = self.eval_model(args, logger, t_loader, model1, device, idx,savePath,fast_mode=False)
                logger.info(str(global_step// args.eval_freq) +"-eval/score/test_acc-: " + str(test_score['score_dict']['accuracy']))
                break
        
        # 记录结束时间
        end_time = time.time()
        # 计算运行时间
        execution_time = end_time - start_time
        logger.info(f"end training time: {time.ctime(end_time)}")
        print(f" time cost : {execution_time:.2f} 秒")
        logger.info("--------------------------  end of training ----------------------------------")  



    def train_batch(self, model, data_batch, optimizer, optimizer_scheduler, args, device):

        total_loss = 0

        input_ids_batch = data_batch['input_ids']
        attention_mask_batch = data_batch['attention_mask']
        n_labels_batch = data_batch['n_labels']

        num_samples_in_batch = len(input_ids_batch)
        input_ids_batch = input_ids_batch.to(device)
        attention_mask_batch = attention_mask_batch.to(device)
        n_labels_batch = n_labels_batch.to(device)

        num_batches = int(np.ceil(num_samples_in_batch / args.nl_batch_size))

        model.zero_grad()
        for i in range(num_batches):
            start = i * args.nl_batch_size
            end = start + args.nl_batch_size
            input_ids = input_ids_batch[start:end]
            attention_mask = attention_mask_batch[start:end]
            n_labels = n_labels_batch[start:end]

            outputs = model(input_ids, attention_mask)['logits']
            loss = F.cross_entropy(outputs, n_labels, reduction='sum')
            loss = loss / num_samples_in_batch
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer_scheduler.step()  # Update learning rate schedule

        return total_loss

    def do_coteaching(self, data_batch, models, optimizers, optimizer_schedulers,
                      losses, forget_rate, args, device):

        model1, model2 = models
        optimizer1, optimizer2 = optimizers
        loss1, loss2 = losses
        optimizer_scheduler1, optimizer_scheduler2 = optimizer_schedulers

        remember_rate = 1 - forget_rate

        filtered_data1, purity1,recordBatch1 = self.filter_data(data_batch, loss1, remember_rate, args)
        filtered_data2, purity2,recordBatch2 = self.filter_data(data_batch, loss2, remember_rate, args)

        # 互相用过滤的数据去训练
        loss1 = self.train_batch(model1, filtered_data2, optimizer1, optimizer_scheduler1, args, device)
        loss2 = self.train_batch(model2, filtered_data1, optimizer2, optimizer_scheduler2, args, device)

        return loss1, loss2, purity1, purity2,recordBatch1,recordBatch2

    def filter_data(self, data_batch, loss, remember_rate, args):
        # 选择出loss比较小的样本，返回其特征和纯度（即和真实标签相等的概率）

        input_ids = data_batch['input_ids']
        attention_mask = data_batch['attention_mask']
        n_labels = data_batch['n_labels']
        dataIndex = data_batch['index']

        # 这里对loss进行排序，得到小的
        _, sort_idx = torch.sort(loss)
        # 
        sort_idx = sort_idx[0:int(len(sort_idx) * remember_rate)]

        recordBatch = []
        for i in range(len(sort_idx)):
            recordBatch.append(dataIndex[sort_idx[i]])


        purity_selected = data_batch['purity'][sort_idx]
        # 计算纯度，就是有多少和真实的标签相等
        purity = torch.sum(purity_selected).true_divide(len(purity_selected))

        return {'input_ids': input_ids[sort_idx], 'attention_mask': attention_mask[sort_idx],
                'n_labels': n_labels[sort_idx]}, purity,recordBatch

    def forward_path_for_sorting_loss(self, model1, model2, data_batch, args, device):

        model1.eval()
        model2.eval()

        input_ids = data_batch['input_ids']
        attention_mask = data_batch['attention_mask']
        n_labels = data_batch['n_labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        n_labels = n_labels.to(device)

        with torch.no_grad():
            res1 = model1(input_ids, attention_mask)
            output1 = res1['logits']
            loss1 = F.cross_entropy(output1, n_labels, reduction='none')
            sampleFeatures1 = res1['pooler_repr'].tolist()
            # 样本特征向量经过全连接层的预测值,GT的标签，噪声标签，loss和所在的epoch ,step
            sampleLogits1 = res1['logits'].tolist()

            res2 = model2(input_ids, attention_mask)
            output2 = res2['logits']
            loss2 = F.cross_entropy(output2, n_labels, reduction='none')
            sampleFeatures2 = res2['pooler_repr'].tolist() 
            # 样本特征向量经过全连接层的预测值,GT的标签，噪声标签，loss和所在的epoch ,step
            sampleLogits2 = res2['logits']
            

        sampleIndex =  list(t.item() for t in data_batch['index'])
        sampleString = data_batch['content']
        sampleLabel =  list(t.item() for t in data_batch['c_labels'])
        sampleNoisyLabel =  list(t.item() for t in data_batch['n_labels'])


        sampleSet = [sampleIndex,sampleString,
                     sampleFeatures1,sampleLogits1,sampleFeatures2,sampleLogits2,sampleLabel,sampleNoisyLabel]

        model1.train()
        model2.train()

        return loss1.detach().cpu(), loss2.detach().cpu() ,sampleSet 

    def get_optimizer(self, model, args, num_training_steps):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        optimizer_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                              num_training_steps=num_training_steps)
        return optimizer, optimizer_scheduler