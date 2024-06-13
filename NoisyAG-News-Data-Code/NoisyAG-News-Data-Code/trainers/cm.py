import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from models.cm import CM
from transformers import AdamW, get_linear_schedule_with_warmup
from trainers.trainer import Trainer
from tqdm import tqdm
from trainers.early_stopper import EarlyStopper
from trainers.loss_noise_tracker import LossNoiseTracker
import time 
import pickle 
# import wandb 

# Reimplementation of the paper: An Effective Label Noise Model for DNN Text Classification
# Check https://aclanthology.org/N19-1328/
# Note that we use BERT instead of textCNN
#
# 利用混淆矩阵的思想看上去也很简单，就是将输出乘上一个混淆矩阵，然后也将混淆矩阵加入到loss中去，不断地优化混淆矩阵
#
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

class CM_Trainer(Trainer):
    def __init__(self, args, logger, log_dir, model_config, full_dataset, random_state):
        super(CM_Trainer, self).__init__(args, logger, log_dir, model_config, full_dataset, random_state)
        self.mse_weight = args.cm_mse_weight


    def train(self, args, logger, full_dataset):
        # wandb.init(project="bertlnl",name="Bert-wn-nl-" + str(args.noise_level) + "-" + str(args.noise_type),mode = "offline")
        
        start_time = time.time()
        logger.info(f"train start time: {time.ctime(start_time)}")
        savePath = create_folders("./saveData/",args.dataset,args.model_name,args.trainer_name,args.noise_type,args.noise_level)

        logger.info(f"savePath: {savePath}")

        logger.info('Bert CM Trainer: training started...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        nl_set, ul_set, v_set, t_set, l2id, id2l,trainMat,valMat  = full_dataset
        # 这里是先通过通用的方式构造一个基础的模型，然后再基础的模型上去添加混淆矩阵。
        base_model = self.create_model(args)

        cm_model = CM(self.model_config, args, base_model)
        cm_model = cm_model.to(device)

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

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in cm_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in cm_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

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
        saveDataDic = {}

        # train the network
        for idx in tqdm(range(num_training_steps), desc=f'[CM Trainer] training..'):
            ce_loss_mean, l2_loss_mean, ul_ce_loss_mean = 0, 0, 0
            # 得到当前的epoch
            nowEpoch = ((idx + 1) * args.nl_batch_size) // len(nl_set)

            for i in range(args.gradient_accumulation_steps):
                cm_model.train()
                try:
                    nl_batch = next(nl_iter)
                except:
                    nl_iter = iter(nl_bucket)
                    nl_batch = next(nl_iter)

                ce_loss, l2_loss,sampleSet,nMat = \
                    self.forward_backward_cm_batch(cm_model, nl_batch, ce_loss_fn, args, device)
                ce_loss_mean += ce_loss
                l2_loss_mean += l2_loss
                saveKey = str(nowEpoch) + "-" + str(idx)
                saveDataDic[saveKey] = sampleSet

                # 每100次打印出转移矩阵
                if idx % 100 == 0:
                    logger.info(f"\n\n-----------------noise mat is --------------\n\n")
                    logger.info(f"{nMat[0]}")
                    logger.info(f"{nMat[1]}")
                    logger.info(f"{nMat[2]}")
                    logger.info(f"{nMat[3]}")
                # hhf save
                # # 多少个batch保存数据，并且希望同一个
                if idx %320  == 0 and idx != 0 and args.saveData:
                
                    with open(savePath+"/train/train_" +args.dataset+"_" + args.model_name+ "_" + args.trainer_name +"_" + str(idx) + ".pkl","wb") as f:
                        pickle.dump(saveDataDic,f)
                    #np.save(savePath+"/" +args.dataset+"_" + args.model_name+ "_" + args.trainer_name +str(idx) + ".npy",saveDataDic)
                    # 把保存的数据置空
                    saveDataDic = {}
                else:
                    saveDataDic = {}


            torch.nn.utils.clip_grad_norm_(cm_model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer_scheduler.step()  # Update learning rate schedule
            cm_model.zero_grad()
            global_step += 1

            # wandb.log({'train/batch_loss': ce_loss_mean})
            logger.info(str(idx) + "-ce_loss_mean-: " + str(ce_loss_mean))
            if self.train_eval_fre(args, global_step):
                # val_score = self.eval_model_with_both_labels(base_model, v_loader, device, fast_mode=args.fast_eval)
                # test_score = self.eval_model(args, logger, t_loader, base_model, device, fast_mode=args.fast_eval)

                val_score = self.eval_model_with_both_labels(args,base_model, v_loader, device, logger,idx,savePath,fast_mode=args.fast_eval)
                
                test_score = self.eval_model(args, logger, t_loader, base_model, device, idx,savePath,fast_mode=args.fast_eval)
                
                early_stopper.register(val_score['score_dict_n']['accuracy'], base_model, optimizer)

                logger.info(
                    str(global_step // args.eval_freq) + "-eval/loss/val_c_loss-: " + str(val_score['val_c_loss']))
                logger.info(
                    str(global_step // args.eval_freq) + "-eval/loss/val_n_loss-: " + str(val_score['val_n_loss']))
                logger.info(str(global_step // args.eval_freq) + "-eval/score/val_c_acc-: &&" + str(
                    val_score['score_dict_c']['accuracy']) +'&&')
                logger.info(str(global_step // args.eval_freq) + "-eval/score/val_n_acc-: &&" + str(
                    val_score['score_dict_n']['accuracy']) +'&&')
                logger.info(str(global_step // args.eval_freq) + "-eval/score/test_acc-: **" + str(
                    test_score['score_dict']['accuracy']) +'**')

                loss_noise_tracker.log_loss(base_model, global_step, device)
                #loss_noise_tracker.log_last_histogram_to_wandb(step=global_step, normalize=True, tag='eval/loss')
            
            # 经过一些step对train数据集进行测试
            # hhf save
            if self.train_eval_fre(args, global_step):
                train_score = self.eval_model_with_both_train_labels(args,base_model, nl_bucket, device, logger,idx,savePath,fast_mode=args.fast_eval)
                logger.info(str(global_step// args.eval_freq) + "-eval/loss/val_c_loss-: " + str(train_score['val_c_loss']))
                logger.info(str(global_step// args.eval_freq) +"-eval/loss/val_n_loss-: " + str(train_score['val_n_loss']))
                logger.info(str(global_step// args.eval_freq) +"-eval/score/val_c_acc-: ||" + str(train_score['score_dict_c']['accuracy']) + "||")
                logger.info(str(global_step// args.eval_freq) +"-eval/score/val_n_acc-: ||" + str(train_score['score_dict_n']['accuracy']) +"||")


            if early_stopper.early_stop:
                
                test_score = self.eval_model(args, logger, t_loader, base_model, device, idx,savePath,fast_mode=False)
                logger.info(str(global_step// args.eval_freq) +"-eval/score/test_acc-: " + str(test_score['score_dict']['accuracy']))

                break
        logger.info("--------------------------  end of training ----------------------------------")
        # 记录结束时间
        end_time = time.time()

        # 计算运行时间
        execution_time = end_time - start_time

        logger.info(f"end training time: {time.ctime(end_time)}")

        print(f" time cost : {execution_time:.2f} 秒")
        # if args.save_loss_tracker_information:
        #     loss_noise_tracker.save_logged_information()
        #     self.logger.info("[Vanilla Trainer]: loss history saved")
        # best_model = self.create_model(args)
        # best_model_weights = early_stopper.get_final_res()["es_best_model"]
        # best_model.load_state_dict(best_model_weights)
        # best_model = best_model.to(device)

        val_score = self.eval_model_with_both_labels(args,base_model, v_loader, device, logger,idx,savePath,fast_mode=args.fast_eval)
                
        test_score = self.eval_model(args, logger, t_loader, base_model, device, idx,savePath,fast_mode=args.fast_eval)
                
        # wandb.run.summary["best_score_on_val_n"] = test_score['score_dict']['accuracy']
        # wandb.run.summary["best_val_n"] = val_score['score_dict_n']['accuracy']
        # wandb.run.summary["best_val_c_on_val_n"] = val_score['score_dict_c']['accuracy']


    def forward_backward_cm_batch(self, cm_model, nl_databatch, loss_fn, args, device):
        # https://huggingface.co/transformers/custom_datasets.html

        input_ids = nl_databatch['input_ids']
        attention_mask = nl_databatch['attention_mask']
        n_labels = nl_databatch['n_labels']
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        n_labels = n_labels.to(device)
        res = cm_model(input_ids, attention_mask)
        outputs = res['logits']

        ce_loss = loss_fn(outputs, n_labels)
        l2_loss = self.mse_weight * torch.sum(cm_model.noise_model ** 2)
        total_loss = (ce_loss + l2_loss)/args.gradient_accumulation_steps
        total_loss.backward()
        sampleLoss = total_loss 
        

        sampleIndex = list(t.item() for t in nl_databatch['index'])
        sampleString = nl_databatch['content']
        sampleFeatures = res['pooler_repr'].tolist() 
        sampleLogits = res['logits'].tolist()
        sampleLabel = list(t.item() for t in nl_databatch['c_labels'])
        sampleNoisyLabel = list(t.item() for t in nl_databatch['n_labels'])
        sampleSet = [sampleIndex,sampleString,sampleFeatures,sampleLogits,sampleLabel,sampleNoisyLabel,sampleLoss]

        return ce_loss.item(), l2_loss.item(),sampleSet,res['mat']
