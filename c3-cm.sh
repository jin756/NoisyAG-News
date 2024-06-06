 CUDA_VISIBLE_DEVICES=1 nohup python main.py \
--dataset AG_News_GT_Sample50000 \
--log_root "./log/" \
--data_root "./data/" \
--trainer_name bert_cm \
--model_name bert-base-uncased \
--gen_val \
--nl_batch_size 32 \
--eval_batch_size 32 \
--gradient_accumulation_steps 1 \
--max_sen_len 64 \
--lr 0.00002 \
--num_training_steps  17500 \
--patience 25 \
--eval_freq 50 \
--store_model 3 \
--noise_level 0.4 \
--noise_type Wmatrix_Ins \
--manualSeed 1234  \
>> bert_cm-Wmatrix_Ins2.log 2>&1 &