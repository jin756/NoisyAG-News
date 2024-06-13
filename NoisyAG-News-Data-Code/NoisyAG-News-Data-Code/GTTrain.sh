#!/bin/bash
# 定义模型名称列表  xlnet-base-cased roberta-base bert-base-uncased  bart-base  deberta-v3-base
modelNames=("bart-base")

# 定义方法列表
methods=(  'ct' ) 

# 定义数据集列表
#dataSets=("gt" "best" "mid" "worst")
dataSets=(  "GtSample50000" )

# 定义固定参数
noiseType="NoneNoise"
NoiseLevel="0.0"

CUDA_DEVICE=5
mySeed=2345

# 循环遍历所有组合
for modelName in "${modelNames[@]}"; do
  for method in "${methods[@]}"; do
    for dataSet in "${dataSets[@]}"; do
      logFile="./pLog/${modelName}/${method}/${dataSet}_${noiseType}_${NoiseLevel}_${mySeed}.log"
      
      echo "Running modelName=${modelName}, method=${method}, dataSet=${dataSet} device ${CUDA_DEVICE}"

      if [ "$method" == "ls" ] && [ "$dataSet" == "worst" ]; then
        smoothing_factor=0.38
      elif [ "$method" == "ls" ] && [ "$dataSet" == "mid" ]; then
        smoothing_factor=0.2
      elif [ "$method" == "ls" ] && [ "$dataSet" == "best" ]; then
        smoothing_factor=0.1
      elif [ "$method" == "nls" ] && [ "$dataSet" == "worst" ]; then
        smoothing_factor=-0.38
      elif [ "$method" == "nls" ] && [ "$dataSet" == "mid" ]; then
        smoothing_factor=-0.2
      elif [ "$method" == "nls" ] && [ "$dataSet" == "best" ]; then
        smoothing_factor=-0.1
      elif [ "$dataSet" == "gt" ]; then
        smoothing_factor=0.0
      else
        smoothing_factor=0.0
      fi
        
      echo "smoothing_factor is ${smoothing_factor}"

      CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup python NoisyAG_News.py \
      --saveData \
      --dataset $dataSet \
      --log_root "./log/" \
      --data_root "./data/" \
      --trainer_name $method \
      --model_name $modelName \
      --gen_val \
      --nl_batch_size 32 \
      --eval_batch_size 32 \
      --gradient_accumulation_steps 1 \
      --max_sen_len 64 \
      --lr 0.00002 \
      --num_training_steps 8000 \
      --patience 36 \
      --eval_freq 100 \
      --store_model 0 \
      --noise_level $NoiseLevel \
      --noise_type $noiseType \
      --smoothing_factor $smoothing_factor \
      --manualSeed $mySeed \
      >> ${logFile} 2>&1 &
      # 等待一小段时间以确保脚本开始运行
      sleep 3

      
    done
  done
done